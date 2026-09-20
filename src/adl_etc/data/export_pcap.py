"""PCAP-to-shard export (spec 002, plan T4): joins ``pcap_source`` -> ``flows``
-> ``tensors`` into the project's first real artefact.

One capture file becomes one ``session_id`` (spec 004's grouped split for D3
needs no flow to cross a file boundary), labelled at file level from
``labels.csv`` (written by the D3/D4 downloaders, spec 001). A fresh
:class:`~adl_etc.data.flows.FlowBuilder` is used per file — flow state must
never leak across captures.
"""

from __future__ import annotations

import csv
import fnmatch
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adl_etc.data import manifest as M
from adl_etc.data import ppi as P
from adl_etc.data.flows import FlowBuilder
from adl_etc.data.pcap_source import PcapStats, parse_pcap
from adl_etc.data.tensors import ShardWriter
from adl_etc.utils.provenance import stable_hash

LABELS_FILENAME = "labels.csv"
PCAP_SUBDIR = "pcap"


@dataclass(frozen=True, slots=True)
class PcapConfig:
    """Loaded from ``configs/data/pcap.yaml``. Field names and defaults match
    :class:`~adl_etc.data.flows.FlowBuilder`'s own constructor exactly."""

    inactive_timeout: float = 30.0
    active_timeout: float = 300.0
    fin_linger: float = 5.0
    udp_idle_timeout: float | None = None
    k_max: int = P.K_MAX
    backend: str = "dpkt"

    @classmethod
    def load(cls, path: str | Path | None) -> PcapConfig:
        """``path=None`` returns the defaults (which is also what an absent
        config file means for every downstream caller)."""
        if path is None:
            return cls()
        from omegaconf import OmegaConf

        raw = OmegaConf.to_container(OmegaConf.load(path))
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a mapping at the top level")
        # Untrusted external input (YAML): kept as `Any` and explicitly
        # coerced field by field below, rather than typed as if it were
        # already validated.
        fields: dict[str, Any] = {str(k): v for k, v in raw.items()}
        unknown = set(fields) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"{path}: unknown config key(s) {sorted(unknown)}")

        defaults = cls()  # slots=True means field defaults aren't class attributes
        udp_idle = fields.get("udp_idle_timeout", defaults.udp_idle_timeout)
        return cls(
            inactive_timeout=float(fields.get("inactive_timeout", defaults.inactive_timeout)),
            active_timeout=float(fields.get("active_timeout", defaults.active_timeout)),
            fin_linger=float(fields.get("fin_linger", defaults.fin_linger)),
            udp_idle_timeout=None if udp_idle is None else float(udp_idle),
            k_max=int(fields.get("k_max", defaults.k_max)),
            backend=str(fields.get("backend", defaults.backend)),
        )


@dataclass
class ExportSummary:
    dataset: str
    n_files: int
    n_files_skipped_unlabeled: int
    n_flows: int
    dropped_zero_ppi: int
    skipped_packets: dict[str, int] = field(default_factory=dict)
    bytes_processed: int = 0
    packets_processed: int = 0
    wall_seconds: float = 0.0

    @property
    def packets_per_second(self) -> float:
        return self.packets_processed / self.wall_seconds if self.wall_seconds > 0 else 0.0

    @property
    def mb_per_second(self) -> float:
        return (self.bytes_processed / 1e6) / self.wall_seconds if self.wall_seconds > 0 else 0.0

    def render(self) -> str:
        header = f"{self.dataset}: {self.n_files} files exported"
        if self.n_files_skipped_unlabeled:
            header += f" ({self.n_files_skipped_unlabeled} skipped, unlabeled)"
        lines = [
            header,
            f"  flows: {self.n_flows} written, {self.dropped_zero_ppi} dropped (zero PPI)",
            f"  packets: {self.packets_processed} in {self.wall_seconds:.1f}s"
            f" ({self.packets_per_second:,.0f} pkt/s, {self.mb_per_second:.1f} MB/s)",
        ]
        if self.skipped_packets:
            skipped = ", ".join(f"{k}={v}" for k, v in self.skipped_packets.items() if v)
            if skipped:
                lines.append(f"  skipped packets: {skipped}")
        return "\n".join(lines)


def _read_labels(labels_path: Path) -> dict[str, tuple[str, str]]:
    """``file_name -> (class_name, category)``. Rows with an empty
    ``class_name`` (an unresolved ISCX heuristic label, spec 001) are
    excluded entirely: an empty label is not the same thing as the
    project's ``-1`` "held-out unknown class" and must not be conflated
    with it."""
    out: dict[str, tuple[str, str]] = {}
    with open(labels_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            class_name = row.get("class_name", "").strip()
            if not class_name:
                continue
            out[row["file_name"]] = (class_name, row.get("category", "").strip())
    return out


def export_dataset(
    *,
    dataset: str,
    raw_root: str | Path = Path("data/raw"),
    out_root: str | Path = Path("data/processed"),
    period: str = "all",
    config: PcapConfig | None = None,
    files_glob: str | None = None,
    manifest_path: str | Path = M.DEFAULT_MANIFEST_PATH,
    overwrite: bool = False,
    log: Callable[[str], None] = print,
) -> ExportSummary:
    """Exports every labelled pcap in ``raw_root/<dataset>/pcap/`` to shards
    under ``out_root/<dataset>/<period>/``. Returns a summary with the
    throughput numbers the plan's 60-minute gate is measured against.
    """
    config = config or PcapConfig()
    if config.backend != "dpkt":
        raise NotImplementedError(
            f"backend {config.backend!r} is not implemented; only 'dpkt' "
            "(spec 002's default, pure-Python) is available"
        )

    raw_root = Path(raw_root)
    dataset_dir = raw_root / dataset
    pcap_dir = dataset_dir / PCAP_SUBDIR
    labels = _read_labels(dataset_dir / LABELS_FILENAME)

    all_pcaps = sorted(pcap_dir.glob("*.pcap"))
    labelled = [p for p in all_pcaps if p.name in labels]
    n_skipped_unlabeled = len(all_pcaps) - len(labelled)
    files = labelled
    if files_glob:
        files = [p for p in files if fnmatch.fnmatch(p.name, files_glob)]

    class_names = sorted({labels[p.name][0] for p in files})
    label_map = {name: i for i, name in enumerate(class_names)}
    category_names = sorted({labels[p.name][1] for p in files})
    category_map = {name: i for i, name in enumerate(category_names)}

    source_manifest_hash = None
    try:
        entry = M.require(dataset, manifest_path=manifest_path)
        source_manifest_hash = stable_hash({"files": entry["files"]})
    except KeyError:
        pass  # no manifest entry yet is not fatal here; the shard set just won't carry one

    pkt_stats = PcapStats()
    packets_processed = 0
    bytes_processed = sum(p.stat().st_size for p in files)

    t0 = time.perf_counter()
    with ShardWriter(
        out_root,
        dataset,
        period,
        label_map=label_map,
        category_map=category_map,
        source_manifest_hash=source_manifest_hash,
        overwrite=overwrite,
    ) as writer:
        for session_id, path in enumerate(files):
            class_name, category = labels[path.name]
            label = label_map[class_name]
            cat_id = category_map[category]
            builder = FlowBuilder(
                inactive_timeout=config.inactive_timeout,
                active_timeout=config.active_timeout,
                fin_linger=config.fin_linger,
                k_max=config.k_max,
                session_id=session_id,
                udp_idle_timeout=config.udp_idle_timeout,
            )
            for pkt in parse_pcap(path, stats=pkt_stats):
                packets_processed += 1
                for rec in builder.add(pkt):
                    writer.add(rec, label=label, category=cat_id)
            for rec in builder.flush():
                writer.add(rec, label=label, category=cat_id)
        meta = writer.close()
    wall = time.perf_counter() - t0

    summary = ExportSummary(
        dataset=dataset,
        n_files=len(files),
        n_files_skipped_unlabeled=n_skipped_unlabeled,
        n_flows=meta["n_flows"],
        dropped_zero_ppi=meta["counters"].get("dropped_zero_ppi", 0),
        skipped_packets=pkt_stats.as_dict(),
        bytes_processed=bytes_processed,
        packets_processed=packets_processed,
        wall_seconds=wall,
    )
    log(summary.render())
    return summary
