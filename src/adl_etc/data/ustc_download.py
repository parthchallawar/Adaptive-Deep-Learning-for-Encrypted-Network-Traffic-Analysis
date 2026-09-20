"""USTC-TFC2016 acquisition (spec 001, D4): the "unusual traffic" anomaly
corpus. Fully automated — the one D3/D4 dataset that needs no registration.

Lists ``Benign/`` and ``Malware/`` in the public GitHub mirror
``davidyslu/USTC-TFC2016`` via the contents API, downloads every ``.pcap``
directly and every ``.7z`` archive then extracts its member(s) (some archives
hold one top-level file named after the class, others hold a subfolder with
several numbered files — both shapes are real, see
:func:`adl_etc.data.download.extract_7z`), and writes ``labels.csv`` mapping
file name to one of the 10 benign application names or 10 malware family
names (spec 001: "10 benign + 10 malware").
"""

from __future__ import annotations

import csv
import fnmatch
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from adl_etc.data import manifest as M
from adl_etc.data.download import GithubFile, download_file, extract_7z, list_github_dir
from adl_etc.utils.provenance import file_sha256

OWNER = "davidyslu"
REPO = "USTC-TFC2016"
REF = "master"
GITHUB_DIRS = {"Benign": "benign", "Malware": "malware"}
LABELS_COLUMNS = ("file_name", "class_name", "category")


@dataclass(frozen=True, slots=True)
class PlanItem:
    category: str
    class_name: str
    gh_file: GithubFile


def plan(*, files_glob: str | None = None, api_base: str | None = None) -> list[PlanItem]:
    """What would be downloaded: one entry per archive/file in Benign/ and
    Malware/, each already carrying its inferred class name (the archive's
    own base name — "BitTorrent.7z" and "BitTorrent.pcap" both -> class
    "BitTorrent", exactly the file-level ground truth spec 001 describes)."""
    kwargs = {"api_base": api_base} if api_base else {}
    out: list[PlanItem] = []
    for gh_dir, category in GITHUB_DIRS.items():
        for f in list_github_dir(OWNER, REPO, gh_dir, ref=REF, **kwargs):
            if files_glob and not fnmatch.fnmatch(f.name, files_glob):
                continue
            out.append(PlanItem(category=category, class_name=Path(f.name).stem, gh_file=f))
    return out


def run(
    *,
    out_dir: Path,
    manifest_path: Path = M.DEFAULT_MANIFEST_PATH,
    files_glob: str | None = None,
    dry_run: bool = False,
    api_base: str | None = None,
    log: Callable[[str], None] = print,
) -> int:
    """Returns the number of pcap files present in ``out_dir/pcap`` on exit
    (0 on a dry run). Idempotent: files/archives already fully downloaded are
    detected and not re-fetched (see the per-branch checks below)."""
    items = plan(files_glob=files_glob, api_base=api_base)
    total_bytes = sum(i.gh_file.bytes for i in items)
    log(f"USTC-TFC2016: {len(items)} archives/files, {total_bytes / 1e6:.1f} MB compressed")
    for i in items:
        mb = i.gh_file.bytes / 1e6
        log(f"  {i.category:8s} {i.class_name:18s} {i.gh_file.name:20s} {mb:7.2f} MB")
    if dry_run:
        return 0

    pcap_dir = out_dir / "pcap"
    pcap_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = out_dir / "_archives"

    labels: list[dict[str, str]] = []
    manifest_files: list[dict[str, object]] = []
    for item in items:
        f = item.gh_file
        if f.name.lower().endswith(".7z"):
            # Idempotency heuristic: if extraction already produced files for
            # this class, trust them rather than re-downloading a multi-MB
            # archive. Not airtight (a half-finished extraction would look
            # complete), but --dry-run plus the printed plan makes a stale
            # partial state easy to spot before it's relied on.
            existing = sorted(pcap_dir.glob(f"{item.class_name}*.pcap"))
            if existing:
                pcaps = existing
            else:
                archive_dir.mkdir(parents=True, exist_ok=True)
                archive_dest = archive_dir / f.name
                log(f"downloading {f.name} ...")
                download_file(f.download_url, archive_dest, expected_bytes=f.bytes)
                pcaps = extract_7z(archive_dest, pcap_dir)
                archive_dest.unlink()
        else:
            dest = pcap_dir / f.name
            log(f"downloading {f.name} ...")
            download_file(f.download_url, dest, expected_bytes=f.bytes)
            pcaps = [dest]

        for p in pcaps:
            labels.append(
                {"file_name": p.name, "class_name": item.class_name, "category": item.category}
            )
            manifest_files.append(
                {"name": p.name, "bytes": p.stat().st_size, "sha256": file_sha256(p)}
            )

    if archive_dir.exists() and not any(archive_dir.iterdir()):
        archive_dir.rmdir()

    labels_path = out_dir / "labels.csv"
    with open(labels_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(LABELS_COLUMNS))
        writer.writeheader()
        writer.writerows(labels)

    M.register(
        "ustc-tfc2016",
        source_url=f"https://github.com/{OWNER}/{REPO}",
        files=manifest_files,
        extracted_to=str(pcap_dir),
        manifest_path=manifest_path,
    )
    log(f"done: {len(labels)} pcaps, labels at {labels_path}")
    return len(labels)
