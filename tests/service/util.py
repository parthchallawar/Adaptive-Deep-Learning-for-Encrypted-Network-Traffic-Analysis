"""Torch-free helpers for the service tests: a deterministic causal stub model, random flows,
a synthetic D1-like period on disk, and a multi-flow pcap writer."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import dpkt
import numpy as np

from adl_etc.data import ppi as P
from adl_etc.inference.drift import SIGNALS
from adl_etc.service.engine import FlowSpec

N_CLASSES = 5
K = P.K_MAX


class StubScorer:
    """Scores that depend only on a flow's packets up to each position (so it is causal, like
    the PAT): ``p_safe`` rises with ``k`` at a flow-specific speed, ``u`` is high for some flows,
    and the class is a function of the running byte count."""

    run_dir = "stub"

    def __init__(self, version: str = "stub-a", shift: int = 0) -> None:
        self.version = version
        self.checkpoint_hash = version
        self.class_names = [f"class{i}" for i in range(N_CLASSES)]
        self.known_ids = set(range(N_CLASSES))
        self.shift = shift
        self.calls = 0
        self.reference = stub_reference()
        self.production = {"offline": {}}

    def score(
        self, ppi: np.ndarray, ppi_len: np.ndarray, k_from: np.ndarray
    ) -> dict[str, np.ndarray]:
        self.calls += 1
        ppi_len = np.asarray(ppi_len)
        sizes = ppi[:, :, P.SIZE_POS].astype(np.float64)
        j = np.arange(K)[None, :]
        first = sizes[:, :1]
        cum = np.cumsum(sizes, axis=1)
        tau = 1.0 + (first % 7)
        p_safe = 1.0 - 0.6 * np.exp(-(j + 1) / tau)
        u = ((first % 11) - 6.0) / 2.0 + 0.04 * j + 0.001 * (cum % 13)
        pred = ((cum // 97).astype(np.int64) + self.shift) % N_CLASSES
        live = j < ppi_len[:, None]
        out: dict[str, np.ndarray] = {
            "p_safe": p_safe,
            "u": u,
            "conf": p_safe,
            "energy": -u,
            "mahal": u / 2,
            "surprise": 1.0 + 0.01 * (cum % 17),
        }
        for name in out:
            out[name] = np.where(live, out[name], np.nan)
        out["pred"] = np.where(live, pred, 0)
        return out

    def theta_reject(self, f: float) -> np.ndarray:
        return np.full(K, 2.0 - 10.0 * f)


def stub_reference() -> dict[str, Any]:
    ref = {s: {"mean": 0.5, "std": 0.1} for s in (*SIGNALS, "theta_commit")}
    return {
        "theta_commit": 0.8,
        "f": 0.02,
        "drift_window": 50,
        "drift_static": {"signals": list(SIGNALS), "reference": ref, "h": 5.0},
        "drift_controller": {
            "signals": ["reject_rate", "mean_safe_at_commit", "mean_pkt_nll", "prior_kl"],
            "reference": ref,
            "h": 5.0,
        },
        "reference_hist": [10.0] * N_CLASSES,
    }


def random_flows(n: int, seed: int, *, min_len: int = 0, max_len: int = K) -> list[FlowSpec]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        n_pkts = int(rng.integers(min_len, max_len + 1))
        ppi = np.zeros((K, P.PPI_CHANNELS), dtype=np.int16)
        ppi[:n_pkts, P.SIZE_POS] = rng.integers(1, 1500, n_pkts)
        ppi[:n_pkts, P.IPT_POS] = rng.integers(0, 50, n_pkts)
        ppi[:n_pkts, P.DIR_POS] = rng.choice([P.DIR_FWD, P.DIR_REV], n_pkts)
        ppi[:n_pkts, P.PUSH_POS] = rng.integers(0, 2, n_pkts)
        if n_pkts:
            ppi[0, P.IPT_POS] = 0
        out.append(FlowSpec(key=f"f{seed}-{i}", session_id=1, ppi=ppi, ppi_len=n_pkts))
    return out


def write_period(
    root: Path, dataset: str, period: str, n: int, *, seed: int = 0, n_labels: int = 7
) -> Path:
    """A D1-like period on disk: labels 0..4 are the stub's known classes, 5.. are unknown."""
    from adl_etc.data.tensors import ShardWriter

    flows = random_flows(n, seed, min_len=1)
    rng = np.random.default_rng(seed + 1)
    label_map = {f"class{i}": i for i in range(n_labels)}
    with ShardWriter(root, dataset, period, label_map=label_map, allow_unknown=True) as w:
        w.add_batch(
            {
                "ppi": np.stack([f.ppi for f in flows]),
                "ppi_len": np.array([f.ppi_len for f in flows], dtype=np.int8),
                "flowstats": np.zeros((n, P.FLOWSTATS_DIM), np.float32),
                "label": rng.integers(0, n_labels, n).astype(np.int16),
                "category": np.zeros(n, np.int8),
                "session_id": np.arange(n, dtype=np.int32),
                "ts": np.arange(n, dtype=np.int64),
            }
        )
    return root / dataset / period


def write_multi_flow_pcap(path: Path, flows: list[list[tuple[float, int, bool, int]]]) -> Path:
    """One TCP flow per entry, each a list of ``(offset_s, payload, to_server, flags)``; flow
    ``i`` uses client port ``40000 + i``."""
    client, server = socket.inet_aton("10.0.0.5"), socket.inet_aton("93.184.216.34")
    frames = []
    for i, pkts in enumerate(flows):
        for off, payload, to_server, flags in pkts:
            src, dst = (client, server) if to_server else (server, client)
            sport, dport = (40_000 + i, 443) if to_server else (443, 40_000 + i)
            tcp = dpkt.tcp.TCP(
                sport=sport,
                dport=dport,
                seq=1,
                ack=1,
                off=5,
                flags=flags,
                win=64_240,
                data=b"\x00" * payload,
            )
            ip = dpkt.ip.IP(src=src, dst=dst, p=dpkt.ip.IP_PROTO_TCP, ttl=64, data=tcp)
            ip.len = len(ip)
            eth = dpkt.ethernet.Ethernet(
                src=b"\x00\x11\x22\x33\x44\x55",
                dst=b"\x66\x77\x88\x99\xaa\xbb",
                type=dpkt.ethernet.ETH_TYPE_IP,
                data=ip,
            )
            frames.append((off, bytes(eth)))
    frames.sort(key=lambda x: x[0])
    with open(path, "wb") as fh:
        w = dpkt.pcap.Writer(fh)
        for off, buf in frames:
            w.writepkt(buf, ts=1_700_000_000.0 + off)
    return path
