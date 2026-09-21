"""prefix_flowstats: the honest early tabular feature (plan phase 2 T4).

The expected vectors for the reference flow are worked out by hand from
``EXPECTED_PPI`` in ``test_flows.py``, not copied from the function's output.
The reference capture has 12 packets but only 6 payload packets, and one
2000-byte packet clipped to 1500 in the PPI, so the stored (whole-flow) and
prefix (PPI-only) vectors must differ in specific, predictable ways.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from adl_etc.data import ppi as P
from adl_etc.data.flows import FlowRecord
from adl_etc.data.prefix_stats import PPI_DERIVED_COLUMNS, prefix_flowstats
from adl_etc.data.tensors import ShardSet
from tests.data.test_flows import EXPECTED_PPI, build_flows

REPO_ROOT = Path(__file__).resolve().parents[2]
COL = {name: i for i, name in enumerate(P.FLOWSTATS_COLUMNS)}


def batch_of(ppi_rows: list[tuple[int, int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    ppi = P.empty_ppi()
    ppi[: len(ppi_rows)] = np.array(ppi_rows, dtype=P.PPI_DTYPE)
    return ppi[None], np.array([len(ppi_rows)], dtype=np.int8)


def vec(**nonzero: float) -> np.ndarray:
    """A 46-vector that is zero except for the named columns."""
    out = np.zeros(P.FLOWSTATS_DIM, dtype=np.float32)
    for name, value in nonzero.items():
        out[COL[name]] = value
    return out


@pytest.fixture
def reference_flow(reference_pcap: Path) -> FlowRecord:
    (flow,) = build_flows(reference_pcap)
    return flow


@pytest.fixture
def reference_batch() -> tuple[np.ndarray, np.ndarray]:
    return batch_of(EXPECTED_PPI)


# --- hand-computed values on the reference flow ----------------------------------


def test_k1_is_one_packet_zero_duration(reference_batch) -> None:
    ppi, ppi_len = reference_batch

    got = prefix_flowstats(ppi, ppi_len, 1)[0]

    # Only the ClientHello (517 B, fwd, ipt 0, push): size bin (512,1024] = 6, ipt bin 0.
    expected = vec(BYTES=517, PACKETS=1, PPI_LEN=1, FLAG_PSH=1, PSIZE_HIST_6=1, IPT_HIST_0=1)
    np.testing.assert_array_equal(got, expected)


def test_k3_counts_the_first_three_packets_only(reference_batch) -> None:
    ppi, ppi_len = reference_batch

    got = prefix_flowstats(ppi, ppi_len, 3)[0]

    # fwd 517@ipt0; rev 1460@ipt45, 1460@ipt5. 1460 > 1024 -> top size bin 7.
    # ipt 45 in (16,64] -> bin 3; ipt 5 in (4,16] -> bin 2.
    expected = vec(
        BYTES=517,
        BYTES_REV=2920,
        PACKETS=1,
        PACKETS_REV=2,
        DURATION=0.050,
        PPI_LEN=3,
        PPI_DURATION=50,
        PPI_ROUNDTRIPS=0,  # + - - : one direction change, halved and floored
        FLAG_PSH=1,
        PSIZE_HIST_6=1,
        PSIZE_HIST_REV_7=2,
        IPT_HIST_0=1,
        IPT_HIST_REV_2=1,
        IPT_HIST_REV_3=1,
    )
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_k6_uses_the_whole_reference_flow(reference_batch) -> None:
    ppi, ppi_len = reference_batch

    got = prefix_flowstats(ppi, ppi_len, 6)[0]

    # fwd: 517@0, 80@30. rev: 1460@45, 1460@5, 1460@10, 1500@380 (clipped size).
    # 80 in (64,128] -> size bin 3; ipt 30 in (16,64] -> bin 3; ipt 380 in (256,1024] -> bin 5.
    expected = vec(
        BYTES=597,
        BYTES_REV=5880,  # 3*1460 + 1500, the *clipped* size the PPI holds
        PACKETS=2,
        PACKETS_REV=4,
        DURATION=0.470,
        PPI_LEN=6,
        PPI_DURATION=470,
        PPI_ROUNDTRIPS=1,  # + - - - + - : 3 direction changes, 3 // 2
        FLAG_PSH=1,
        PSIZE_HIST_3=1,
        PSIZE_HIST_6=1,
        PSIZE_HIST_REV_7=4,
        IPT_HIST_0=1,
        IPT_HIST_3=1,
        IPT_HIST_REV_2=2,
        IPT_HIST_REV_3=1,
        IPT_HIST_REV_5=1,
    )
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_k_beyond_the_flow_length_clamps_to_the_flow(reference_batch) -> None:
    ppi, ppi_len = reference_batch

    np.testing.assert_array_equal(
        prefix_flowstats(ppi, ppi_len, 30), prefix_flowstats(ppi, ppi_len, 6)
    )
    assert prefix_flowstats(ppi, ppi_len, 30)[0, COL["PPI_LEN"]] == 6


# --- properties -----------------------------------------------------------------


def random_batch(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ppi = np.zeros((n, P.K_MAX, P.PPI_CHANNELS), dtype=P.PPI_DTYPE)
    ppi_len = rng.integers(1, P.K_MAX + 1, n).astype(np.int8)
    for i, length in enumerate(ppi_len):
        ppi[i, :length, P.IPT_POS] = rng.integers(0, P.IPT_MAX_MS + 1, length)
        ppi[i, 0, P.IPT_POS] = 0
        ppi[i, :length, P.DIR_POS] = rng.choice([P.DIR_FWD, P.DIR_REV], length)
        ppi[i, :length, P.SIZE_POS] = rng.integers(1, P.SIZE_MAX + 1, length)
        ppi[i, :length, P.PUSH_POS] = rng.integers(0, 2, length)
    return ppi, ppi_len


def test_every_column_is_non_decreasing_in_k() -> None:
    """More packets can only add packets, bytes, time, histogram mass and a push."""
    ppi, ppi_len = random_batch(300, seed=1)

    stacked = np.stack([prefix_flowstats(ppi, ppi_len, k) for k in range(1, P.K_MAX + 1)])

    assert (np.diff(stacked, axis=0) >= 0).all()


def test_every_column_is_constant_once_k_reaches_the_flow_length() -> None:
    ppi, ppi_len = random_batch(300, seed=2)
    at_30 = prefix_flowstats(ppi, ppi_len, P.K_MAX)

    for k in (5, 12, 20):
        at_k = prefix_flowstats(ppi, ppi_len, k)
        done = ppi_len <= k  # flows that have shown all their packets by k
        np.testing.assert_array_equal(at_k[done], at_30[done])


def test_a_flow_is_unaffected_by_packets_after_k() -> None:
    """The leakage property itself: scrambling everything past K changes nothing."""
    ppi, ppi_len = random_batch(200, seed=3)
    k = 7
    tampered = ppi.copy()
    rng = np.random.default_rng(99)
    tampered[:, k:, P.SIZE_POS] = rng.integers(1, P.SIZE_MAX + 1, tampered[:, k:, 0].shape)
    tampered[:, k:, P.DIR_POS] = rng.choice([P.DIR_FWD, P.DIR_REV], tampered[:, k:, 0].shape)
    tampered[:, k:, P.PUSH_POS] = 1

    np.testing.assert_array_equal(
        prefix_flowstats(ppi, ppi_len, k), prefix_flowstats(tampered, ppi_len, k)
    )


def test_prefix_5_differs_from_prefix_30_on_a_long_flow() -> None:
    """If this fails, the future leaked into the past."""
    ppi, ppi_len = batch_of([(0 if i == 0 else 10, P.DIR_FWD, 100, 0) for i in range(12)])

    early = prefix_flowstats(ppi, ppi_len, 5)[0]
    late = prefix_flowstats(ppi, ppi_len, 30)[0]

    assert early[COL["PACKETS"]] == 5 and late[COL["PACKETS"]] == 12
    assert early[COL["BYTES"]] == 500 and late[COL["BYTES"]] == 1200
    assert not np.array_equal(early, late)


def test_only_the_push_flag_is_ever_nonzero() -> None:
    ppi, ppi_len = random_batch(200, seed=4)
    flag_cols = [COL[c] for c in P.FLOWSTATS_COLUMNS if c.startswith("FLAG_") and c != "FLAG_PSH"]

    out = prefix_flowstats(ppi, ppi_len, P.K_MAX)

    assert not out[:, flag_cols].any()


def test_push_flag_only_counts_pushes_inside_the_prefix() -> None:
    # Push appears only on the 4th packet.
    rows = [
        (0, P.DIR_FWD, 50, 0),
        (5, P.DIR_REV, 50, 0),
        (5, P.DIR_FWD, 50, 0),
        (5, P.DIR_REV, 50, 1),
    ]
    ppi, ppi_len = batch_of(rows)

    assert prefix_flowstats(ppi, ppi_len, 3)[0, COL["FLAG_PSH"]] == 0
    assert prefix_flowstats(ppi, ppi_len, 4)[0, COL["FLAG_PSH"]] == 1


# --- against the independent loop implementation ------------------------------------


def flow_record_for(ppi_one: np.ndarray, length: int) -> FlowRecord:
    """A FlowRecord whose ``flowstats()`` (phase 1's loop-based code) we can call."""
    return FlowRecord(
        key=((b"", 0), (b"", 0), 6),
        proto=6,
        start_ts=0.0,
        end_ts=0.0,
        ppi=ppi_one.copy(),
        ppi_len=length,
        packets=0,
        packets_rev=0,
        bytes=0,
        bytes_rev=0,
        flags_seen=0,
        end_reason="eof",
    )


@pytest.mark.parametrize("k", [1, 2, 5, 13, 30])
def test_ppi_derived_columns_match_the_loop_implementation_on_random_flows(k: int) -> None:
    """Vectorised NumPy vs FlowRecord.flowstats() (phase 1, hand-tested): a
    FlowRecord truncated to k packets must give the same PPI-derived columns."""
    ppi, ppi_len = random_batch(150, seed=10 + k)
    got = prefix_flowstats(ppi, ppi_len, k)
    cols = [COL[c] for c in PPI_DERIVED_COLUMNS]

    for i in range(len(ppi)):
        length = min(k, int(ppi_len[i]))
        truncated = ppi[i].copy()
        truncated[length:] = 0
        expected = flow_record_for(truncated, length).flowstats()
        np.testing.assert_array_equal(got[i, cols], expected[cols])


# --- F2: the stored vector is a different quantity -------------------------------------


def test_prefix_at_30_matches_stored_only_on_ppi_derived_columns(reference_flow) -> None:
    stored = reference_flow.flowstats()
    ppi, ppi_len = reference_flow.ppi[None], np.array([reference_flow.ppi_len], dtype=np.int8)

    got = prefix_flowstats(ppi, ppi_len, P.K_MAX)[0]

    derived = [COL[c] for c in PPI_DERIVED_COLUMNS]
    np.testing.assert_array_equal(got[derived], stored[derived])

    # And it must NOT match on the whole-flow columns. Reference flow: 12 packets
    # vs 6 payload packets, a clipped 2000 B packet, 0.61 s vs 0.47 s, and flags
    # (SYN, FIN, ACK) that only the whole flow ever sees.
    assert stored[COL["PACKETS"]] + stored[COL["PACKETS_REV"]] == 12
    assert got[COL["PACKETS"]] + got[COL["PACKETS_REV"]] == 6
    assert stored[COL["BYTES_REV"]] == 6380 and got[COL["BYTES_REV"]] == 5880
    assert stored[COL["DURATION"]] == pytest.approx(0.61, abs=1e-4)
    assert got[COL["DURATION"]] == pytest.approx(0.47, abs=1e-6)
    for flag in ("FLAG_SYN", "FLAG_FIN", "FLAG_ACK"):
        assert stored[COL[flag]] == 1 and got[COL[flag]] == 0
    assert not np.array_equal(got, stored)


# --- validation ------------------------------------------------------------------------


@pytest.mark.parametrize("bad_k", [0, -1, P.K_MAX + 1, 2.5, True, "3"])
def test_bad_k_is_rejected(reference_batch, bad_k) -> None:
    ppi, ppi_len = reference_batch

    with pytest.raises(ValueError, match="k must be"):
        prefix_flowstats(ppi, ppi_len, bad_k)


def test_zero_length_flow_is_rejected_not_silently_computed() -> None:
    ppi = P.empty_ppi()[None]

    with pytest.raises(ValueError, match="zero-packet"):
        prefix_flowstats(ppi, np.array([0], dtype=np.int8), 5)


def test_bad_shapes_are_rejected(reference_batch) -> None:
    ppi, ppi_len = reference_batch

    with pytest.raises(ValueError, match="ppi must be"):
        prefix_flowstats(ppi[:, :10], ppi_len, 5)
    with pytest.raises(ValueError, match="ppi_len must be"):
        prefix_flowstats(ppi, np.array([6, 6], dtype=np.int8), 5)


def test_output_layout_matches_the_stored_vector() -> None:
    ppi, ppi_len = random_batch(10, seed=5)

    out = prefix_flowstats(ppi, ppi_len, 8)

    assert out.shape == (10, P.FLOWSTATS_DIM) and out.dtype == np.float32
    assert np.isfinite(out).all()


# --- real data (skipped, with a reason, when the shards aren't on this machine) ---------

D4 = REPO_ROOT / "data" / "processed" / "ustc-tfc2016" / "all"
needs_d4 = pytest.mark.skipif(not (D4 / "meta.json").exists(), reason="D4 shards not exported here")


@needs_d4
def test_real_d4_ppi_derived_columns_equal_the_exporters_stored_values() -> None:
    """Independent check on real bytes: the vectorised prefix at k=30 must
    reproduce, for every one of 50,000 real flows, the PPI-derived columns
    the phase-1 exporter wrote from its own loop-based code."""
    ss = ShardSet.open(D4)
    try:
        n = 50_000
        ppi = np.asarray(ss.column("ppi")[:n])
        ppi_len = np.asarray(ss.column("ppi_len")[:n])
        stored = np.asarray(ss.column("flowstats")[:n])
    finally:
        ss.close()

    got = prefix_flowstats(ppi, ppi_len, P.K_MAX)

    derived = [COL[c] for c in PPI_DERIVED_COLUMNS]
    np.testing.assert_array_equal(got[:, derived], stored[:, derived])


@needs_d4
def test_real_d4_prefix_is_not_the_stored_whole_flow_vector() -> None:
    """F2 on real traffic: real flows carry pure ACKs and post-PPI packets, so
    the packet counts must differ for many flows, and early must differ from late."""
    ss = ShardSet.open(D4)
    try:
        n = 50_000
        ppi = np.asarray(ss.column("ppi")[:n])
        ppi_len = np.asarray(ss.column("ppi_len")[:n])
        stored = np.asarray(ss.column("flowstats")[:n])
    finally:
        ss.close()

    late = prefix_flowstats(ppi, ppi_len, P.K_MAX)
    early = prefix_flowstats(ppi, ppi_len, 5)

    stored_packets = stored[:, COL["PACKETS"]] + stored[:, COL["PACKETS_REV"]]
    prefix_packets = late[:, COL["PACKETS"]] + late[:, COL["PACKETS_REV"]]
    assert (stored_packets != prefix_packets).mean() > 0.5
    long_flows = ppi_len > 5
    assert long_flows.any()
    assert (early[long_flows] != late[long_flows]).any(axis=1).all()


@needs_d4
def test_prefix_flowstats_is_fast_enough_for_thirteen_evaluations() -> None:
    """B1 evaluates 13 K values, so one call over 100k real flows must be cheap."""
    import time

    ss = ShardSet.open(D4)
    try:
        ppi = np.asarray(ss.column("ppi")[:100_000])
        ppi_len = np.asarray(ss.column("ppi_len")[:100_000])
    finally:
        ss.close()

    start = time.perf_counter()
    prefix_flowstats(ppi, ppi_len, 10)
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0, f"{elapsed:.2f}s for 100k flows"
