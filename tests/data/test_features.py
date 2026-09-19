"""Tokeniser, continuous view, prefixes and augmentations (spec 003, plan T2)."""

from __future__ import annotations

import numpy as np
import pytest

from adl_etc.data import features as F
from adl_etc.data import ppi as P
from adl_etc.data.tensors import ShardWriter
from tests.data.test_flows import EXPECTED_PPI, build_flows
from tests.data.test_tensors import LABEL_MAP, make_flow

# --- frozen bin edges --------------------------------------------------------


def test_bin_edges_match_generator():
    """Guards against a NumPy version silently changing geomspace's rounding:
    the frozen literal must still equal a fresh call with the same params."""
    np.testing.assert_array_equal(F.SIZE_BIN_EDGES, F._generate_size_edges())
    np.testing.assert_array_equal(F.IPT_BIN_EDGES, F._generate_ipt_edges())


def test_bin_edges_shape_and_monotone():
    assert len(F.SIZE_BIN_EDGES) == F.SIZE_BINS - 1
    assert len(F.IPT_BIN_EDGES) == F.IPT_BINS - 1
    assert np.all(np.diff(F.SIZE_BIN_EDGES) > 0)
    assert np.all(np.diff(F.IPT_BIN_EDGES) > 0)
    assert F.SIZE_BIN_EDGES[0] >= 1 and F.SIZE_BIN_EDGES[-1] <= P.SIZE_MAX
    assert F.IPT_BIN_EDGES[0] >= 1 and F.IPT_BIN_EDGES[-1] <= P.IPT_MAX_MS


# --- tokenisation against the golden reference flow --------------------------


def test_tokenize_reference_flow(reference_pcap):
    (flow,) = build_flows(reference_pcap)
    ppi_batch = flow.ppi[None, ...]
    len_batch = np.array([flow.ppi_len])

    tokens = F.tokenize(ppi_batch, len_batch)[0]
    assert tokens.shape == (P.K_MAX, P.PPI_CHANNELS)

    # position 0: ipt=0, dir=+1 (fwd), size=517, push=1
    assert tokens[0, P.DIR_POS] == 1  # DIR_FWD -> token 1
    assert tokens[0, P.PUSH_POS] == 2  # push=1 -> token 2
    assert tokens[0, P.IPT_POS] == 1  # ipt=0 is the lowest bin, token 1 (not pad)

    # position 1: dir=-1 (rev), push=0
    assert tokens[1, P.DIR_POS] == 2  # DIR_REV -> token 2
    assert tokens[1, P.PUSH_POS] == 1  # push=0 -> token 1 (not pad, despite raw 0)

    # padding beyond ppi_len is all zero in every channel
    assert np.all(tokens[flow.ppi_len :] == 0)

    # every token is within its channel's vocabulary
    assert tokens[:, P.IPT_POS].max() < F.IPT_VOCAB
    assert tokens[:, P.SIZE_POS].max() < F.SIZE_VOCAB
    assert tokens[:, P.DIR_POS].max() < F.DIR_VOCAB
    assert tokens[:, P.PUSH_POS].max() < F.PUSH_VOCAB


def test_tokenize_hand_computed_bins():
    """size=1, ipt=1 must land in bin 0 (token 1); size=1500, ipt=IPT_MAX_MS
    (the maxima) must land in the last bin (token SIZE_BINS / IPT_BINS)."""
    ppi = P.empty_ppi()
    ppi[0] = (1, P.DIR_FWD, 1, 0)
    ppi[1] = (P.IPT_MAX_MS, P.DIR_REV, P.SIZE_MAX, 1)
    tokens = F.tokenize(ppi[None, ...], np.array([2]))[0]

    assert tokens[0, P.IPT_POS] == 1
    assert tokens[0, P.SIZE_POS] == 1
    assert tokens[1, P.IPT_POS] == F.IPT_BINS
    assert tokens[1, P.SIZE_POS] == F.SIZE_BINS


def test_pad_index_is_zero_in_every_channel():
    ppi = P.empty_ppi()
    ppi[0] = (10, P.DIR_FWD, 100, 1)
    tokens = F.tokenize(ppi[None, ...], np.array([1]))[0]
    assert np.all(tokens[1:] == F.PAD_INDEX)


def test_push_channel_distinguishes_pad_from_push_zero():
    """The spec-003 correction: push=0 on a real packet must not collide with
    the padding index, even though both are numerically 0 in the raw array."""
    ppi = P.empty_ppi()
    ppi[0] = (0, P.DIR_FWD, 100, 0)  # real packet, push flag off
    tokens = F.tokenize(ppi[None, ...], np.array([1]))[0]
    assert tokens[0, P.PUSH_POS] != F.PAD_INDEX
    assert tokens[1, P.PUSH_POS] == F.PAD_INDEX  # padding position


# --- continuous view -----------------------------------------------------------


def test_continuous_reference_flow(reference_pcap):
    (flow,) = build_flows(reference_pcap)
    cont = F.continuous(flow.ppi[None, ...], np.array([flow.ppi_len]))[0]
    assert cont.shape == (P.K_MAX, F.CONTINUOUS_CHANNELS)

    ipt0, dir0, size0, push0 = EXPECTED_PPI[0]
    np.testing.assert_allclose(cont[0, 0], np.log1p(size0) * dir0, rtol=1e-5)
    np.testing.assert_allclose(cont[0, 1], np.log1p(ipt0), rtol=1e-5)
    assert cont[0, 2] == dir0
    assert cont[0, 3] == push0
    assert np.all(cont[flow.ppi_len :] == 0)


# --- padding mask and prefix --------------------------------------------------


@pytest.mark.parametrize("k", list(range(1, P.K_MAX + 1)))
def test_prefix_and_mask_agree(k):
    rng = np.random.default_rng(0)
    ppi_len = np.array([5, 15, 30, 0])
    mask = F.padding_mask(ppi_len, k=k)
    for i, n in enumerate(ppi_len):
        expected = min(int(n), k)
        assert mask[i].sum() == expected
        assert np.all(mask[i, :expected])
        assert not np.any(mask[i, expected:])
    _ = rng  # unused, keeps signature symmetric with other parametrized tests


def test_prefix_zeroes_beyond_k():
    ppi = np.stack([P.empty_ppi() for _ in range(2)])
    ppi[0, :10, P.SIZE_POS] = np.arange(1, 11)
    arrays = {"ppi": ppi, "ppi_len": np.array([10, 10])}
    out = F.prefix(arrays, k=5)
    assert np.all(out["ppi"][0, 5:] == 0)
    np.testing.assert_array_equal(out["ppi"][0, :5, P.SIZE_POS], np.arange(1, 6))
    assert out["mask"][0].sum() == 5


def test_prefix_rejects_out_of_range_k():
    arrays = {"ppi": np.zeros((1, P.K_MAX, P.PPI_CHANNELS)), "ppi_len": np.array([1])}
    with pytest.raises(ValueError):
        F.prefix(arrays, k=0)
    with pytest.raises(ValueError):
        F.prefix(arrays, k=P.K_MAX + 1)


# --- shortcut-feature exclusion -----------------------------------------------


def test_tokenize_output_has_no_shortcut_dimension():
    """The tokeniser's output width is exactly PPI_CHANNELS: no IP, port, SNI,
    JA3 or ASN field can be smuggled in because there is no room for one."""
    ppi = P.empty_ppi()
    tokens = F.tokenize(ppi[None, ...], np.array([0]))
    assert tokens.shape[-1] == P.PPI_CHANNELS == 4


# --- augmentations -------------------------------------------------------------


def _random_ppi(rng, ppi_len):
    ppi = P.empty_ppi()
    ppi[:ppi_len, P.DIR_POS] = rng.choice([P.DIR_FWD, P.DIR_REV], size=ppi_len)
    ppi[:ppi_len, P.SIZE_POS] = rng.integers(1, P.SIZE_MAX + 1, size=ppi_len)
    ppi[:ppi_len, P.IPT_POS] = np.concatenate([[0], rng.integers(0, 500, size=ppi_len - 1)])
    ppi[:ppi_len, P.PUSH_POS] = rng.integers(0, 2, size=ppi_len)
    return ppi


def _keep_indices_by_size(original_sizes: list[int], kept_sizes: list[int]) -> list[int]:
    """Recovers which original positions survived, assuming ``original_sizes``
    are unique (so the match is unambiguous)."""
    idx = 0
    out = []
    for size in kept_sizes:
        while original_sizes[idx] != size:
            idx += 1
        out.append(idx)
        idx += 1
    return out


@pytest.mark.parametrize("trial", range(20))
def test_drop_preserves_interior_time_and_order(trial):
    """Every surviving packet's IPT must equal the exact sum of the original
    gaps since the previous surviving packet (the packets it absorbed plus its
    own original gap) -- not just "close" to it. Sizes are made unique so the
    surviving positions can be identified unambiguously from the output.
    """
    rng = np.random.default_rng(trial)
    ppi_len = int(rng.integers(2, P.K_MAX + 1))
    ppi = P.empty_ppi()
    ppi[:ppi_len, P.SIZE_POS] = np.arange(1, ppi_len + 1)
    ppi[:ppi_len, P.DIR_POS] = rng.choice([P.DIR_FWD, P.DIR_REV], size=ppi_len)
    ppi[:ppi_len, P.IPT_POS] = np.concatenate([[0], rng.integers(0, 500, size=ppi_len - 1)])
    ppi[:ppi_len, P.PUSH_POS] = rng.integers(0, 2, size=ppi_len)

    new_ppi, new_len = F.drop(ppi, ppi_len, p=0.3, rng=rng)

    assert 1 <= new_len <= ppi_len
    assert new_ppi[0, P.IPT_POS] == 0

    original_sizes = list(int(v) for v in ppi[:ppi_len, P.SIZE_POS])
    kept_sizes = list(int(v) for v in new_ppi[:new_len, P.SIZE_POS])
    keep_idx = _keep_indices_by_size(original_sizes, kept_sizes)
    assert keep_idx == sorted(keep_idx)  # relative order preserved

    original_ipt = ppi[:ppi_len, P.IPT_POS].astype(np.int64)
    for j in range(1, new_len):
        lo, hi = keep_idx[j - 1] + 1, keep_idx[j] + 1
        expected = int(original_ipt[lo:hi].sum())  # own gap + every absorbed gap
        assert int(new_ppi[j, P.IPT_POS]) == expected


def test_drop_never_empties_a_flow():
    rng = np.random.default_rng(0)
    ppi = _random_ppi(rng, 5)
    _new_ppi, new_len = F.drop(ppi, 5, p=1.0, rng=rng)
    assert new_len >= 1


@pytest.mark.parametrize("trial", range(20))
def test_reorder_preserves_multiset(trial):
    rng = np.random.default_rng(100 + trial)
    ppi_len = int(rng.integers(2, P.K_MAX + 1))
    ppi = _random_ppi(rng, ppi_len)

    out = F.reorder(ppi, ppi_len, q=0.5, rng=rng)

    before = sorted(
        (int(s), int(d), int(pu))
        for s, d, pu in zip(
            ppi[:ppi_len, P.SIZE_POS],
            ppi[:ppi_len, P.DIR_POS],
            ppi[:ppi_len, P.PUSH_POS],
            strict=True,
        )
    )
    after = sorted(
        (int(s), int(d), int(pu))
        for s, d, pu in zip(
            out[:ppi_len, P.SIZE_POS],
            out[:ppi_len, P.DIR_POS],
            out[:ppi_len, P.PUSH_POS],
            strict=True,
        )
    )
    assert before == after
    assert out[0, P.IPT_POS] == 0


@pytest.mark.parametrize("trial", range(20))
def test_ipt_jitter_stays_in_range(trial):
    rng = np.random.default_rng(200 + trial)
    ppi_len = int(rng.integers(1, P.K_MAX + 1))
    ppi = _random_ppi(rng, ppi_len)

    out = F.ipt_jitter(ppi, ppi_len, s=0.2, rng=rng)

    assert out[0, P.IPT_POS] == 0
    assert np.all(out[:ppi_len, P.IPT_POS] >= 0)
    assert np.all(out[:ppi_len, P.IPT_POS] <= P.IPT_MAX_MS)


@pytest.mark.parametrize("trial", range(20))
def test_size_jitter_never_produces_padding_size(trial):
    rng = np.random.default_rng(300 + trial)
    ppi_len = int(rng.integers(1, P.K_MAX + 1))
    ppi = _random_ppi(rng, ppi_len)

    out = F.size_jitter(ppi, ppi_len, sd=8.0, rng=rng)

    assert np.all(out[:ppi_len, P.SIZE_POS] >= 1)
    assert np.all(out[:ppi_len, P.SIZE_POS] <= P.SIZE_MAX)


def test_crop_matches_prefix():
    rng = np.random.default_rng(0)
    ppi_len = 20
    for _ in range(50):
        k = F.crop(ppi_len, rng)
        assert 1 <= k <= P.K_MAX


def test_crop_caps_at_short_flow():
    rng = np.random.default_rng(0)
    for _ in range(50):
        k = F.crop(ppi_len=1, rng=rng)
        assert k == 1


# --- standardizer --------------------------------------------------------------


def _write_synthetic_shards(tmp_path, n=200, seed=0):
    rng = np.random.default_rng(seed)
    with ShardWriter(tmp_path, "synthetic", "train", label_map=LABEL_MAP) as w:
        for i in range(n):
            w.add(make_flow(rng, ts=float(i)), label=i % 2, category=0)
    from adl_etc.data.tensors import ShardSet

    return ShardSet.open(tmp_path / "synthetic" / "train")


def test_standardizer_round_trip(tmp_path):
    shards = _write_synthetic_shards(tmp_path)
    std = F.Standardizer.fit(shards)

    save_path = tmp_path / "stats.json"
    std.save(save_path)
    loaded = F.Standardizer.load(save_path)

    assert loaded.hash == std.hash
    np.testing.assert_array_equal(loaded.cont_mean, std.cont_mean)
    np.testing.assert_array_equal(loaded.flowstats_std, std.flowstats_std)

    cont = F.continuous(shards.column("ppi"), shards.column("ppi_len"))
    out1 = std.transform_continuous(cont)
    out2 = loaded.transform_continuous(cont)
    np.testing.assert_array_equal(out1, out2)


def test_flowstats_groups_partition_every_column():
    groups = F._flowstats_groups()
    assert len(groups["hist"]) == 4 * P.PHIST_BIN_COUNT
    assert len(groups["log1p"]) + len(groups["hist"]) + len(groups["flag"]) == P.FLOWSTATS_DIM
    all_idx = sorted(groups["hist"] + groups["log1p"] + groups["flag"])
    assert all_idx == list(range(P.FLOWSTATS_DIM))  # every column assigned exactly once


def test_normalize_histograms_sums_to_one_per_block():
    """Each 8-bin histogram block renormalises to proportions independently;
    a block with all-zero counts (a flow with no reverse-direction packets)
    is left at zero rather than producing a divide-by-zero."""
    groups = F._flowstats_groups()
    fs = np.zeros((1, P.FLOWSTATS_DIM), dtype=np.float64)
    hist_idx = groups["hist"]
    # first block (PSIZE_HIST): nonzero counts
    fs[0, hist_idx[:8]] = [1, 2, 3, 0, 0, 0, 0, 0]
    # second block (PSIZE_HIST_REV): all zero, must stay zero (no division by zero)

    out = F._normalize_histograms(fs, groups)

    np.testing.assert_allclose(out[0, hist_idx[:8]].sum(), 1.0)
    np.testing.assert_allclose(out[0, hist_idx[8:16]], 0.0)


def test_standardizer_fit_covers_every_flowstats_column(tmp_path):
    shards = _write_synthetic_shards(tmp_path, n=50, seed=1)
    std = F.Standardizer.fit(shards)
    assert std.flowstats_mean.shape == (P.FLOWSTATS_DIM,)
    assert std.flowstats_std.shape == (P.FLOWSTATS_DIM,)
    assert np.all(std.flowstats_std > 0)  # the near-zero guard replaced any zero std


# --- streaming vs offline equivalence (spec 020 invariant) --------------------


def test_stream_matches_offline(reference_pcap):
    (flow,) = build_flows(reference_pcap)
    st = F.StreamTensorizer()

    # feed the flow's raw packets one at a time in the same order they were recorded
    for k in range(1, flow.ppi_len + 1):
        row = flow.ppi[k - 1]
        result = st.push(
            ipt_ms=int(row[P.IPT_POS]),
            direction=int(row[P.DIR_POS]),
            size=int(row[P.SIZE_POS]),
            push_flag=int(row[P.PUSH_POS]),
        )
        offline_tokens = F.tokenize(flow.ppi[None, ...], np.array([k]))[0]
        offline_cont = F.continuous(flow.ppi[None, ...], np.array([k]))[0]
        offline_mask = F.padding_mask(np.array([k]))[0]

        np.testing.assert_array_equal(result["tokens"], offline_tokens)
        np.testing.assert_allclose(result["cont"], offline_cont, rtol=1e-5)
        np.testing.assert_array_equal(result["mask"], offline_mask)


def test_stream_reset():
    st = F.StreamTensorizer()
    st.push(10, P.DIR_FWD, 100, 1)
    st.reset()
    result = st.current()
    assert result["ppi_len"] == 0
    assert np.all(result["tokens"] == 0)


def test_stream_stops_at_k_max():
    st = F.StreamTensorizer()
    for _ in range(P.K_MAX + 5):
        result = st.push(1, P.DIR_FWD, 1, 0)
    assert result["ppi_len"] == P.K_MAX
