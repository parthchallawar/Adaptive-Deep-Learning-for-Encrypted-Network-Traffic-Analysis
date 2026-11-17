"""The model pass behind ScoreArrays (plan phase 4, T1). Needs torch."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from adl_etc.data import features as FT  # noqa: E402
from adl_etc.data import ppi as P  # noqa: E402
from adl_etc.data.features import Standardizer  # noqa: E402
from adl_etc.evaluation import scores as S  # noqa: E402
from adl_etc.evaluation.policies import energy_score  # noqa: E402
from adl_etc.inference.unknown import GaussianSums, class_validity  # noqa: E402
from adl_etc.models.baselines.predict import predict_dense  # noqa: E402
from adl_etc.models.pat import PAT  # noqa: E402
from adl_etc.models.registry import build_model  # noqa: E402
from adl_etc.training import score_pass as SP  # noqa: E402
from adl_etc.training.datasets import ArrayData  # noqa: E402
from adl_etc.training.labels import LabelSpace  # noqa: E402
from tests.training.util import make_flows, write_shards  # noqa: E402

C = 5
SPACE = LabelSpace.create(range(C))


@pytest.fixture(scope="module")
def std(tmp_path_factory: pytest.TempPathFactory) -> Standardizer:
    shards = write_shards(tmp_path_factory.mktemp("std"), make_flows(400, C, seed=1))
    out = Standardizer.fit(shards)
    shards.close()
    return out


@pytest.fixture(scope="module")
def flows() -> ArrayData:
    return make_flows(60, C, seed=7, min_len=1, max_len=30)


def small_pat() -> PAT:
    torch.manual_seed(0)
    return PAT(C, d_model=32, layers=2, heads=4, ffn=64).eval()


def calibration():
    return SP.identity_calibration(C, "test")


# --- next_packet_columns: the alignment is the easy thing to get wrong -------------------------


def fake_heads(b: int, peak_on: np.ndarray | None = None, shift: int = 0) -> dict:
    """Next-packet heads that are uniform, or peaked on the token ``shift`` positions ahead of
    the position they are read at (shift = 1 is the correct alignment: after j packets, packet
    j + 1)."""
    sizes = {
        "next_size": (FT.SIZE_VOCAB, P.SIZE_POS),
        "next_ipt": (FT.IPT_VOCAB, P.IPT_POS),
        "next_dir": (FT.DIR_VOCAB, P.DIR_POS),
    }
    out = {}
    for name, (v, pos) in sizes.items():
        logits = torch.zeros(b, P.K_MAX, v)
        if peak_on is not None:
            tok = torch.from_numpy(peak_on[..., pos]).long()  # [B, 30] tokens of packets 1..30
            for j in range(P.K_MAX):  # head at index j is read after j + 1 packets
                src = j + shift  # the packet it is told is "next" (0-based index)
                if 0 <= src < P.K_MAX:
                    logits[torch.arange(b), j, tok[:, src]] = 30.0
        out[name] = logits
    return out


def tokens_for(ppi_len: list[int]) -> torch.Tensor:
    flows = make_flows(len(ppi_len), C, seed=3, min_len=1, max_len=30)
    lens = np.asarray(ppi_len)
    ppi = flows.ppi.copy()
    for i, n in enumerate(lens):
        ppi[i, n:, :] = 0
    return torch.from_numpy(FT.tokenize(ppi, lens)).long()


def test_uniform_heads_give_log_vocab_per_existing_packet_and_zero_elsewhere() -> None:
    lens = [1, 2, 5, 30]
    tokens = tokens_for(lens)
    nll, ent = SP.next_packet_columns(fake_heads(len(lens)), tokens)
    per_packet = math.log(FT.SIZE_VOCAB) + math.log(FT.IPT_VOCAB) + math.log(FT.DIR_VOCAB)
    for i, n in enumerate(lens):
        expect = np.zeros(P.K_MAX)
        expect[1:n] = per_packet  # packets 2..n exist: nll indices 1..n-1
        np.testing.assert_allclose(nll[i], expect, atol=1e-4)
    np.testing.assert_allclose(ent, 1.0, atol=1e-5)  # uniform = maximal normalised entropy


def test_peaked_on_the_next_packet_is_near_zero_and_the_wrong_shift_is_not() -> None:
    lens = [6, 9, 30]
    tokens = tokens_for(lens)
    tok_np = tokens.numpy()
    good, _ = SP.next_packet_columns(fake_heads(3, tok_np, shift=1), tokens)
    wrong, _ = SP.next_packet_columns(fake_heads(3, tok_np, shift=0), tokens)
    for i, n in enumerate(lens):
        assert good[i, 1:n].max() < 1e-3, (i, good[i, :n])
        if n < P.K_MAX:
            assert good[i, n:].max() == 0.0  # nothing past the flow's end
    assert wrong[0, 1:6].mean() > 1.0  # peaking on the packet already seen is penalised


# --- score_flows ------------------------------------------------------------------------------


def test_shapes_dtypes_and_fill_forward(flows: ArrayData, std: Standardizer) -> None:
    sc = SP.score_flows(small_pat(), flows, SPACE, std, calibration(), batch_size=16)
    assert len(sc) == len(flows) and sc.meta["heads"] == "pat"
    for name, dt in S.FIELDS_2D.items():
        arr = getattr(sc, name)
        assert arr.shape == (len(flows), P.K_MAX) and arr.dtype == dt
    ks = np.arange(1, P.K_MAX + 1)[None, :]
    past = ks > sc.ppi_len[:, None]
    for name in ("conf", "p_safe", "energy", "pred", "mahal"):
        arr = getattr(sc, name)
        last = arr[np.arange(len(sc)), np.minimum(sc.ppi_len, P.K_MAX) - 1]
        assert (arr[past] == np.broadcast_to(last[:, None], arr.shape)[past]).all(), name
    assert ((sc.conf >= 0) & (sc.conf <= 1.001)).all()
    assert ((sc.margin >= 0) & (sc.margin <= 1.001)).all()


def test_agrees_with_the_repos_own_prediction_and_energy_code(
    flows: ArrayData, std: Standardizer
) -> None:
    model = small_pat()
    sc = SP.score_flows(model, flows, SPACE, std, calibration())
    dense = predict_dense(model, flows, SPACE, std)
    for k in (1, 3, 10, 30):
        np.testing.assert_array_equal(sc.pred_raw[:, k - 1], dense.predictions_at(k, flows.ppi_len))
        np.testing.assert_allclose(
            sc.energy_raw[:, k - 1], energy_score(dense.at(k, flows.ppi_len)), atol=1e-4
        )


def test_the_score_does_not_depend_on_the_batch_size(flows: ArrayData, std: Standardizer) -> None:
    model = small_pat()
    a = SP.score_flows(model, flows, SPACE, std, calibration(), batch_size=7)
    b = SP.score_flows(model, flows, SPACE, std, calibration(), batch_size=60)
    for name in S.FIELDS_2D:
        np.testing.assert_allclose(
            getattr(a, name).astype(np.float64), getattr(b, name).astype(np.float64), atol=2e-3
        )
    np.testing.assert_array_equal(a.pred, b.pred)


def test_unknown_class_flows_are_scored_and_marked(flows: ArrayData, std: Standardizer) -> None:
    data = ArrayData(
        ppi=flows.ppi, ppi_len=flows.ppi_len, label=flows.label.copy(), session_id=None
    )
    data.label[:10] = -1
    sc = SP.score_flows(small_pat(), data, SPACE, std, calibration())
    assert (
        (sc.labels[:10] == -1).all() and sc.is_unknown[:10].all() and not sc.is_unknown[10:].any()
    )


def test_a_gaussian_bank_adds_a_finite_mahalanobis_column(
    flows: ArrayData, std: Standardizer
) -> None:
    model = small_pat()
    y = SPACE.to_model(flows.label)
    sums = GaussianSums(C, class_validity(y, flows.ppi_len, C, min_class_flows=3), dim=128)
    SP.accumulate_projection_sums(model, flows, SPACE, std, sums, batch_size=16)
    # counts are exactly the number of flows that reached each K
    for k in (1, 5, 20):
        assert sums.n_all[k - 1] == int((flows.ppi_len >= k).sum())
    sc = SP.score_flows(model, flows, SPACE, std, calibration(), bank=sums.bank())
    assert np.isfinite(sc.mahal).all() and np.abs(sc.mahal).max() > 0
    without = SP.score_flows(model, flows, SPACE, std, calibration())
    assert (without.mahal == 0).all()


def test_the_projection_sums_refuse_unknown_class_flows(
    flows: ArrayData, std: Standardizer
) -> None:
    data = ArrayData(
        ppi=flows.ppi, ppi_len=flows.ppi_len, label=flows.label.copy(), session_id=None
    )
    data.label[0] = -1
    sums = GaussianSums(C, np.ones((P.K_MAX, C), bool), dim=128)
    with pytest.raises(ValueError, match="unknown"):
        SP.accumulate_projection_sums(small_pat(), data, SPACE, std, sums)


def test_a_model_without_pat_heads_still_gets_class_columns(
    flows: ArrayData, std: Standardizer
) -> None:
    gru = build_model({"name": "gru"}, C).eval()
    sc = SP.score_flows(gru, flows, SPACE, std, calibration())
    assert sc.meta["heads"] == "cls-only"
    np.testing.assert_array_equal(sc.p_safe, sc.conf)  # no safety head: confidence stands in
    assert (sc.mahal == 0).all() and (sc.pkt_nll == 0).all() and (sc.next_entropy == 0).all()
    with pytest.raises(ValueError, match="projection"):
        SP.score_flows(gru, flows, SPACE, std, calibration(), bank=object())  # type: ignore[arg-type]


def test_sample_known_per_class_caps_each_class_and_never_draws_unknowns() -> None:
    rng = np.random.default_rng(0)
    labels = np.concatenate([np.zeros(500, int), np.ones(40, int), np.full(100, -1)])
    idx = SP.sample_known_per_class(labels, 100, rng)
    assert np.bincount(labels[idx][labels[idx] >= 0]).tolist() == [100, 40]
    assert (labels[idx] >= 0).all() and (np.diff(idx) > 0).all()
