"""ScoreArrays and its helpers (plan phase 4, T1)."""

from __future__ import annotations

import numpy as np
import pytest

from adl_etc.evaluation import scores as S
from adl_etc.evaluation.calibration import Calibration, SafeMap
from adl_etc.evaluation.policies import energy_score


def make_scores(n: int = 40, seed: int = 0) -> S.ScoreArrays:
    rng = np.random.default_rng(seed)
    ppi_len = rng.integers(1, 40, n)
    kw = {
        "pred": rng.integers(0, 150, (n, 30)),
        "conf": rng.random((n, 30)),
        "margin": rng.random((n, 30)),
        "p_safe": rng.random((n, 30)),
        "energy": rng.normal(size=(n, 30)),
        "energy_raw": rng.normal(size=(n, 30)),
        "mahal": rng.normal(size=(n, 30)),
        "pkt_nll": rng.random((n, 30)),
        "next_entropy": rng.random((n, 30)),
    }
    kw = {k: S.fill_forward(v, ppi_len).astype(S.FIELDS_2D[k]) for k, v in kw.items()}
    return S.ScoreArrays(
        **kw,
        labels=rng.integers(-1, 150, n),
        ppi_len=ppi_len,
        flow_index=np.arange(n),
        period=np.zeros(n, dtype=np.int16),
        meta={"checkpoint_hash": "abc", "periods": ["val"]},
    )


def test_fill_forward_repeats_the_last_real_value() -> None:
    v = np.arange(30)[None, :].repeat(3, axis=0).astype(float)
    out = S.fill_forward(v, np.array([1, 5, 40]))
    assert out[0].tolist() == [0.0] * 30
    assert out[1].tolist() == [0, 1, 2, 3, 4] + [4] * 25
    assert out[2].tolist() == list(range(30))  # longer than 30: untouched


def test_running_surprise_hand_computed() -> None:
    # nll[j] = NLL of packet j+1 given the first j; only j in 1..len-1 exist.
    nll = np.zeros((2, 30))
    nll[0, :5] = [9.0, 2.0, 4.0, 6.0, 8.0]  # ppi_len 4 -> valid j = 1, 2, 3 (the 9.0 is ignored)
    nll[1, :3] = [7.0, 3.0, 5.0]  # ppi_len 1 -> nothing valid
    out = S.running_surprise(nll, np.array([4, 1]))
    assert out[0, 0] == 0.0  # S_1 = 0
    assert out[0, 1] == pytest.approx(2.0)  # mean over packet 2
    assert out[0, 2] == pytest.approx(3.0)  # (2 + 4) / 2
    assert out[0, 3] == pytest.approx(4.0)  # (2 + 4 + 6) / 3
    assert (out[0, 3:] == out[0, 3]).all()  # filled forward past the flow's end
    assert (out[1] == 0.0).all()


def test_save_load_round_trip_and_hash(tmp_path) -> None:
    a = make_scores()
    a.save(tmp_path)
    b = S.ScoreArrays.load(tmp_path)
    assert b.content_hash() == a.content_hash()
    assert b.meta == a.meta
    for name in (*S.FIELDS_2D, *S.FIELDS_1D):
        assert getattr(b, name).dtype == S.FIELDS_2D.get(name, S.FIELDS_1D.get(name))
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_load_refuses_a_damaged_copy(tmp_path) -> None:
    a = make_scores()
    a.save(tmp_path)
    arr = np.load(tmp_path / "energy.npy")
    arr[0, 0] += 1.0
    np.save(tmp_path / "energy.npy", arr)
    with pytest.raises(ValueError, match="hash"):
        S.ScoreArrays.load(tmp_path)


def test_same_inputs_same_hash() -> None:
    assert make_scores(seed=3).content_hash() == make_scores(seed=3).content_hash()
    assert make_scores(seed=3).content_hash() != make_scores(seed=4).content_hash()


def test_take_and_concat() -> None:
    a = make_scores(30)
    sub = a.take(np.arange(10))
    assert len(sub) == 10
    both = S.concat([sub, a.take(np.arange(10, 30))])
    assert len(both) == 30
    np.testing.assert_array_equal(both.pred, a.pred)
    assert np.array_equal(a.take(a.is_unknown).labels, a.labels[a.labels < 0])


def test_shape_and_length_are_checked() -> None:
    a = make_scores(5)
    bad = {name: getattr(a, name) for name in (*S.FIELDS_2D, *S.FIELDS_1D)}
    bad["conf"] = bad["conf"][:, :20]
    with pytest.raises(ValueError, match="conf"):
        S.ScoreArrays(**bad)
    bad = {name: getattr(a, name) for name in (*S.FIELDS_2D, *S.FIELDS_1D)}
    bad["ppi_len"] = np.zeros(5, dtype=np.int64)
    with pytest.raises(ValueError, match="ppi_len"):
        S.ScoreArrays(**bad)


def test_class_scores_agree_with_calibration_and_policies() -> None:
    rng = np.random.default_rng(1)
    b, c = 25, 7
    logits = rng.normal(0, 2, (b, 30, c)).astype(np.float32)
    adjustment = rng.normal(0, 0.3, c)
    temps = rng.uniform(0.7, 1.6, 30)
    out = S.class_scores(logits, adjustment, temps)

    cal = Calibration(
        checkpoint_hash="x",
        adjustment=adjustment.tolist(),
        temperatures=temps.tolist(),
        safe_maps=[SafeMap(kind="constant", values=[0.5])] * 30,
        n_fit=[1] * 30,
    )
    p = cal.probabilities(logits)
    np.testing.assert_allclose(out["conf"], p.max(-1), atol=1e-9)
    np.testing.assert_array_equal(out["pred"], cal.adjusted(logits).argmax(-1))
    np.testing.assert_array_equal(out["pred_raw"], logits.argmax(-1))
    srt = np.sort(p, axis=-1)
    np.testing.assert_allclose(out["margin"], srt[..., -1] - srt[..., -2], atol=1e-9)
    np.testing.assert_allclose(
        out["energy_raw"], energy_score(logits.astype(np.float64)), atol=1e-9
    )
    # energy at temperature 1 with zero adjustment equals the plain energy score
    plain = S.class_scores(logits, np.zeros(c), np.ones(30))
    np.testing.assert_allclose(plain["energy"], plain["energy_raw"], atol=1e-9)
