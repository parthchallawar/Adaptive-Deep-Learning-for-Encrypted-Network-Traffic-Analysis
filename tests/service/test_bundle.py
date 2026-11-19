"""The real model bundle against the offline pipeline (plan phase 6, T2). Needs torch, the local
headline checkpoint (``results/models/production.json``) and the D1 shards; skipped otherwise.

Streaming 400 real ``test_id`` flows packet by packet through the engine must give the decisions
that phase 4's offline path gives: ``score_flows`` over whole flows, then ``replay.safe_stop``.
The two forwards see different sequence lengths (the service runs only a prefix's positions), so
fp32 sums differ in the last bits and a float16 ``p_safe`` can land one step apart. A decision may
therefore differ only where a score sits within that rounding of its threshold; the test checks
that every disagreement is such a tie and that there are almost none.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from adl_etc.evaluation import replay as R  # noqa: E402
from adl_etc.service.engine import BudgetConfig, Engine  # noqa: E402
from adl_etc.service.ingest import flows_from_shards  # noqa: E402
from tests.service.test_engine import stream  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PROD = REPO / "results" / "models" / "production.json"
SHARDS = REPO / "results" / "kaggle-full" / "shards"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not PROD.is_file() or not SHARDS.is_dir(),
        reason="needs results/models/production.json and the local D1 shards",
    ),
]


@pytest.fixture(scope="module")
def bundle():
    import torch

    from adl_etc.service.bundle import ModelBundle

    torch.set_num_threads(4)
    return ModelBundle.load(PROD, root=REPO)


def test_streamed_real_decisions_equal_the_offline_score_pass(bundle) -> None:
    from adl_etc.training.datasets import ArrayData
    from adl_etc.training.score_pass import score_flows

    flows = flows_from_shards(
        SHARDS,
        "cesnet-tls-year22",
        "WEEK-2022-31",
        count=400,
        session_id=1,
        known_ids={int(i) for i in bundle.label_space.known},
        seed=3,
    )
    theta = float(bundle.reference["theta_commit"])
    f = float(bundle.reference["f"])
    eng = Engine(bundle, budget=BudgetConfig(mode="manual", theta_init=theta, f=f))
    got = stream(eng, flows, 0)

    ppi = np.stack([fl.ppi for fl in flows])
    n = np.array([fl.ppi_len for fl in flows])
    data = ArrayData(ppi=ppi, ppi_len=n, label=np.full(len(flows), -1))
    off = score_flows(
        bundle.model,
        data,
        bundle.label_space,
        bundle.standardizer,
        bundle.calibration,
        bank=bundle.unknown.bank,
        batch_size=128,
    )
    u = bundle.unknown.score_of(off)
    theta_r = bundle.unknown.theta_reject(f)
    res = R.safe_stop(off.p_safe, u, off.pred, off.ppi_len, theta, theta_r)

    differ = []
    for i, fl in enumerate(flows):
        r = got[fl.key]
        if (r.k, r.decision.code, r.decision.cls) != (
            int(res.k[i]),
            int(res.decision[i]),
            int(res.prediction[i]),
        ):
            differ.append(i)
    print(f"real model: {len(differ)} of {len(flows)} streamed decisions differ from offline")
    assert len(got) == len(flows)
    assert len(differ) <= 4, f"{len(differ)} of {len(flows)} decisions differ"
    for i in differ:  # each must be a numerical tie at some prefix the two paths both read
        upto = max(got[flows[i].key].k, int(res.k[i]))
        ps = off.p_safe[i, :upto].astype(np.float64)
        near_commit = np.abs(ps - theta) <= 2e-3
        near_reject = np.abs(u[i, :upto] - theta_r[:upto]) <= 1e-3
        assert (near_commit | near_reject).any(), f"flow {i} differs without a tie"
    codes = {int(c) for c in res.decision}
    assert len(codes) >= 3


def test_explain_gives_per_k_detail(bundle) -> None:
    flows = flows_from_shards(
        SHARDS, "cesnet-tls-year22", "WEEK-2022-31", count=3, session_id=1, known_ids=set(), seed=0
    )
    fl = max(flows, key=lambda x: x.ppi_len)
    ex = bundle.explain(fl.ppi, fl.ppi_len)
    assert len(ex["k"]) == fl.ppi_len == len(ex["top"]) == len(ex["u"])
    assert all(len(t) == 3 and t[0]["p"] >= t[1]["p"] for t in ex["top"])
    assert bundle.describe()["n_parameters"] == 885_821
