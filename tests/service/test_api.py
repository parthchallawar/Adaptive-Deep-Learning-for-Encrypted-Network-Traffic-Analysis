"""The service API (spec 016; plan phase 6, T2) end to end with a stub model: shard replay and a
PCAP upload go in, decisions come out of ``/flows`` and the WebSocket, the budget knob moves
``theta_commit``, and the UI is served from the same process."""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from adl_etc.service.app import WEB_DIR, create_app
from tests.service.util import StubScorer, write_multi_flow_pcap, write_period

REPO = Path(__file__).resolve().parents[2]
TH_SYN, TH_FIN, TH_PUSH, TH_ACK = 0x02, 0x01, 0x08, 0x10


@pytest.fixture
def client(tmp_path: Path):
    data = tmp_path / "shards"
    write_period(data, "cesnet-tls-year22", "WEEK-2022-31", 400, seed=1)
    write_period(data, "cesnet-tls-year22", "WEEK-2022-50", 200, seed=2)
    app = create_app(
        "configs/service.yaml",
        [
            f"db_url=sqlite:///{(tmp_path / 'svc.db').as_posix()}",
            f"data_root={data.as_posix()}",
            "budget.window=50",
            "budget.min_flows=20",
            "drift.window=100",
            "drift.warmup=1",
        ],
        scorer=StubScorer(),
        root=REPO,
    )
    with TestClient(app) as c:
        yield c


def wait_job(c: TestClient, job_id: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = c.get(f"/jobs/{job_id}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {j}")


def settle(c: TestClient) -> None:
    eng = c.app.state.svc.engine  # type: ignore[attr-defined]
    eng.drain()
    c.app.state.svc.store.flush()  # type: ignore[attr-defined]


def test_health_and_model(client: TestClient) -> None:
    h = client.get("/health").json()
    assert h["status"] == "ready" and h["model_version"] == "stub-a"
    m = client.get("/model").json()
    assert m["active"]["version"] == "stub-a"


def test_shard_replay_produces_decisions_and_stats(client: TestClient) -> None:
    periods = client.get("/ingest/periods").json()["periods"]
    assert {"period": "WEEK-2022-31", "split": "test_id"} in periods
    job = client.post(
        "/ingest/shards", json={"period": "WEEK-2022-31", "count": 300, "rate": 3000, "speed": 200}
    ).json()
    assert job["total"] == 300
    assert wait_job(client, job["id"])["status"] == "done"
    settle(client)
    rows = client.get("/flows", params={"limit": 1000}).json()["flows"]
    assert len(rows) == 300
    assert all(r["true_class"].startswith("class") for r in rows)
    assert any(r["decision"] in ("reject", "forced_reject") for r in rows)
    assert all(r["correct"] in (True, False) for r in rows)
    newest = rows[0]["decision_id"]
    assert client.get("/flows", params={"since": newest}).json()["flows"] == []
    s = client.get("/stats", params={"window_s": 600}).json()
    assert s["n"] == 300 and 1 <= s["mean_k"] <= 30 and s["total_decisions"] == 300
    assert 0 <= s["total_accuracy"] <= 1
    series = client.get("/stats/series", params={"minutes": 5, "bucket_s": 1}).json()["series"]
    assert sum(b["n"] for b in series) == 300
    detail = client.get(f"/flows/{rows[0]['flow_id']}").json()
    assert len(detail["packets"]) == detail["ppi_len"] and detail["decision"]["k"] == rows[0]["k"]
    assert detail["trace"] and len(detail["trace"]["p_safe"]) == rows[0]["k"]
    unk = client.get("/unknown").json()
    assert sum(unk["rejection_k_histogram"].values()) == sum(
        r["decision"] in ("reject", "forced_reject") for r in rows
    )
    drift = client.get("/drift").json()
    assert drift["enabled"] and drift["windows_closed"] == 3 and len(drift["history"]) == 3


def test_an_unknown_period_is_a_404(client: TestClient) -> None:
    r = client.post("/ingest/shards", json={"period": "WEEK-2099-01", "count": 5})
    assert r.status_code == 404


def test_a_lower_budget_lowers_theta_commit_after_replayed_traffic(client: TestClient) -> None:
    assert client.post("/budget", json={"mode": "auto", "target": 20.0}).status_code == 200
    j = client.post(
        "/ingest/shards", json={"period": "WEEK-2022-31", "count": 300, "rate": 5000, "speed": 500}
    ).json()
    wait_job(client, j["id"])
    settle(client)
    high = client.get("/budget").json()["theta_commit"]
    client.post("/budget", json={"target": 2.0})
    j = client.post(
        "/ingest/shards",
        json={"period": "WEEK-2022-31", "count": 400, "rate": 5000, "speed": 500, "seed": 1},
    ).json()
    wait_job(client, j["id"])
    settle(client)
    b = client.get("/budget").json()
    assert b["target"] == 2.0 and b["theta_commit"] < high
    assert len(b["history"]) >= 5 and {"theta_commit", "observed"} <= set(b["history"][-1])
    bad = client.post("/budget", json={"target": -3})
    assert bad.status_code == 422


def test_budget_manual_mode_holds_the_threshold(client: TestClient) -> None:
    b = client.post("/budget", json={"mode": "manual", "theta_commit": 0.66}).json()
    assert b["mode"] == "manual" and b["theta_commit"] == 0.66


def test_a_pcap_upload_stores_a_handshake_only_flow_as_undecidable(
    client: TestClient, tmp_path: Path
) -> None:
    hs = [(0.0, 0, True, TH_SYN), (0.01, 0, False, TH_SYN | TH_ACK), (0.02, 0, True, TH_ACK)]
    talk = [(0.03 + 0.01 * i, 120 + 31 * i, i % 2 == 0, TH_ACK | TH_PUSH) for i in range(8)]
    fin = [(0.4, 0, True, TH_FIN | TH_ACK), (0.41, 0, False, TH_FIN | TH_ACK)]
    pcap = write_multi_flow_pcap(tmp_path / "up.pcap", [hs + talk + fin, hs + fin])
    with pcap.open("rb") as fh:
        job = client.post(
            "/ingest/pcap", files={"file": ("up.pcap", fh)}, data={"speed": "50"}
        ).json()
    done = wait_job(client, job["id"])
    assert done["status"] == "done" and done["total"] == 2 and done["detail"]["undecidable"] == 1
    settle(client)
    rows = client.get("/flows").json()["flows"]
    assert sorted(r["decision"] == "undecidable" for r in rows) == [False, True]
    assert all(len(r["key_hash"]) == 40 for r in rows)
    assert all("10.0.0.5" not in str(r) for r in rows)  # no address is ever stored


def test_live_capture_is_not_implemented(client: TestClient) -> None:
    assert client.post("/ingest/live/start").status_code == 501
    assert client.post("/ingest/live/stop").status_code == 501


def test_score_flow_returns_per_k_rows_and_the_first_decision(client: TestClient) -> None:
    ppi = [[0, 1, 300, 1], [4, -1, 1200, 0], [3, -1, 1400, 0], [9, 1, 90, 1]] * 3
    r = client.post("/score/flow", json={"ppi": ppi}).json()
    assert len(r["per_k"]) == 12 and r["decision"] is not None
    assert client.post("/score/flow", json={"ppi": []}).json()["decision"] == "undecidable"
    assert client.post("/score/flow", json={"ppi": [[1, 2]]}).status_code == 422


def test_the_websocket_streams_decisions(client: TestClient) -> None:
    with client.websocket_connect("/stream/decisions") as ws:
        client.post(
            "/ingest/shards",
            json={"period": "WEEK-2022-50", "count": 20, "rate": 2000, "speed": 500},
        )
        ev = ws.receive_json()
    assert {"flow_id", "k", "decision", "class_name", "p_safe", "u", "ts"} <= set(ev)


def test_a_model_reload_failure_keeps_the_old_model(client: TestClient) -> None:
    r = client.post("/model", json={"production": "does/not/exist.json"})
    assert r.status_code == 422
    assert client.get("/health").json()["model_version"] == "stub-a"


def test_metrics_are_prometheus_text(client: TestClient) -> None:
    text = client.get("/metrics").text
    assert "adl_service_up 1" in text and 'adl_batch_latency_ms{quantile="0.95"}' in text


def test_experiments_and_system(client: TestClient) -> None:
    assert "tables" in client.get("/experiments").json()
    assert client.get("/system").json()["status"] == "ready"


def test_the_ui_is_served_and_every_endpoint_it_calls_exists(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200 and "<html" in page.text.lower()
    js = "\n".join(p.read_text(encoding="utf-8") for p in WEB_DIR.glob("*.js"))
    called = set(re.findall(r"""(?:api|post|fetch)\(\s*["'`](/[a-z/_]+)""", js))
    assert called, "the UI calls no endpoint?"
    routes = {getattr(r, "path", "") for r in client.app.routes}
    missing = {
        c
        for c in called
        if not any(
            re.fullmatch(re.sub(r"\{[^}]+\}", "[^/]+", p), c) or p.startswith(c) for p in routes
        )
    }
    assert not missing, missing
    for asset in re.findall(r'(?:src|href)="(/static/[^"]+)"', page.text):
        assert client.get(asset).status_code == 200, asset
