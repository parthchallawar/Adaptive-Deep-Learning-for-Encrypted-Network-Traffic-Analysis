"""The inference service (spec 016; plan phase 6, T2): FastAPI on ``127.0.0.1``.

    python -m adl_etc.service                      # configs/service.yaml
    python -m adl_etc.service --override port=8001 drift.window=1000

Start-up loads the model bundle on a background thread (``/health`` says ``loading`` until it is
ready), opens the database (one writer thread, WAL) and starts the inference loop. The web UI
(``adl_etc.dashboard``, correction S10) is served at ``/`` from the same process and talks to the
endpoints below.

``create_app(scorer=...)`` injects a stub scorer for tests, so the API is testable without torch.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shutil
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import insert, select, update
from starlette.websockets import WebSocketDisconnect

from adl_etc.data import ppi as P
from adl_etc.inference.policy import decide
from adl_etc.service import schemas as SC
from adl_etc.service.db import models as DBM
from adl_etc.service.db.store import Store
from adl_etc.service.engine import (
    BudgetConfig,
    DecisionRecord,
    DriftConfig,
    Engine,
    EngineConfig,
    Scorer,
    drift_reference_from,
)
from adl_etc.service.ingest import (
    Job,
    Replayer,
    arrival_times,
    flows_from_pcap,
    flows_from_shards,
    pcap_start_times,
)

log = logging.getLogger("adl_etc.service")

WEB_DIR = Path(__file__).resolve().parents[1] / "dashboard" / "web"
SPLIT_NAMES = ("train", "val", "test_id", "test_drift")


class ServiceState:
    """Everything the endpoints share. One per app."""

    def __init__(self, cfg: Any, root: Path) -> None:
        self.cfg = cfg
        self.root = root
        self.store = Store(str(cfg.db_url))
        self.engine: Engine | None = None
        self.bundle: Any = None  # ModelBundle when the real model is loaded
        self.status = "starting"
        self.error: str | None = None
        self.jobs: dict[str, Job] = {}
        self.ws_clients: set[asyncio.Queue[dict[str, Any]]] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.started_at = time.time()
        self.split = _load_split_meta(root / str(cfg.split))
        self._reload_lock = threading.Lock()

    # -- model ---------------------------------------------------------------------------------

    def production_path(self) -> Path:
        return self.root / str(self.cfg.production)

    def load_bundle(self, production: Path | None = None) -> Any:
        import torch

        from adl_etc.service.bundle import ModelBundle

        torch.set_num_threads(int(self.cfg.torch_threads))
        return ModelBundle.load(production or self.production_path(), root=self.root)

    def start_engine(self, scorer: Scorer) -> None:
        reference = getattr(scorer, "reference", None) or {}
        e, b, d = self.cfg.engine, self.cfg.budget, self.cfg.drift
        theta0 = float(reference.get("theta_commit", 0.9))
        budget = BudgetConfig(
            mode=str(b.mode),
            metric=str(b.metric),
            target=float(b.target),
            window=int(b.window),
            min_flows=int(b.min_flows),
            kp=float(b.kp),
            ki=float(b.ki),
            i_max=float(b.i_max),
            theta_min=float(b.theta_min),
            theta_max=float(b.theta_max),
            quality_theta=float(b.quality_theta),
            f=float(reference.get("f", b.f)),
            theta_init=theta0,
        )
        ref_window = int(reference.get("drift_window", 5000))
        drift = DriftConfig(
            window=int(d.window),
            warmup=int(d.warmup),
            span=int(d.span),
            static=(
                drift_reference_from(reference["drift_static"], ref_window)
                if "drift_static" in reference
                else None
            ),
            controller=(
                drift_reference_from(reference["drift_controller"], ref_window)
                if "drift_controller" in reference
                else None
            ),
            reference_hist=(
                np.asarray(reference["reference_hist"], dtype=np.float64)
                if "reference_hist" in reference
                else None
            ),
        )
        engine = Engine(
            scorer,
            config=EngineConfig(
                max_batch=int(e.max_batch),
                max_wait_ms=float(e.max_wait_ms),
                idle_timeout_s=float(e.idle_timeout_s),
                max_flows=int(e.max_flows),
                queue_limit=int(e.queue_limit),
                order=str(e.order),
                k_min=int(e.k_min),
                k_ood=int(e.k_ood),
                trace=bool(e.trace),
            ),
            budget=budget,
            drift=drift,
            sink=self.store.put,
            first_flow_id=self.store.next_flow_id(),
        )
        engine.listeners.append(self._broadcast)
        engine.start()
        self.engine = engine
        self._record_model(scorer)
        self.status = "ready"

    def _record_model(self, scorer: Scorer) -> None:
        version = str(scorer.version)
        existing = self.store._read(
            select(DBM.model_version.c.version).where(DBM.model_version.c.version == version)
        )
        values = {
            "run_dir": str(getattr(scorer, "run_dir", "stub")),
            "checkpoint_hash": str(getattr(scorer, "checkpoint_hash", version)),
            "loaded_at": time.time(),
        }
        if existing:
            self.store.execute_now(
                update(DBM.model_version)
                .where(DBM.model_version.c.version == version)
                .values(**values)
            )
        else:
            self.store.execute_now(insert(DBM.model_version).values(version=version, **values))

    def known_ids(self) -> set[int]:
        if self.bundle is not None:
            return {int(i) for i in self.bundle.label_space.known}
        scorer = self.engine.scorer if self.engine else None
        return set(getattr(scorer, "known_ids", set()))

    # -- websocket fan-out ---------------------------------------------------------------------

    def _broadcast(self, rec: DecisionRecord) -> None:
        loop = self.loop
        if loop is None or not self.ws_clients:
            return
        event = rec.to_event()
        for q in list(self.ws_clients):
            loop.call_soon_threadsafe(_offer, q, event)

    # -- jobs ------------------------------------------------------------------------------------

    def new_job(self, kind: str, total: int, session_id: int, detail: dict[str, Any]) -> Job:
        job = Job(type=kind, total=total, session_id=session_id, detail=detail)
        self.jobs[job.id] = job
        self.store.execute_now(
            insert(DBM.job).values(
                id=job.id,
                type=kind,
                status=job.status,
                progress=0,
                total=total,
                created_at=job.created_at,
                session_id=session_id,
            )
        )
        return job

    def job_done(self, job: Job) -> None:
        with contextlib.suppress(Exception):
            self.store.execute_now(
                update(DBM.job)
                .where(DBM.job.c.id == job.id)
                .values(
                    status=job.status,
                    progress=job.progress,
                    error=job.error,
                    finished_at=job.finished_at,
                )
            )
            if job.session_id is not None:
                self.store.end_session(job.session_id)


def _offer(q: asyncio.Queue[dict[str, Any]], event: dict[str, Any]) -> None:
    if q.qsize() < 2000:  # a slow client misses live events; the database still has every one
        q.put_nowait(event)


def _load_split_meta(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"dataset": "cesnet-tls-year22", "periods": {}}
    from omegaconf import OmegaConf

    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(raw, dict)
    periods: dict[str, str] = {}
    for name in SPLIT_NAMES:
        for p in (raw.get("splits", {}).get(name, {}) or {}).get("periods", []) or []:
            periods[str(p)] = name
    return {"dataset": str(raw.get("dataset", "cesnet-tls-year22")), "periods": periods}


def _engine(state: ServiceState) -> Engine:
    if state.engine is None:
        raise HTTPException(
            503, detail=f"model {state.status}" + (f": {state.error}" if state.error else "")
        )
    return state.engine


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --- the app --------------------------------------------------------------------------------------


def create_app(
    config: str | Path = "configs/service.yaml",
    overrides: tuple[str, ...] | list[str] = (),
    *,
    scorer: Scorer | None = None,
    root: str | Path = ".",
) -> FastAPI:
    from adl_etc.utils.config import load_config

    root = Path(root)
    cfg = load_config(root / config if not Path(config).is_absolute() else config, overrides)
    state = ServiceState(cfg, root)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state.loop = asyncio.get_running_loop()
        state.store.start()
        if scorer is not None:
            state.start_engine(scorer)
        else:
            state.status = "loading"

            def _load() -> None:
                try:
                    state.bundle = state.load_bundle()
                    state.start_engine(state.bundle)
                    log.info("model %s ready", state.bundle.version)
                except Exception as exc:  # noqa: BLE001 - reported on /health
                    log.exception("model bundle failed to load")
                    state.status, state.error = "error", str(exc)

            threading.Thread(target=_load, name="bundle-load", daemon=True).start()
        try:
            yield
        finally:
            for job in state.jobs.values():
                job.cancel.set()
            if state.engine is not None:
                state.engine.stop()
            state.store.stop()

    app = FastAPI(
        title="Adaptive Deep Learning for Encrypted Network Traffic Analysis: inference service",
        version="0.1.0",
        description="Early, open-world classification of encrypted flows (specs 016-018).",
        lifespan=lifespan,
    )
    app.state.svc = state

    # -- UI ----------------------------------------------------------------------------------

    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> Any:
        page = WEB_DIR / "index.html"
        if page.is_file():
            return FileResponse(page)
        return RedirectResponse("/docs")

    # -- health and metrics ------------------------------------------------------------------

    @app.get("/health")
    def health() -> dict[str, Any]:
        eng = state.engine
        return {
            "status": state.status,
            "error": state.error,
            "uptime_s": time.time() - state.started_at,
            "model_version": eng.scorer.version if eng else None,
            "db_pending_rows": state.store.pending,
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        lines = [
            "# HELP adl_service_up 1 when the model is loaded",
            "# TYPE adl_service_up gauge",
            f"adl_service_up {1 if state.status == 'ready' else 0}",
        ]
        if state.engine is not None:
            m = state.engine.metrics()
            for name in (
                "events",
                "ignored_events",
                "dropped_flows",
                "decisions",
                "rejected",
                "undecidable",
                "batches",
                "forwards",
                "idle_evictions",
            ):
                lines += [f"# TYPE adl_{name}_total counter", f"adl_{name}_total {m[name]}"]
            for name in ("open_flows", "queue_size", "events_per_s"):
                lines += [f"# TYPE adl_{name} gauge", f"adl_{name} {m[name]}"]
            lines += [
                "# TYPE adl_batch_latency_ms summary",
                f'adl_batch_latency_ms{{quantile="0.5"}} {m["batch_ms_p50"]:.3f}',
                f'adl_batch_latency_ms{{quantile="0.95"}} {m["batch_ms_p95"]:.3f}',
            ]
        lines += [
            "# TYPE adl_db_rows_written_total counter",
            f"adl_db_rows_written_total {state.store.rows_written}",
            "# TYPE adl_db_pending_rows gauge",
            f"adl_db_pending_rows {state.store.pending}",
        ]
        return "\n".join(lines) + "\n"

    @app.get("/system")
    def system() -> dict[str, Any]:
        import os
        import platform

        eng = state.engine
        return {
            "status": state.status,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "engine": eng.metrics() if eng else None,
            "engine_config": vars(eng.cfg) if eng else None,
            "db_url": str(state.cfg.db_url),
            "db_rows_written": state.store.rows_written,
            "db_failed_rows": state.store.failed_rows,
        }

    # -- ingest ----------------------------------------------------------------------------------

    @app.get("/ingest/periods")
    def periods() -> dict[str, Any]:
        base = state.root / str(state.cfg.data_root)
        ds = state.split["dataset"]
        found = []
        for d in (base / ds, base):
            if d.is_dir():
                found = sorted(p.name for p in d.iterdir() if (p / "meta.json").is_file())
                if found:
                    break
        return {
            "dataset": ds,
            "periods": [{"period": p, "split": state.split["periods"].get(p, "")} for p in found],
        }

    @app.post("/ingest/shards")
    def ingest_shards(req: SC.ShardReplayRequest) -> dict[str, Any]:
        eng = _engine(state)
        session = state.store.new_session("shards", req.period, backend="shard-replay")
        try:
            flows = flows_from_shards(
                state.root / str(state.cfg.data_root),
                state.split["dataset"],
                req.period,
                count=req.count,
                session_id=session,
                known_ids=state.known_ids(),
                seed=req.seed,
            )
        except FileNotFoundError as exc:
            state.store.end_session(session)
            raise HTTPException(404, detail=str(exc)) from exc
        job = state.new_job(
            "shards",
            len(flows),
            session,
            {
                "period": req.period,
                "split": state.split["periods"].get(req.period, ""),
                "rate": req.rate,
                "speed": req.speed,
            },
        )
        Replayer(
            eng,
            flows,
            arrival_times(len(flows), req.rate, req.seed),
            job=job,
            speed=req.speed,
            on_done=state.job_done,
        ).start()
        return job.public()

    @app.post("/ingest/pcap")
    async def ingest_pcap(
        file: Annotated[UploadFile, File()], speed: Annotated[float, Form()] = 1.0
    ) -> dict[str, Any]:
        eng = _engine(state)
        if speed <= 0:
            raise HTTPException(422, detail="speed must be positive")
        uploads = Path(str(state.cfg.db_url).removeprefix("sqlite:///")).parent / "uploads"
        if not uploads.is_absolute():
            uploads = state.root / uploads
        uploads.mkdir(parents=True, exist_ok=True)
        name = Path(file.filename or "upload.pcap").name
        dest = uploads / f"{int(time.time())}-{name}"
        with dest.open("wb") as fh:  # streamed to disk, never held in memory (spec 016)
            await asyncio.to_thread(shutil.copyfileobj, file.file, fh, 1 << 20)
        session = state.store.new_session("pcap", name, backend="dpkt")
        job = state.new_job("pcap", 0, session, {"file": name, "speed": speed})

        def _run() -> None:
            try:
                job.status = "running"
                flows, stats = flows_from_pcap(dest, session_id=session)
                job.total = len(flows)
                job.detail["pcap_stats"] = stats
                job.detail["undecidable"] = sum(1 for f in flows if f.ppi_len == 0)
                Replayer(
                    eng,
                    flows,
                    pcap_start_times(flows, speed),
                    job=job,
                    speed=speed,
                    on_done=state.job_done,
                ).start()
            except Exception as exc:  # noqa: BLE001 - reported through the job
                log.exception("pcap job failed")
                job.status, job.error, job.finished_at = "failed", str(exc), time.time()
                state.job_done(job)

        threading.Thread(target=_run, name=f"pcap-{job.id}", daemon=True).start()
        return job.public()

    @app.post("/ingest/live/start")
    @app.post("/ingest/live/stop")
    def live() -> JSONResponse:
        return JSONResponse(
            status_code=501,
            content={
                "detail": "live capture is not built (plan phase 6, correction S1): Npcap is not "
                "installed. Use POST /ingest/shards or POST /ingest/pcap."
            },
        )

    @app.get("/jobs")
    def jobs() -> list[dict[str, Any]]:
        return [j.public() for j in sorted(state.jobs.values(), key=lambda j: -j.created_at)]

    @app.get("/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        j = state.jobs.get(job_id)
        if j is None:
            raise HTTPException(404, detail="no such job")
        return j.public()

    @app.post("/jobs/{job_id}/cancel")
    def cancel(job_id: str) -> dict[str, Any]:
        j = state.jobs.get(job_id)
        if j is None:
            raise HTTPException(404, detail="no such job")
        j.cancel.set()
        return j.public()

    # -- scoring one flow ---------------------------------------------------------------------

    @app.post("/score/flow")
    def score_flow(req: SC.ScoreFlowRequest) -> dict[str, Any]:
        eng = _engine(state)
        n = len(req.ppi)
        if n == 0:
            return {"decision": "undecidable", "k": 0, "per_k": []}
        ppi = np.zeros((1, P.K_MAX, P.PPI_CHANNELS), dtype=np.int16)
        ppi[0, :n] = np.asarray(req.ppi, dtype=np.int16)
        sc = eng.scorer
        cols = sc.score(ppi, np.array([n]), np.array([1]))
        theta_c, theta_r = eng.theta_commit, np.asarray(sc.theta_reject(eng.f))
        names = sc.class_names
        per_k, first = [], None
        for k in range(1, n + 1):
            pred = int(cols["pred"][0, k - 1])
            d = decide(
                k,
                float(cols["p_safe"][0, k - 1]),
                float(cols["u"][0, k - 1]),
                pred,
                theta_commit=theta_c,
                theta_reject_k=float(theta_r[k - 1]),
                last_k=n,
                k_min=eng.cfg.k_min,
                k_ood=eng.cfg.k_ood,
                order=eng.cfg.order,
            )
            row = {
                "k": k,
                "class_name": names[pred] if 0 <= pred < len(names) else None,
                "p_safe": float(cols["p_safe"][0, k - 1]),
                "u": float(cols["u"][0, k - 1]),
                "theta_reject": float(theta_r[k - 1]),
                "decision": d.name if d else None,
            }
            per_k.append(row)
            if d is not None and first is None:
                first = {**row, "decision": d.name, "class_id": d.cls}
        return {"theta_commit": theta_c, "decision": first, "per_k": per_k}

    # -- reading decisions ---------------------------------------------------------------------

    @app.get("/flows")
    def flows(
        since: int = Query(0, ge=0), limit: int = Query(500, ge=1, le=5000)
    ) -> dict[str, Any]:
        rows = state.store.recent_flows(since, limit)
        return {"flows": rows, "last_id": rows[0]["decision_id"] if rows else since}

    @app.get("/flows/{flow_id}")
    def flow_detail(flow_id: int) -> dict[str, Any]:
        d = state.store.flow_detail(flow_id)
        if d is None:
            raise HTTPException(404, detail="no such flow (it may not be flushed yet)")
        ppi = np.frombuffer(d.pop("ppi"), dtype=np.int16).reshape(P.K_MAX, P.PPI_CHANNELS)
        d.pop("flowstats", None)
        n = int(d["ppi_len"])
        d["packets"] = [
            {
                "ipt_ms": int(r[P.IPT_POS]),
                "dir": int(r[P.DIR_POS]),
                "size": int(r[P.SIZE_POS]),
                "push": int(r[P.PUSH_POS]),
            }
            for r in ppi[:n]
        ]
        if d.get("trace"):
            d["trace"] = json.loads(d["trace"])
        if state.engine is not None:
            th = state.engine.scorer.theta_reject(state.engine.f)
            d["theta_reject"] = [float(x) if np.isfinite(x) else None for x in th]
        explain = getattr(state.engine.scorer if state.engine else None, "explain", None)
        d["explain"] = explain(ppi, n) if (explain is not None and n > 0) else None
        return d

    @app.get("/stats")
    def stats(window_s: float = Query(10.0, gt=0, le=86_400)) -> dict[str, Any]:
        out = state.store.stats(window_s)
        if state.engine is not None:
            m = state.engine.metrics()
            out.update(
                open_flows=m["open_flows"],
                queue_size=m["queue_size"],
                events_per_s=m["events_per_s"],
                batch_ms_p95=m["batch_ms_p95"],
                theta_commit=state.engine.theta_commit,
                budget_mode=state.engine.mode,
                drift_level=state.engine.drift_state()["level"],
                total_decisions=m["decisions"],
                total_rejected=m["rejected"],
                total_undecidable=m["undecidable"],
                total_accuracy=(m["correct"] / m["labelled"]) if m["labelled"] else None,
            )
        out["active_jobs"] = sum(
            1 for j in state.jobs.values() if j.status in ("queued", "running")
        )
        return out

    @app.get("/stats/series")
    def series(
        minutes: float = Query(10.0, gt=0, le=1440), bucket_s: float = Query(5.0, gt=0, le=3600)
    ) -> dict[str, Any]:
        return {"bucket_s": bucket_s, "series": state.store.series(minutes, bucket_s)}

    # -- budget ----------------------------------------------------------------------------------

    @app.get("/budget")
    def budget_get(history: int = Query(200, ge=0, le=5000)) -> dict[str, Any]:
        eng = _engine(state)
        return {
            **eng.budget_state(),
            "history": state.store.controller_history(history) if history else [],
            "offline": _offline_budget(state),
        }

    @app.post("/budget")
    def budget_set(req: SC.BudgetRequest) -> dict[str, Any]:
        eng = _engine(state)
        try:
            return eng.set_budget(
                mode=req.mode,
                metric=req.metric,
                target=req.target,
                theta_commit=req.theta_commit,
                window=req.window,
            )
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc

    # -- unknown and drift ---------------------------------------------------------------------

    @app.get("/unknown")
    def unknown(limit: int = Query(200, ge=1, le=5000)) -> dict[str, Any]:
        p4 = _offline(state, "phase4") or {}
        auroc = (p4.get("h3") or {}).get("auroc", {})
        return {
            "rejected": state.store.rejected(limit),
            "rejection_k_histogram": state.store.rejection_k_histogram(),
            "decision_k_histogram": state.store.decision_k_histogram(),
            "offline_auroc": {s: auroc.get(s) for s in ("u", "energy_raw", "mahal", "surprise")},
            "offline_open_world": (p4.get("h2") or {}).get("open_world"),
        }

    @app.get("/drift")
    def drift(history: int = Query(100, ge=0, le=2000)) -> dict[str, Any]:
        eng = _engine(state)
        p5 = _offline(state, "phase5") or {}
        static = p5.get("static") or {}
        return {
            **eng.drift_state(),
            "history": state.store.drift_history(history) if history else [],
            "offline": {
                "weekly": static.get("weekly"),
                "lead": static.get("lead"),
                "per_signal_first_alarm": static.get("per_signal_first_alarm"),
            },
        }

    # -- model -----------------------------------------------------------------------------------

    @app.get("/model")
    def model_get() -> dict[str, Any]:
        eng = state.engine
        sc = eng.scorer if eng else None
        describe = getattr(sc, "describe", None)
        return {
            "status": state.status,
            "error": state.error,
            "active": describe() if describe else ({"version": sc.version} if sc else None),
            "production": str(state.cfg.production),
            "in_flight_flows": eng.open_flows if eng else 0,
        }

    @app.post("/model")
    def model_reload(req: SC.ModelReloadRequest) -> dict[str, Any]:
        """Reload the bundle. New flows use the new model; flows in flight finish on the old one."""
        eng = _engine(state)
        if not state._reload_lock.acquire(blocking=False):
            raise HTTPException(409, detail="a reload is already running")
        try:
            path = state.root / req.production if req.production else None
            try:
                new = state.load_bundle(path)
            except Exception as exc:  # noqa: BLE001 - the old model keeps serving
                raise HTTPException(422, detail=f"reload failed, old model kept: {exc}") from exc
            state.bundle = new
            eng.swap_scorer(new)
            state._record_model(new)
            return {"active": new.describe(), "in_flight_on_previous": eng.open_flows}
        finally:
            state._reload_lock.release()

    # -- offline results -------------------------------------------------------------------------

    @app.get("/experiments")
    def experiments() -> dict[str, Any]:
        summaries = state.root / "results" / "summaries"
        tables = {}
        for name in ("phase3_table", "phase4_table", "phase5_table", "main_table"):
            p = summaries / f"{name}.md"
            if p.is_file():
                tables[name] = p.read_text(encoding="utf-8")
        criteria = {
            name: _read_json(summaries / f"{name}.json")
            for name in ("phase4_criteria", "phase5_criteria")
        }
        eff = _read_json(state.root / "results" / "phase5" / "efficiency.json")
        return {"tables": tables, "criteria": criteria, "efficiency": eff}

    # -- live stream ------------------------------------------------------------------------

    @app.websocket("/stream/decisions")
    async def stream(ws: WebSocket) -> None:
        await ws.accept()
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        state.ws_clients.add(q)
        try:
            while True:
                event = await q.get()
                await ws.send_json(SC.DecisionEvent(**event).model_dump())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            state.ws_clients.discard(q)

    return app


def _offline(state: ServiceState, which: str) -> Any:
    sc = state.engine.scorer if state.engine else None
    prod = getattr(sc, "production", None)
    if not prod:
        prod = _read_json(state.production_path()) or {}
    path = (prod.get("offline") or {}).get(which)
    return _read_json(state.root / path) if path else None


def _offline_budget(state: ServiceState) -> Any:
    p4 = _offline(state, "phase4") or {}
    cw = ((p4.get("h2") or {}).get("closed_world") or {}).get("safe_stop") or {}
    return {"acc_at_budget": cw.get("acc_at_budget"), "front": cw.get("front")}


__all__ = ["create_app"]
