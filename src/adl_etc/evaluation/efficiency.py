"""Efficiency measurements (plan phase 5, T3; spec 013).

Two kinds of cost, kept apart on purpose:

* **observation cost**: how many packets are read and how long, in the flow's own clock, an
  operator waits before the decision (:func:`policy_cost`, NumPy over saved scores plus the
  shards' inter-packet times);
* **compute cost**: parameters, FLOPs, and CPU latency of a model (:func:`bench_model`; torch is
  imported inside the functions, like ``plotstyle.py``, so this module imports without it).

Latencies are wall-clock on whatever CPU runs them and depend on that machine and its load; the
hardware is recorded by :func:`hardware_info`. Each model is timed twice, interleaved, and the
smaller median is kept (laptop thermal throttling).
"""

from __future__ import annotations

import platform
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from adl_etc.evaluation import controller_replay as CR
from adl_etc.evaluation import replay as R
from adl_etc.evaluation.scores import K_MAX, ScoreArrays

# --- hardware -----------------------------------------------------------------------------------


def hardware_info(threads: int) -> dict[str, Any]:
    import os

    import torch

    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_logical": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": threads,
        "device": "cpu",
    }
    try:  # psutil is optional: it only adds the physical core count, clock and RAM
        import psutil

        freq = psutil.cpu_freq()
        info["cpu_physical"] = psutil.cpu_count(logical=False)
        info["cpu_max_mhz"] = None if freq is None else round(freq.max, 0)
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    return info


# --- timing -------------------------------------------------------------------------------------


def time_calls(fn: Callable[[], Any], *, warmup: int = 50, iters: int = 500) -> dict[str, float]:
    """Median and p95 wall-clock milliseconds of ``fn()`` after ``warmup`` untimed calls."""
    for _ in range(warmup):
        fn()
    ms = np.empty(iters)
    for i in range(iters):
        t = time.perf_counter()
        fn()
        ms[i] = (time.perf_counter() - t) * 1e3
    return {
        "median_ms": float(np.median(ms)),
        "p95_ms": float(np.percentile(ms, 95)),
        "mean_ms": float(ms.mean()),
        "iters": float(iters),
    }


def synthetic_batch(model: Any, b: int, seed: int = 0) -> dict[str, Any]:
    """A batch of ``b`` random 30-packet flows with every position real. Token ids are drawn
    inside the model's own embedding tables (a baseline ignores ``tokens``)."""
    import torch

    g = torch.Generator().manual_seed(seed)
    tokens = torch.stack(
        [
            torch.randint(1, int(getattr(model, name).num_embeddings), (b, K_MAX), generator=g)
            for name in ("emb_ipt", "emb_dir", "emb_size", "emb_push")
        ]
        if hasattr(model, "emb_ipt")
        else [torch.zeros(b, K_MAX, dtype=torch.long)] * 4,
        dim=-1,
    )
    return {
        "tokens": tokens,
        "cont": torch.randn(b, K_MAX, 4, generator=g),
        "mask": torch.ones(b, K_MAX, dtype=torch.bool),
    }


def count_params(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def count_flops(fn: Callable[[], Any]) -> int:
    """Floating-point operations of one call of ``fn`` (``FlopCounterMode``: matrix products,
    convolutions and fused attention; element-wise operations are not counted)."""
    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode(display=False) as counter:
        fn()
    return int(counter.get_total_flops())


# --- one model ----------------------------------------------------------------------------------


def _kind(model: Any) -> str:
    name = type(model).__name__.lower()
    if "pat" in name:
        return "pat"
    return "gru" if hasattr(model, "rnn") else "cnn"


def bench_model(
    model: Any,
    *,
    warmup: int = 50,
    iters: int = 500,
    stream_flows: int = 200,
    memory_batch: int = 64,
) -> dict[str, Any]:
    """Parameters, FLOPs and CPU latency of one model (already on the CPU, in ``eval`` mode)."""
    import torch

    model.eval()
    kind = _kind(model)
    b1, b64 = synthetic_batch(model, 1), synthetic_batch(model, 64)
    out: dict[str, Any] = {"kind": kind, "params": count_params(model)}

    with torch.no_grad():
        out["flops_full_flow"] = count_flops(lambda: model(b1))
        out["latency_batch1"] = time_calls(lambda: model(b1), warmup=warmup, iters=iters)
        out["latency_batch64"] = time_calls(
            lambda: model(b64), warmup=max(warmup // 5, 5), iters=max(iters // 5, 20)
        )
        out["flows_per_s_batch64_1thread_equiv"] = 64.0 / (
            out["latency_batch64"]["median_ms"] / 1e3
        )

        if kind == "pat":
            out.update(_pat_stream(model, stream_flows))
        elif kind == "gru":
            out.update(_rnn_stream(model, stream_flows))
        else:
            # non-causal: deciding after every packet means one full pass per packet
            full = out["latency_batch1"]["median_ms"]
            out["stream_step"] = {"median_ms": full, "p95_ms": out["latency_batch1"]["p95_ms"]}
            out["stream_note"] = "non-causal: one full pass per packet"
            out["stream_total_30_ms"] = full * K_MAX
            out["flops_stream_total_30"] = out["flops_full_flow"] * K_MAX
    out["weights_mb"] = out["params"] * 4 / 2**20
    out["rss_mb_after_batch"] = _rss_after(lambda: model(b64), memory_batch)
    return out


def _pat_stream(model: Any, n_flows: int) -> dict[str, Any]:
    """``PATStream.step`` (KV cache) timed over ``n_flows`` flows of 30 packets, batch 1, and its
    FLOPs summed over the whole flow, to compare with the full pass."""
    import torch

    from adl_etc.models.pat import PATStream

    batch = synthetic_batch(model, n_flows, seed=1)
    stream = PATStream(model)
    ms: list[float] = []
    for i in range(n_flows):
        tok, cont = batch["tokens"][i : i + 1], batch["cont"][i : i + 1]
        with torch.no_grad():
            stream.start(1)
            for k in range(K_MAX):
                t = time.perf_counter()
                stream.step(tok[:, k], cont[:, k])
                ms.append((time.perf_counter() - t) * 1e3)
    arr = np.array(ms)
    warm = arr[K_MAX * 5 :] if len(arr) > K_MAX * 10 else arr  # drop the first flows
    tok, cont = batch["tokens"][:1], batch["cont"][:1]

    def one_flow() -> None:
        stream.start(1)
        for k in range(K_MAX):
            stream.step(tok[:, k], cont[:, k])

    with torch.no_grad():
        flops = count_flops(one_flow)
    return {
        "stream_step": {
            "median_ms": float(np.median(warm)),
            "p95_ms": float(np.percentile(warm, 95)),
            "mean_ms": float(warm.mean()),
            "steps": float(len(warm)),
        },
        "stream_total_30_ms": float(warm.mean() * K_MAX),
        "flops_stream_total_30": flops,
    }


def _rnn_stream(model: Any, n_flows: int) -> dict[str, Any]:
    """A GRU carries its hidden state, so one packet is one cell step (stem, one recurrent step,
    head); this times that step directly, which is what a streaming deployment would run."""
    import torch

    batch = synthetic_batch(model, n_flows, seed=1)
    ms: list[float] = []
    with torch.no_grad():
        for i in range(n_flows):
            h = None
            cont = batch["cont"][i : i + 1]
            for k in range(K_MAX):
                t = time.perf_counter()
                out, h = model.rnn(model.stem(cont[:, k : k + 1]), h)
                model.head(out)
                ms.append((time.perf_counter() - t) * 1e3)
    arr = np.array(ms)
    warm = arr[K_MAX * 5 :] if len(arr) > K_MAX * 10 else arr
    cont1 = batch["cont"][:1]

    def one_flow() -> None:
        h = None
        for k in range(K_MAX):
            out, h = model.rnn(model.stem(cont1[:, k : k + 1]), h)
            model.head(out)

    with torch.no_grad():
        flops = count_flops(one_flow)
    return {
        "stream_step": {
            "median_ms": float(np.median(warm)),
            "p95_ms": float(np.percentile(warm, 95)),
            "mean_ms": float(warm.mean()),
            "steps": float(len(warm)),
        },
        "stream_total_30_ms": float(warm.mean() * K_MAX),
        "flops_stream_total_30": flops,
    }


def _rss_after(fn: Callable[[], Any], n: int) -> float | None:
    """Process resident memory (MB) after running ``fn`` a few times, or None without psutil. A
    coarse figure: it includes the interpreter and every library loaded, so only differences
    between models run in the same process mean anything."""
    try:
        import psutil
    except ImportError:
        return None
    for _ in range(max(n // 16, 3)):
        fn()
    return float(psutil.Process().memory_info().rss / 2**20)


def bench_all(
    models: Mapping[str, Any],
    *,
    rounds: int = 2,
    threads: int = 4,
    **kw: Any,
) -> dict[str, Any]:
    """Time every model ``rounds`` times, interleaved (``a b c, a b c``), keep per metric the
    round with the smaller batch-1 median, and record the spread between rounds."""
    import torch

    torch.set_num_threads(threads)
    results: dict[str, list[dict[str, Any]]] = {n: [] for n in models}
    for _ in range(rounds):
        for name, model in models.items():
            results[name].append(bench_model(model, **kw))
    out: dict[str, Any] = {}
    for name, runs in results.items():
        best = min(runs, key=lambda r: r["latency_batch1"]["median_ms"])
        meds = [r["latency_batch1"]["median_ms"] for r in runs]
        out[name] = {**best, "batch1_median_ms_by_round": meds}
    return out


# --- observation cost of the policies -------------------------------------------------------------

POLICY_LABELS = {
    "safe_stop": "Safe-Stop",
    "p_echo": "P-ECHO",
    "p_cape": "P-CAPE",
    "p_rl": "P-RL",
}


def time_to_decision_ms(ipt_ms: np.ndarray, ppi_len: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Milliseconds of flow time until the decision packet: the inter-packet times of packets
    1..min(k, ppi_len), summed. ``ipt_ms`` is ``[N, 30]`` (the first packet's value is its own
    offset, normally 0)."""
    kk = np.minimum(np.asarray(k), np.asarray(ppi_len)).astype(np.int64)
    pos = np.arange(ipt_ms.shape[1])[None, :]
    return (np.where(pos < kk[:, None], ipt_ms.astype(np.float64), 0.0)).sum(axis=1)


def _dist(x: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
    }


def operating_points(
    val: ScoreArrays, test: ScoreArrays, n_classes: int, *, target_mean_k: float
) -> dict[str, tuple[R.ReplayResult, float]]:
    """Each policy's decisions on the known flows of ``test`` at the knob whose mean K on the known
    flows of ``val`` is closest to ``target_mean_k`` (every fit on validation only). Returns
    ``{policy: (result, mean_k_on_val)}``; the results are aligned with ``test``'s known flows."""
    from adl_etc.evaluation import phase4 as P4
    from adl_etc.evaluation import rl_stop

    val_k, test_k = R.closed_world(val), R.closed_world(test)
    zero_u = np.zeros((len(test_k), K_MAX))
    out: dict[str, tuple[R.ReplayResult, float]] = {}

    def pick(cands: Sequence[tuple[float, float]]) -> float:
        return min(cands, key=lambda c: abs(c[1] - target_mean_k))[0]

    # Safe-Stop: bisect theta to the target on validation
    theta, _, got = CR.calibrate_theta(val_k, np.arange(len(val_k)), target_mean_k)
    out["safe_stop"] = (
        R.safe_stop(test_k.p_safe, zero_u, test_k.pred, test_k.ppi_len, theta, R.NO_REJECTION),
        got,
    )
    # P-ECHO: grid on tau
    taus = np.concatenate([np.linspace(0.02, 0.5, 40), R.logit_grid(160)])
    cands = [
        (float(t), float(R.echo(val_k.conf, val_k.pred, val_k.ppi_len, float(t)).k.mean()))
        for t in taus
    ]
    tau = pick(cands)
    out["p_echo"] = (
        R.echo(test_k.conf, test_k.pred, test_k.ppi_len, tau),
        dict(cands)[tau],
    )
    # P-CAPE: grid on the per-class precision target
    cands = []
    for prec in np.linspace(0.5, 0.999, 60):
        tau_c = P4.cape_thresholds(val_k, n_classes, float(prec))
        r = R.cape(val_k.conf, val_k.pred, val_k.energy_raw, val_k.ppi_len, tau_c, np.inf)
        cands.append((float(prec), float(r.k.mean())))
    best_prec = pick(cands)
    tau_c = P4.cape_thresholds(val_k, n_classes, best_prec)
    out["p_cape"] = (
        R.cape(test_k.conf, test_k.pred, test_k.energy_raw, test_k.ppi_len, tau_c, np.inf),
        dict(cands)[best_prec],
    )
    # P-RL: grid on the waiting cost
    cands = []
    for cost in np.logspace(-3, 1.5, 60):
        pol = rl_stop.fit_rl(val_k, float(cost))
        cands.append((float(cost), float(rl_stop.apply_rl(pol, val_k).k.mean())))
    best_cost = pick(cands)
    pol = rl_stop.fit_rl(val_k, best_cost)
    out["p_rl"] = (rl_stop.apply_rl(pol, test_k), dict(cands)[best_cost])
    return out


def policy_cost(
    val: ScoreArrays,
    test: ScoreArrays,
    ipt_ms: np.ndarray,
    n_classes: int,
    *,
    target_mean_k: float = 6.0,
) -> dict[str, Any]:
    """Packets-to-decision and time-to-decision of every policy on the known flows of ``test`` at a
    matched mean K (chosen on validation). ``ipt_ms`` is ``[N_test, 30]``, row-aligned with
    ``test`` (not with its known subset)."""
    known = test.labels >= 0
    ipt_known = ipt_ms[known]
    test_k = test.take(known)
    out: dict[str, Any] = {"target_mean_k": target_mean_k, "n_flows": int(known.sum())}
    for name, (res, val_mean_k) in operating_points(
        val, test, n_classes, target_mean_k=target_mean_k
    ).items():
        t = time_to_decision_ms(ipt_known, test_k.ppi_len, res.k)
        full = time_to_decision_ms(ipt_known, test_k.ppi_len, np.full(len(res.k), K_MAX))
        out[name] = {
            "label": POLICY_LABELS[name],
            "val_mean_k": val_mean_k,
            "packets": _dist(res.k.astype(np.float64)),
            "time_ms": _dist(t),
            "time_share_of_flow": float(t.mean() / max(full.mean(), 1e-9)),
            "accuracy": float((res.prediction == test_k.labels).mean()),
        }
    return out
