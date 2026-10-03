"""Linear probes on a pretrained PAT's frozen prefix representations (spec 007,
"Evaluation of the pretraining itself"; plan phase 3, T5).

For each K, a multinomial logistic regression on ``z_K`` (the representation
after K packets, taken at each flow's effective K) is trained on a seeded
``label_fraction`` of the training flows and scored on validation. It answers
"how much class information does pretraining alone put into a short prefix",
before any fine-tuning, which is what H1 is about.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from adl_etc.training.datasets import FlowBatches, to_torch


@torch.no_grad()
def _representations(
    model: torch.nn.Module, data: FlowBatches, idx: np.ndarray, ks: tuple[int, ...],
    *, batch_size: int, device: torch.device,
) -> tuple[dict[int, np.ndarray], np.ndarray]:  # fmt: skip
    model.eval()
    out: dict[int, list[np.ndarray]] = {k: [] for k in ks}
    labels: list[np.ndarray] = []
    for start in range(0, len(idx), batch_size):
        sel = idx[start : start + batch_size]
        np_batch = FlowBatches.batch(data, sel)  # the plain batch, never the SSL views
        z = model(to_torch(np_batch, device))["z"].float()
        length = torch.as_tensor(np_batch["ppi_len"], device=z.device)
        rows = torch.arange(len(sel), device=z.device)
        for k in ks:
            out[k].append(z[rows, (torch.clamp(length, max=k) - 1).clamp_min(0)].cpu().numpy())
        labels.append(np_batch["y"])
    return {k: np.concatenate(v) for k, v in out.items()}, np.concatenate(labels)


def _fit_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, n_classes: int,
    *, steps: int, seed: int,
) -> np.ndarray:  # fmt: skip
    torch.manual_seed(seed)
    mean, std = x_train.mean(0), x_train.std(0) + 1e-6
    xt = torch.as_tensor((x_train - mean) / std, dtype=torch.float32)
    yt = torch.as_tensor(y_train, dtype=torch.long)
    head = torch.nn.Linear(xt.shape[1], n_classes)
    opt = torch.optim.Adam(head.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(head(xt), yt).backward()
        opt.step()
    with torch.no_grad():
        logits = head(torch.as_tensor((x_val - mean) / std, dtype=torch.float32))
    return logits.argmax(-1).numpy()


def linear_probe(
    model: torch.nn.Module,
    train: FlowBatches,
    val: FlowBatches,
    *,
    ks: tuple[int, ...] = (3, 5, 8, 30),
    label_fraction: float = 0.1,
    train_cap: int = 50_000,
    val_cap: int = 50_000,
    steps: int = 300,
    seed: int = 0,
    batch_size: int = 4096,
    device: str = "cpu",
) -> dict[str, float]:
    """``{"probe_acc_k3": ..., ...}``: validation accuracy of a linear probe on
    frozen ``z_K`` at each K, trained on ``label_fraction`` of the training
    flows (at most ``train_cap``), scored on up to ``val_cap`` known-class val
    flows."""
    rng = np.random.default_rng([seed, 7])
    n_train = min(train_cap, max(1, round(label_fraction * len(train))))
    train_idx = np.sort(rng.choice(len(train), size=n_train, replace=False))
    known = val.known_indices
    val_idx = np.sort(rng.choice(known, size=min(val_cap, len(known)), replace=False))
    dev = torch.device(device)
    x_tr, y_tr = _representations(model, train, train_idx, ks, batch_size=batch_size, device=dev)
    x_va, y_va = _representations(model, val, val_idx, ks, batch_size=batch_size, device=dev)
    n_classes = train.label_space.n_classes
    out = {}
    for k in ks:
        pred = _fit_predict(x_tr[k], y_tr, x_va[k], n_classes, steps=steps, seed=seed)
        out[f"probe_acc_k{k}"] = float((pred == y_va).mean())
    out["probe_train_flows"] = float(n_train)
    return out
