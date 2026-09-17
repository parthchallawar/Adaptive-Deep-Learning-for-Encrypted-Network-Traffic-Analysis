# Spec 013: Efficiency and Latency Benchmarking

- **Status:** draft
- **Owner:** Parth Challawar
- **Created:** 2026-09-17
- **Depends on:** 005, 006, 009, 011. **Used by:** 004 (report), 016.

## Problem

"Reducing the computational cost of traffic classification" is a stated objective. Cost has two very different axes here: **observation cost** (how many packets must be buffered and how long the decision is delayed) and **compute cost** (FLOPs, latency, memory per flow). A model can be cheap on one and expensive on the other. Every claim about efficiency must be measured the same way for all models and policies, on the same hardware, with the same batch regime.

## Goals

- Standard measurements for every model: parameter count, FLOPs per full flow and per packet step, GPU throughput (flows/s at batch 4096), CPU latency per flow (batch 1 and batch 64), streaming latency per packet step, peak memory.
- For policies: mean/median/p95 packets-to-decision, mean time-to-decision in ms of flow time (from the IPT sequence), mean layers evaluated (if depth exits), and compute per flow under the spec-006 cost model.
- A single script producing the efficiency table and the "accuracy vs cost" scatter used in the report.

## Non-goals

- Hardware-specific optimisation beyond ONNX Runtime and TorchScript; quantisation-aware training (int8 post-training quantisation is measured but not optimised for).

## Method

- FLOPs: `torch.utils.flop_counter.FlopCounterMode` on a batch of 1, reported per flow and per packet step (streaming path).
- Latency: warm-up 50 iterations, then 500 timed iterations, median and p95; CPU runs pinned to 4 threads; GPU runs use CUDA events. Both PyTorch eager and ONNX Runtime (CPU) are reported for our model; baselines in eager only.
- Streaming latency: `PATStream.step` timed over 1000 flows x 30 steps on CPU; this is the service-relevant number.
- Memory: `torch.cuda.max_memory_allocated` (GPU) and RSS (CPU) at batch 64.
- Time-to-decision: for each flow, the sum of IPTs up to the decision packet (how long an operator waits in wall-clock terms), reported as a distribution.
- Energy proxy (optional): GPU power via `nvidia-smi` sampling during throughput runs.

Hardware documented per run: Kaggle T4/P100 for GPU numbers, the owner's laptop CPU (model, cores, RAM) for CPU numbers, recorded in `results/hardware.json`.

## Inputs and outputs

- Inputs: model checkpoints or ONNX files; policy replay results (spec 009); `configs/eval/efficiency.yaml`.
- Outputs: `results/<run>/efficiency.json`, table in `results/summaries/efficiency.md`, scatter plot accuracy vs (mean K, ms/flow, FLOPs).

## Edge cases

- XGBoost has no FLOPs in the DL sense: report tree count and depth, and latency only.
- 30pktTCNET is non-causal: streaming latency reported as "one full pass per packet" (its real cost in an anytime setting), which is part of the argument for causal models.
- Thermal throttling on the laptop: interleave models and repeat twice; report the minimum median.

## Performance considerations

- The benchmark itself should finish in under 30 minutes for all models.

## Testing

- Smoke test that all registered models produce all efficiency fields.
- Consistency test: FLOPs per full flow equals the sum over packet steps for the causal model within 5%.

## Interactions

- Feeds spec 004's main table and the paper's efficiency section; spec 016 uses the ONNX export and reports live latency.

## Success criteria

- PAT streaming step latency <= 1 ms on CPU (batch 1) and full-flow CPU latency <= 5 ms; parameter count < 2M; efficiency table complete for all baselines and policies.

## Open questions

- Whether to include int8 ONNX quantisation results in the paper (default: appendix only).
