# Spec 020: Testing, Quality and CI

- **Status:** draft
- **Owner:** Parth Challawar
- **Created:** 2026-09-17
- **Build step:** continuous, never queued: every spec lands with its own tests
- **Depends on:** all. **Used by:** all.

## Problem

Research code fails silently: a mask bug inflates early-K accuracy, a leakage bug makes drift disappear, a tensoriser mismatch makes the demo disagree with the paper. Each spec lists its own tests; this spec fixes the shared rules, fixtures, tiers and CI so that those tests actually run.

## Goals

- Test tiers: `unit` (< 60 s total, every push), `integration` (< 10 min, nightly or pre-merge), `slow` (dataset downloads, Kaggle smoke; manual).
- Shared fixtures: synthetic flows with known properties, a 200-flow labelled mini-dataset, a fixture pcap with hand-verified PPI, a tiny trained model checkpoint (< 1 MB) for service tests.
- Static quality: `ruff` (lint + format), `mypy` on `src/` (lenient), `pre-commit` hooks, type-checked configs.
- Research-specific invariants enforced by tests: causality, no leakage, calibration monotonicity, policy monotonicity, controller stability, stream/offline equivalence, tensoriser parity.

## Non-goals

- 100% coverage targets; property-based fuzzing of the whole pipeline.

## Layout

```
tests/
  conftest.py            # fixtures: synthetic_flows(), mini_dataset(), fixture_pcap(), tiny_model()
  data/                  # specs 001-004
  models/                # 005, 006
  training/              # 007, 008
  inference/             # 009-012
  evaluation/            # 004, 013
  service/               # 016-018
  dashboard/             # 017
  fixtures/              # pcaps, expected arrays, plots
```

Markers: `@pytest.mark.slow`, `@pytest.mark.gpu`, `@pytest.mark.integration`. Default `pytest` runs unit only.

## Invariant tests (cross-cutting, must never be skipped)

1. **Causality:** outputs at k unchanged by perturbing packets > k (spec 006).
2. **Leakage:** train/test `session_id` disjoint (D3); unknown classes absent from train; normalisation stats hash equals training-period hash (specs 003, 004).
3. **Parity:** DataZoo row vs PCAP-pipeline flow tokenise identically on the fixture (specs 002, 003).
4. **Monotonicity:** thresholds vs mean K / rejection rate (spec 009); isotonic tables (spec 008).
5. **Equivalence:** streaming vs full forward (006); online vs offline policy (009/016).
6. **Controller:** converges in simulation, no oscillation, anti-windup (011).
7. **Reproducibility:** same seed → same report hash on the mini-dataset (014).

## CI (GitHub Actions)

- `ci.yml`: ruff, mypy, unit tests on Ubuntu and Windows (Windows job skips ipfixprobe/NFStream tests).
- `nightly.yml`: integration tests and kernel `--smoke`. (A container smoke test would belong here if spec 019 is ever reactivated.)
- Artefacts: coverage report, smoke report JSON.

## Code conventions

- `src/` is a package (`adl_etc`), imported everywhere (kernel, service, dashboard, notebooks); no notebook-only logic.
- Every public function has a docstring stating tensor shapes.
- Configs validated by dataclass schemas (OmegaConf structured configs).

## Inputs and outputs

- Inputs: source tree, fixtures.
- Outputs: pass/fail, coverage, smoke reports.

## Edge cases

- Tests needing an optional backend (NFStream, ipfixprobe) or Docker: skipped with an explicit reason, never silently passing. The default pure-Python path is always exercised.
- Randomness in tests: all seeded; flaky tests are bugs.

## Testing the tests

- Mutation spot-checks for the invariant tests (e.g. remove the causal mask, confirm the causality test fails) done once and recorded in `docs/testing.md`.

## Interactions

- Every spec's "Testing" section maps to a folder here.

## Success criteria

- Unit suite green on both OSes; invariant tests present and passing before phase 3 results are reported.

## Open questions

- Whether to gate merges on nightly integration (default: no, but failures are tracked in `plans/`).
