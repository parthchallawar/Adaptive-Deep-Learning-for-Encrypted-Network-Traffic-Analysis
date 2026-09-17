# Spec 019: Deployment (Docker, Linux)

- **Status:** draft
- **Owner:** Parth Challawar
- **Created:** 2026-09-17
- **Depends on:** 002, 014, 016, 017, 018. **Used by:** demo, evaluators.

## Problem

The system has several moving parts (capture/exporter, inference service, dashboard, database, MLflow) with OS-specific dependencies (ipfixprobe and NFStream are Linux-only; Npcap is Windows-only). Reviewers and the owner need a one-command, reproducible way to run the whole system on a Linux host or on Windows via Docker Desktop/WSL2.

## Goals

- `docker compose up` brings up: `db` (PostgreSQL), `mlflow`, `api` (FastAPI, CPU), `dashboard` (Streamlit), and an optional `exporter` (ipfixprobe) profile for pcap processing and Linux live capture.
- Reproducible images with pinned dependencies (`uv`/`pip-tools` lock file), non-root users, health checks.
- A CPU-only PyTorch image under 2 GB; the ONNX Runtime path for inference.

## Non-goals

- GPU containers (training runs on Kaggle); Kubernetes; cloud deployment.

## Design

```
docker/
  api.Dockerfile          # python:3.11-slim, torch CPU wheel, onnxruntime, app code
  dashboard.Dockerfile
  ipfixprobe.Dockerfile   # builds ipfixprobe with pcap input and pstats plugin
docker-compose.yml        # services + profiles: default, exporter, dev
.env.example              # DB URL, API key, model stage, capture iface
```

- Volumes: `./data` (pcaps, shards), `./results` (MLflow store, checkpoints), `pgdata`.
- `api` waits for `db` and `mlflow` (healthchecks); loads the model from the MLflow registry stage set in `.env`, or from `results/models/production.onnx` if MLflow is unreachable.
- `exporter` profile: `ipfixprobe -i "pcap;file=/data/in.pcap" -p pstats -p basicplus -o "text;..."` writing to a shared volume watched by `api`, or `-i "raw;ifc=eth0"` with `network_mode: host` and `NET_ADMIN`/`NET_RAW` capabilities for live Linux capture.
- Windows live capture stays outside Docker (Npcap + Scapy in the host Python), talking to the `api` container over localhost.
- `make demo` target: starts the stack, loads a bundled 50 MB fixture pcap, opens the dashboard URL.

## Inputs and outputs

- Inputs: `.env`, `docker-compose.yml`, images.
- Outputs: running services on localhost ports 8000 (api), 8501 (dashboard), 5000 (mlflow), 5432 (db).

## Edge cases

- Model artefact missing: `api` starts in "no model" mode with a clear `/health` status instead of crashing.
- Image size bloat from CUDA wheels: install `torch` from the CPU index explicitly.
- Permission errors on bind mounts under WSL2: documented in `docs/deployment.md`.

## Performance considerations

- `api` container limited to 4 CPUs by default; throughput target as in spec 016.

## Testing

- CI job builds the images and runs `docker compose up` with the fixture pcap, then hits `/health`, `/flows`, and checks the decision count against the expected value.

## Interactions

- Packages 016, 017, 018, 014; uses 002's exporter.

## Success criteria

- Fresh Linux VM: `git clone && cp .env.example .env && docker compose up` yields a working dashboard with the fixture pcap processed within 5 minutes.

## Open questions

- Whether to publish images to GHCR for evaluators (default: build locally).
