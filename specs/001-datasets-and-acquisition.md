# Spec 001: Datasets and Acquisition

- **Status:** draft
- **Owner:** Parth Challawar
- **Created:** 2026-09-17
- **Depends on:** none. **Used by:** 002, 003, 004, 015.

## Problem

The project needs (a) a large, labeled, time-stamped corpus of encrypted flows with per-packet metadata for the main research claims, (b) raw PCAPs to prove the end-to-end pipeline, (c) an unlabeled corpus for self-supervised pretraining, and (d) a malicious-traffic corpus for the "unusual traffic" experiment. The corpora must be downloadable without institutional access, fit local disk and Kaggle quotas, and be usable without violating privacy.

## Goals

- One reproducible script per dataset that downloads, verifies (size/hash) and registers it in a local manifest.
- Uniform on-disk layout under `data/raw/<dataset>/` and `data/processed/<dataset>/`.
- Dataset cards (`docs/datasets/<name>.md`) documenting license, citation, classes, known issues.

## Non-goals

- Re-hosting any dataset publicly. Kaggle copies are private.
- Full-size CESNET downloads (136 GB) in this project.

## Datasets

| ID | Dataset | Role | Size (download) | Classes | Time structure | Access |
|---|---|---|---|---|---|---|
| D1 | CESNET-TLS-Year22, size XS | primary: supervised, drift, open-set | 2.69 GB `.h5` (S: 6.7 GB) | 180 apps, 24 categories | all of 2022, monthly periods in DataZoo | `cesnet-datazoo` download (liberouter.org bucket) |
| D2 | CESNET-QUIC22, size XS | cross-protocol transfer; SSL corpus; baseline parity with 30pktTCNET | 2.71 GB `.h5` | 102 apps + 3 background | 4 weeks (W-2022-44..47) | `cesnet-datazoo` |
| D3 | ISCX VPN-nonVPN 2016 | PCAP pipeline validation; standard-benchmark comparability; category-level transfer | about 28 GB pcap (subset of files acceptable) | 14 (7 categories x VPN/non-VPN) | none usable | direct HTTP from cicresearch.ca |
| D4 | USTC-TFC2016 | "unusual traffic" anomaly experiment | 3.7 GB pcap | 10 benign + 10 malware | none | GitHub mirror (davidyslu/USTC-TFC2016) |
| D5 | Self-captured demo traffic | dashboard demo only | < 1 GB | ad hoc | live | Npcap/Scapy or ipfixprobe |

Optional later: CESNET-TLS22 XS (1.29 GB, 2 weeks, 191 apps) for a second in-distribution check; CESNET-QUICEXT-25 for a second year-long drift study.

## How it works

### D1/D2 via DataZoo

```python
from cesnet_datazoo.datasets import CESNET_TLS_Year22
from cesnet_datazoo.config import DatasetConfig, AppSelection, ValidationApproach

ds = CESNET_TLS_Year22("data/raw/CESNET-TLS-Year22", size="XS")   # downloads once, resumable
cfg = DatasetConfig(
    dataset=ds,
    apps_selection=AppSelection.FIXED,           # explicit known/unknown lists (spec 004)
    apps_selection_fixed_known=KNOWN_150,
    apps_selection_fixed_unknown=UNKNOWN_30,
    train_period_name="M-2022-3",                # extended to 3..6 by concatenation
    test_period_name="M-2022-8",
    val_approach=ValidationApproach.VALIDATION_DATES,
    return_tensors=False,                        # we export numpy shards ourselves
)
ds.set_dataset_config_and_initialize(cfg)
```

The exporter script `scripts/export_datazoo.py` iterates the DataZoo dataframes for each period and writes shards (spec 003) so that Kaggle never needs DataZoo or internet.

### D3/D4 PCAPs

`scripts/download_iscx.py` and `scripts/download_ustc.py` fetch the archives, verify sizes, extract into `data/raw/<dataset>/pcap/`, and write `labels.csv` mapping file name to class (ISCX and USTC label at file level). The PCAP pipeline (spec 002) turns them into PPI shards.

### Manifest

`data/manifest.json` records for each dataset: source URL, version/date, bytes, SHA-256 of archives, extraction date, and the git commit of the exporter. Tests fail if a shard's manifest hash is missing.

## Inputs and outputs

- Inputs: network access, `data/raw/` write access, dataset choice from `configs/data/*.yaml`.
- Outputs: raw archives in `data/raw/`, dataset cards in `docs/datasets/`, `data/manifest.json`.

## Edge cases

- DataZoo download interrupted: `resumable_download` resumes; the script re-verifies file size against the HTTP `Content-Range` total.
- ISCX host slow or partial (files > 1 GB): download per file with retries; a `--files` flag limits to a subset (e.g. one file per class) for development.
- USTC mirror renamed files: map via `labels.csv`, not by hard-coded names.
- Week-10 exporter artefact in D1: excluded by the split definition (spec 004), documented in the card.
- Class imbalance in D1 (heavy-tailed): reported in the card; handled in training (spec 008), never by dropping test flows.

## Privacy and ethics

- D1/D2 are already anonymised by CESNET (source IPs anonymised, no payload). We never store or use SNI/JA3/IP/ASN as model inputs; they are dropped at export (spec 003).
- Self-captured traffic (D5) is only from the owner's own devices, never committed to git, and deleted after the demo.

## Performance considerations

- D1 XS in DataZoo's HDF5 form needs about 3 GB disk plus about 1 GB of statistics cache; export to shards needs another 3 GB.
- Exporting 10M rows via pandas is memory-heavy; the exporter streams period by period and writes shards of 500k flows.

## Testing

- `tests/data/test_manifest.py`: every dataset in configs has a manifest entry with a hash.
- `tests/data/test_datazoo_export.py` (marked `slow`, skipped in CI): exports 10k flows and checks shapes, dtypes, label ranges, and that unknown classes are absent from train.
- Dataset card review checklist: license, citation, class list, known issues, drift artefacts.

## Interactions

- Spec 002 consumes D3/D4/D5 PCAPs. Spec 003 defines the shard format for all datasets. Spec 004 fixes the known/unknown lists and periods. Spec 015 uploads shards to Kaggle.

## Success criteria

- `python scripts/download_all.py --datasets d1 d2` completes on a fresh machine and produces a valid manifest.
- Dataset cards exist for D1 to D4.

## Open questions

- Whether to also pull D1 at size S (25M flows, 6.7 GB) for a final "scale" experiment near the end. Default: no.
- ISCX subset size for the category-level transfer experiment (default: all pcaps, since disk allows it).
