# Spec 001: Datasets and Acquisition

- **Status:** draft
- **Owner:** Parth Challawar
- **Created:** 2026-09-17
- **Build step:** step 1 of 18
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
| D1 | CESNET-TLS-Year22, size XS | primary: supervised, drift, open-set | 2.69 GB `.h5` (S: 6.7 GB; raw weekly CSV mirror: about 30 GB) | 180 apps, 24 categories | all of 2022; monthly periods in DataZoo, **weekly/daily in the raw CSV form** | `cesnet-datazoo` download (liberouter.org bucket), or the Kaggle mirror (see Acquisition paths) |
| D2 | CESNET-QUIC22, size XS | cross-protocol transfer; SSL corpus; baseline parity with 30pktTCNET | 2.71 GB `.h5` | 102 apps + 3 background | 4 weeks (W-2022-44..47) | `cesnet-datazoo` |
| D3 | ISCX VPN-nonVPN 2016 | PCAP pipeline validation; standard-benchmark comparability; category-level transfer | about 28 GB pcap (subset of files acceptable) | 14 (7 categories x VPN/non-VPN) | none usable | registration-gated (see below), not direct HTTP |
| D4 | USTC-TFC2016 | "unusual traffic" anomaly experiment | 3.7 GB pcap | 10 benign + 10 malware | none | GitHub mirror (davidyslu/USTC-TFC2016) |
| D5 | Self-captured demo traffic | dashboard demo only | < 1 GB | ad hoc | live | Npcap/Scapy or ipfixprobe |

Optional later: CESNET-TLS22 XS (1.29 GB, 2 weeks, 191 apps) for a second in-distribution check; CESNET-QUICEXT-25 for a second year-long drift study.

## Acquisition paths for D1/D2

Two paths exist and both are supported; the choice affects time granularity and how data reaches Kaggle.

| | Path A: DataZoo HDF5 (canonical) | Path B: Kaggle mirror of the raw release |
|---|---|---|
| Source | `cesnet-datazoo` downloads `CESNET-TLS-Year22-XS.h5` (2.69 GB) from liberouter.org | public Kaggle dataset `pranjalkar99/cesnet-22` (about 30 GB), verified 2026-09-17 to contain `CESNET-TLS-Year22/WEEK-2022-00..52/<date>/flows-*.csv.xz` plus per-day and per-week `stats-*.json`; `zilinpeng/cesnet-quic22` mirrors QUIC22 the same way |
| Granularity | DataZoo periods (monthly for this dataset) | **daily and weekly**, the full 53-week span |
| Loading | DataZoo API: known/unknown class selection, period configs, DataLoaders | our own CSV parser into the spec-003 shard format |
| Local cost | 2.69 GB download, then upload of about 4 GB of shards to a private Kaggle dataset | **zero local download**: mount the mirror inside a Kaggle CPU session, export shards there, save them as that kernel's output dataset |
| Needs internet in kernels | yes (or the shard upload path) | no |
| Trust | canonical, published by CESNET | third-party re-upload: must be verified before use |

**Decision:** Path B is the default for the *drift study* (weekly granularity is what the research question needs and it removes the upload entirely), with Path A used for verification and for anything that benefits from DataZoo's open-set helpers. Verification of the mirror is mandatory before any result depends on it: per-day flow counts must match the `stats-*.json` shipped alongside, the class list must match the 180 services in the official servicemap, and a random sample of 10k flows must match the DataZoo XS HDF5 on the overlapping fields. If verification fails, fall back to Path A and document it.

## How it works

### D1/D2 via DataZoo (Path A)

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

### D1/D2 via the raw weekly CSVs (Path B)

`scripts/export_raw_csv.py` reads `flows-YYYYMMDD.csv.xz` directly (streaming, chunked), parses the PPI columns (packet sizes, directions, inter-packet times as delimited strings) and the flow-statistics columns, drops all identifier columns (SNI, JA3, IPs, ASN, ports) and writes the same shards as Path A, partitioned by ISO week. The same script runs locally or inside a Kaggle kernel with the mirror mounted read-only at `/kaggle/input/cesnet-22/`; in the Kaggle case its output goes to `/kaggle/working/` and is saved as a dataset version for training kernels to mount (spec 015).

A `--verify` mode implements the checks listed above (per-day counts against `stats-*.json`, class list, and a sampled field-level comparison against the DataZoo HDF5 when it is available locally).

### D3/D4 PCAPs

`scripts/download_iscx.py` and `scripts/download_ustc.py` fetch the archives, verify sizes, extract into `data/raw/<dataset>/pcap/`, and write `labels.csv` mapping file name to class (ISCX and USTC label at file level). The PCAP pipeline (spec 002) turns them into PPI shards.

**Correction (2026-09-20), D3 access:** the table above originally said "direct HTTP from cicresearch.ca". The actual page at `unb.ca/cic/datasets/vpn.html` links to `cicresearch.ca/CICDataset/ISCX-VPN-NonVPN-2016/`, which serves a registration form (`action="insert.php"`, collecting name/email/institution/job title/country) rather than a file listing — confirmed by fetching the page directly while building `download_iscx.py`, not assumed. This project does not automate that submission: it is the user's own personal information, and submitting it as if from an automated client rather than the actual registrant would misrepresent who is asking. `scripts/download_iscx.py` therefore requires a `--base-url` (the file-listing URL the user receives after registering in their own browser) or a `--files-from` list, and discovers `*.pcap` links generically from whatever page that turns out to be (`adl_etc.data.iscx_download.discover_files`). D4 (USTC-TFC2016, GitHub-hosted) needs no such step and is fully automated.

ISCX ships no per-flow label column, so `scripts/download_iscx.py` infers each file's class (7 categories x VPN/non-VPN, spec 001's D3 row) from its file name via `adl_etc.data.iscx_labels`, using the dataset's documented category grouping plus the VPN-substring convention. This is a **documented heuristic**, written and tested against plausible file names reconstructed from the widely-cited ISCX naming convention, not verified against the real file listing (which is behind the registration gate above). Every row in `labels.csv` carries a `label_confidence` column (`"heuristic"`); a file the heuristic cannot classify gets an empty `class_name` rather than a guess, and the downloader prints every such file so a human reviews it once real files are in hand.

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
- Whether the raw mirror's flows are the full population or a sample per day (the official release is already sampled; the `stats-*.json` files answer this during verification).
