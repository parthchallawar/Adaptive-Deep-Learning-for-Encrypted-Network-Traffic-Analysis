# CESNET-QUIC22 (D2)

- **Role:** cross-protocol transfer and SSL-corpus baseline parity (spec 001) — a model pretrained on D1 (self-supervised) then fine-tuned on D2 at 1%/10%/100% labels.
- **Status: not acquired.** No file of this dataset has been downloaded, and its raw-CSV schema has not been probed against a real file the way D1's was (plan T5a). Everything below is what spec 001 and this dataset's own publication state; nothing here has been independently checked against real bytes, unlike every other card in this project. Treat every claim in this card as provisional until a T5a-style probe is run for D2 specifically.
- **Source and retrieval:** the mirror `zilinpeng/cesnet-quic22` on Kaggle is named in spec 001 as mirroring QUIC22 the same way `pranjalkar99/cesnet-22` mirrors D1, but this has not been confirmed by listing its files (unlike D1's mirror, which was listed and probed in plan T5a). The canonical source is `cesnet-datazoo`'s `CESNET_QUIC22` loader (Path A).
- **License:** CC BY 4.0 (Creative Commons Attribution 4.0 International), per the dataset's Zenodo record.
- **Citation:** "CESNET-QUIC22: A large one-month QUIC network traffic dataset from backbone lines." *Data in Brief* (2023). DOI: [10.1016/j.dib.2023.108888](https://doi.org/10.1016/j.dib.2023.108888). Zenodo: [10.5281/zenodo.7409924](https://doi.org/10.5281/zenodo.7409924).
- **Size:** 2.71 GB (`.h5`, Path A, spec 001).
- **Time structure:** one month, 2022-10-31 to 2022-11-27, split by spec 004 into `train=W-2022-44, val=W-2022-45, test=W-2022-46, test_drift=W-2022-47`.
- **Classes:** 102 apps + 3 background classes (spec 001).

## What's genuinely unverified here

Every other card in this project either has real data behind it or states
plainly what's blocked and why (D3's registration gate). D2 is different:
it is simply **not yet started**. Specifically unverified, in order of what
a T5a-style probe would need to check first:

1. Whether QUIC's raw-CSV columns match D1's (same `PPI`/`APP`/`CATEGORY`
   naming, same identifier columns to drop) or differ in a QUIC-specific
   way (e.g. no TCP flags, a QUIC Initial Packet field set instead).
2. Whether `zilinpeng/cesnet-quic22` actually contains the same
   `WEEK-YYYY-NN/<date>/flows-<date>.csv.xz` + `stats-*.json` layout D1's
   mirror does, or something else.
3. Whether this project's `udp_idle_timeout` (added in plan T4 for the PCAP
   path, `flows.py`) is validated against any real QUIC/UDP data at all —
   it currently is not: D1's schema probe found zero UDP rows (TLS-only),
   and D2 has not been probed yet either (see `docs/datasets/cesnet-tls-year22.md`'s "Known issues").

## Decisions this project made

- **Deferred past plan T5** (phase 1's data pipeline): T5 scoped itself to D1's Path B exporter; D2 acquisition and schema work is out of scope for this phase's task list and not claimed as done anywhere in `plans/phase-1-data-pipeline.md`.
- **`configs/splits/d2_quic.yaml` exists** with spec 004's target periods recorded, so the split boundaries are fixed and reviewable ahead of the data — the same pattern used for `d1_main.yaml`'s weeks 11-52 before that corpus was exported.
