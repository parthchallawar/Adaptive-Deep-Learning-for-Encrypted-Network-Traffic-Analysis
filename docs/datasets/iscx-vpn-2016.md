# ISCX VPN-nonVPN 2016 (D3)

- **Role:** PCAP pipeline validation, standard-benchmark comparability, category-level transfer (spec 001). Results are documented as **secondary** (spec 004) — the labels are a heuristic, not ground truth (see below).
- **Status: blocked on registration, code complete.** The real bulk download has not run. Nothing in `data/processed/iscx-vpn-2016/` or `data/manifest.json` exists yet. Everything below that is not explicitly marked "real" is a heuristic or a spec-stated target, not a measurement.
- **Source and retrieval:** `https://www.unb.ca/cic/datasets/vpn.html` links to `cicresearch.ca/CICDataset/ISCX-VPN-NonVPN-2016/`, which serves a **registration form** (`action="insert.php"`, collecting name/email/institution/job title/country), not a file listing — confirmed by fetching the page directly (spec 001's "Correction (2026-09-20)"). This project does not automate that submission: it is the user's own personal information, and submitting it as if from an automated client would misrepresent who is asking. Once the user registers in their own browser and receives a file-listing URL:
  ```
  python scripts/download_iscx.py --base-url <the post-registration URL> --out data/raw/iscx-vpn-2016
  ```
  (or `--files-from <a local URL list>`). `adl_etc.data.iscx_download.discover_files` finds `*.pcap` links generically from whatever page it's given.
- **License:** no formal open-source license is stated on the dataset page; it is "publicly available for researchers" with a mandatory citation requirement (confirmed by fetching `unb.ca/cic/datasets/vpn.html` directly).
- **Citation:** Draper-Gil, G., Lashkari, A. H., Mamun, M. S. I. & Ghorbani, A. A. "Characterization of Encrypted and VPN Traffic Using Time-Related Features." *Proceedings of the 2nd International Conference on Information Systems Security and Privacy (ICISSP 2016)*, pp. 407-414, Rome, Italy.
- **Size:** ~28 GB pcap (spec 001); a full-corpus export is not planned (see "Decisions" below) — a per-class subset is.
- **Time structure:** none usable (spec 001).

## Class list (heuristic, unverified — the central caveat of this card)

7 categories x VPN/non-VPN condition = 14 classes (spec 001's D3 row):
`browsing`, `chat`, `streaming`, `mail`, `voip`, `p2p`, `file_transfer`,
each in a `_vpn` and a non-VPN variant. Inferred entirely from each pcap's
**file name** (e.g. `facebook_chat_4a.pcap`, `vpn_youtube_A.pcap`) via
`adl_etc.data.iscx_labels.infer_label` — this dataset ships no per-flow
label column at all (spec 001: "none usable" time structure applies to
labels too, not just time).

The category grouping comes from the dataset's own published description
and from an independent third-party analysis of this dataset's naming
convention (`Mr-Pepe/iscx-analysis` on GitHub); the multi-category apps
(Skype, Facebook, Hangouts — each spans Chat *and* VoIP, Skype also File
Transfer) are this project's own best-effort reconstruction, **not checked
against a real file listing**, because the real file names are behind the
registration gate above. Every row `labels.csv` produces carries
`label_confidence="heuristic"`; a file name the heuristic cannot resolve
gets an empty `class_name` rather than a guess, and the downloader prints
every such file so a human reviews it once real files are in hand.
**Re-verify this mapping against real file names before any D3 result is
reported** — this is the single most important thing to do before trusting
this card's numbers, and the reason spec 004 already calls D3's results
secondary.

## Known issues

- **Registration-gated access** (above) — blocks the real download entirely until the user acts outside this project.
- **Heuristic labels, unverified** (above) — the central open risk for this dataset specifically.
- **No per-flow ground truth of any kind**, label or otherwise (spec 001).
- **28 GB is too slow to export in full**: measured on real D4 captures (the closest real proxy available, since no ISCX capture exists yet), the pure-Python `dpkt` backend runs ~1.7 MB/s after plan T4's performance fix — extrapolated, the full 28 GB corpus would take ~281 minutes, past the 60-minute gate this plan set for itself. See "Decisions" below.

## Privacy notes

Per spec 001: never store or use IP/port/SNI/JA3/ASN as model inputs
(enforced generically by `pcap_source.py`/`flows.py`, not D3-specific).
This dataset's registration form itself collects a *researcher's* personal
information (name/email/institution), not traffic-subject information —
the reason this project declines to automate it is a different concern
(not impersonating the registrant), documented in
`scripts/download_iscx.py`'s module docstring.

## Decisions this project made

- **Registration is never automated** (spec 001, `download_iscx.py`'s docstring) — the user must register in their own browser first.
- **Full-corpus export is out of scope**; D3 will be exported as a **per-class subset** (`export_pcap.py --files <glob>`) once unblocked, per the measured throughput gate above (plan T4's progress log has the exact numbers).
- **File-name-based labels are explicitly flagged low-confidence** (`label_confidence="heuristic"`) throughout the pipeline (manifest, `labels.csv`, shard `meta.json`) rather than presented as equivalent to D1/D4's ground truth.
