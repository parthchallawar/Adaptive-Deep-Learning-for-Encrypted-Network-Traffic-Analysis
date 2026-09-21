# ISCX VPN-nonVPN 2016 (D3)

- **Role:** PCAP pipeline validation, standard-benchmark comparability, category-level transfer (spec 001). Results are documented as **secondary** (spec 004) — the labels are a heuristic, not ground truth (see below).
- **Status: partially downloaded, real data exists.** As of 2026-09-21: the user completed registration and downloaded 2 of the 5 real archives (`VPN-PCAPS-01.zip`, `NonVPN-PCAPs-01.zip`); both are extracted and exported for real. **130,303 real flows**, 8 real classes, `data/manifest.json` has a real `iscx-vpn-2016` entry. The remaining 3 archives (`VPN-PCAPs-02.zip`, `NonVPN-PCAPs-02/03.zip`) have not been fetched yet. Everything below marked "real" is measured from this run, not a spec target.
- **Source and retrieval:** `https://www.unb.ca/cic/datasets/vpn.html` links to `cicresearch.ca/CICDataset/ISCX-VPN-NonVPN-2016/`, which serves a **registration form** (`action="insert.php"`, collecting name/email/institution/job title/country), not a file listing — confirmed by fetching the page directly (spec 001's "Correction (2026-09-20)"). This project does not automate that submission. Once registered, the real `/PCAPs` listing turned out to serve **zip archives**, not individual pcaps directly (spec 001's "Correction (2026-09-21)") — `VPN-PCAPs-01/02.zip`, `NonVPN-PCAPs-01/02/03.zip`:
  ```
  python scripts/download_iscx.py --base-url <the post-registration /PCAPs URL> --out data/raw/iscx-vpn-2016
  ```
  (or `--files-from <a local URL list>`). `adl_etc.data.iscx_download.discover_files` finds `*.zip` links generically, downloads and extracts each with `adl_etc.data.download.extract_zip`.
- **License:** no formal open-source license is stated on the dataset page; it is "publicly available for researchers" with a mandatory citation requirement (confirmed by fetching `unb.ca/cic/datasets/vpn.html` directly).
- **Citation:** Draper-Gil, G., Lashkari, A. H., Mamun, M. S. I. & Ghorbani, A. A. "Characterization of Encrypted and VPN Traffic Using Time-Related Features." *Proceedings of the 2nd International Conference on Information Systems Security and Privacy (ICISSP 2016)*, pp. 407-414, Rome, Italy.
- **Size:** ~28 GB pcap total (spec 001); the 2 archives downloaded so far are 1,786.8 MB of real pcap/pcapng data (37 files).
- **Time structure:** none usable (spec 001).

## Class list and support (real, from the 2 archives downloaded so far)

7 categories x VPN/non-VPN condition = 14 possible classes (spec 001's D3
row); these 2 archives cover 8 of them (both are VPN-01 and NonVPN-01, so
`browsing` and `streaming` are entirely absent so far, and no file from the
`p2p`/`file_transfer` categories exists on the non-VPN side yet):

| Class | Category | Condition |
|---|---|---|
| `chat_vpn` | chat | vpn |
| `chat_nonvpn` | chat | nonvpn |
| `mail_vpn` | mail | vpn |
| `mail_nonvpn` | mail | nonvpn |
| `voip_vpn` | voip | vpn |
| `voip_nonvpn` | voip | nonvpn |
| `p2p_vpn` | p2p | vpn |
| `file_transfer_vpn` | file_transfer | vpn |

**130,303 real flows** across all 37 files (1,590 dropped for zero PPI),
`session_id` 0-36 (one per file), `source_manifest_hash` recorded.

Labels are inferred entirely from each pcap's **file name** (e.g.
`facebook_chat_4a.pcap`, `vpn_youtube_A.pcap`) via
`adl_etc.data.iscx_labels.infer_label` — this dataset ships no per-flow
label column at all. **The heuristic was checked against these 37 real
file names for the first time on 2026-09-21: zero unresolved.** This is
real evidence the heuristic (written and tested months earlier against
*reconstructed* file names, never a real listing) holds up, but it is
evidence from 2 of 5 archives, not all of them — every row in `labels.csv`
still carries `label_confidence="heuristic"` rather than being upgraded to
"confirmed" on partial coverage. A file name the heuristic cannot resolve
gets an empty `class_name` rather than a guess, and the downloader prints
every such file so a human reviews it.

## Known issues

- **3 of 5 archives not yet downloaded** — `browsing` and `streaming` categories, and non-VPN `p2p`/`file_transfer`, have zero real coverage so far.
- **The real archive shape was different from what the downloader first assumed** (spec 001's "Correction (2026-09-21)"): zip archives, not flat pcap files — the code has since been corrected and tested, but this was a real gap in what shipped, not merely a documentation lag.
- **A third of the real non-VPN archive is `.pcapng`, not `.pcap`.** Both `extract_zip`'s default suffix match and a separate, independent bug in `export_pcap.py`'s own file-discovery glob (`*.pcap` only, silently skipping `.pcapng` with no error — found and fixed the same day) had to account for this.
- **Heuristic labels, partially checked.** Real, but only against 2 of 5 archives — the central open risk for this dataset specifically until the rest are downloaded.
- **No per-flow ground truth of any kind**, label or otherwise (spec 001).
- **The 60-minute throughput gate was originally measured on a proxy (D4), and the proxy was wrong for this dataset**: the 2026-09-20 estimate (~1.7 MB/s, ~281 minutes for 28 GB) used D4's `Neris.pcap`, a botnet capture with many short bursty flows, because no real ISCX file existed yet to measure directly. The **real** rate, once real ISCX files existed to measure (37 files, 1,786.8 MB, 64.6 s): **27.7 MB/s — 16x faster**, extrapolating to **~17 minutes** for the full 28 GB corpus. See "Decisions" below — the subset-only decision this card previously recorded is reversed.

## Privacy notes

Per spec 001: never store or use IP/port/SNI/JA3/ASN as model inputs
(enforced generically by `pcap_source.py`/`flows.py`, not D3-specific).
This dataset's registration form itself collects a *researcher's* personal
information (name/email/institution), not traffic-subject information —
the reason this project declines to automate it is a different concern
(not impersonating the registrant), documented in
`scripts/download_iscx.py`'s module docstring.

## Decisions this project made

- **Registration is never automated** (spec 001, `download_iscx.py`'s docstring) — the user registered in their own browser first, then supplied the downloaded archives.
- **Full-corpus export is no longer assumed out of scope.** The original decision (a per-class subset only, because a D4-proxy extrapolation suggested 281 minutes) is reversed now that a real ISCX measurement exists: 27.7 MB/s real throughput puts the full 28 GB corpus at ~17 minutes, under the 60-minute gate. Whether to actually fetch and export the remaining 3 archives is a bandwidth/time tradeoff for the user, not a code or throughput blocker anymore.
- **File-name-based labels are explicitly flagged low-confidence** (`label_confidence="heuristic"`) throughout the pipeline (manifest, `labels.csv`, shard `meta.json`) rather than presented as equivalent to D1/D4's ground truth, even though the first real check found zero mismatches.
