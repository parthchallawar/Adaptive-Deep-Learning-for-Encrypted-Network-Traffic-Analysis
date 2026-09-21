# USTC-TFC2016 (D4)

- **Role:** "unusual traffic" anomaly experiment — a model trained on D1 (or D3 benign) must flag this dataset's malware flows as unknown/anomalous (spec 004). Not used for its own supervised training.
- **Manifest entry:** `ustc-tfc2016` (`data/manifest.json`).
- **Source and retrieval:** GitHub mirror `davidyslu/USTC-TFC2016`, fully automated (spec 001, no registration gate, unlike D3):
  ```
  python scripts/download_ustc.py --out data/raw/ustc-tfc2016
  ```
- **License:** the GitHub mirror repository itself is tagged `MPL-2.0` (confirmed via the GitHub API), but that tag covers the mirror maintainer's own repository, not necessarily a relicensing of the underlying captures. The traffic data originates from two sources per the original paper (Wang et al., below): 10 malware classes drawn from the CTU malware capture dataset (Czech Technical University, captured 2011-2015) and 10 benign classes captured by the paper's authors via normal application use. **Cite the paper**, not the mirror's repo-level license tag, when using this data.
- **Citation:** Wang, W., Zhu, M., Zeng, X., Ye, X. & Sheng, Y. "Malware traffic classification using convolutional neural network for representation learning." *2017 International Conference on Information Networking (ICOIN)*, pp. 712-717. DOI: [10.1109/ICOIN.2017.7899588](https://doi.org/10.1109/ICOIN.2017.7899588).
- **Size:** 3.8 GB on disk after extraction (24 pcap files — SMB and Weibo's `.7z` archives each held several numbered files; see `extract_7z`'s handling in `data/download.py`), matching spec 001's "3.7 GB pcap" estimate.
- **Time structure:** none usable (spec 001) — captures are independent per-class dumps, not a continuous timeline.

## Class list and support (real — the only dataset card in this project with a complete one)

Exported for real (plan T4): 403,394 flows written, 232,348 dropped for
zero PPI (mostly MySQL/FTP/SMB's control-heavy traffic — pure ACKs and
handshakes with no payload-carrying packet, so nothing to classify by
definition), from `data/processed/ustc-tfc2016/all`:

| Class | Category | Flows |
|---|---|---|
| BitTorrent | benign | 7,502 |
| Cridex | malware | 31,623 |
| FTP | benign | 83,696 |
| Facetime | benign | 6,000 |
| Geodo | malware | 13,734 |
| Gmail | benign | 6,442 |
| Htbot | malware | 8,679 |
| Miuref | malware | 7,487 |
| MySQL | benign | 79,036 |
| Neris | malware | 14,888 |
| Nsis-ay | malware | 7,467 |
| Outlook | benign | 7,475 |
| SMB | benign | 38,459 |
| Shifu | malware | 11,229 |
| Skype | benign | 6,089 |
| Tinba | malware | 10,633 |
| Virut | malware | 8,689 |
| Weibo | benign | 39,860 |
| WorldOfWarcraft | benign | 7,853 |
| Zeus | malware | 6,553 |

Categories: `benign` (282,412 flows, 10 classes) / `malware` (120,982
flows, 10 classes) — the exact binary ground truth spec 004's AUROC
evaluation needs.

## Known issues

- **USTC's `.7z` archives are not internally uniform**: `Shifu.7z` holds one top-level file; `SMB.7z` and `Weibo.7z` hold a subfolder with several numbered files. `extract_7z` (`data/download.py`) handles both shapes generically; this is why the real export has 24 pcap files for 20 classes, not 20:20.
- **File names differ from mirror to mirror** (spec 001) — labelling goes through `labels.csv` built from a directory walk during download, never hard-coded names.
- **Heavy class imbalance**: FTP (83,696) and MySQL (79,036) each carry roughly 10x BitTorrent's or Facetime's flow count. Reported here, not corrected — spec 004 handles imbalance in training, never by dropping test flows.
- **D4 is heavily duplicated at the PPI level (measured 2026-09-21, plan phase 2 T5).** The 403,394 exported flows contain only **41,992 distinct PPI arrays (10.4%)**. Where identical inputs carry different labels no model can do better than the label-majority ceiling, which is **0.9126** over the whole corpus (0.938 in a 512-flow sample). Even among *distinct* PPIs there are near-clashes: in a random 512-flow sample, 234 flows have a differently-labelled neighbour within 0.5 in standardised feature space (minimum distance 0.000). Consequences: (1) a random flow-level split leaks exact duplicates from train into val and inflates every score, so any reported number must use a session-grouped split (`configs/splits/d4_anomaly.yaml`) or a de-duplicated one; (2) D4's role as an *anomaly* corpus is unaffected, since that test scores unknown-vs-known separation and not closed-set accuracy; (3) a small model memorises a distinct-PPI subset slowly (0.955 after 1,600 steps), so overfit sanity checks need a step budget sized to that, see spec 005.
- **Shard order is by capture file**, so a contiguous slice of the shards (or of a `subsample`, which keeps original order) holds disjoint classes: an 80/20 contiguous split of 60,000 flows shared 1 class of 20 between train and val. Always shuffle, or split by `session_id`.
- **232,348 of 635,742 candidate flows (36.5%) carry zero payload-bearing packets** and are dropped rather than assigned a fake single-token classification — see the class table's implicit denominator above.

## Privacy notes

Per spec 001, D5's "self-captured, never committed" rule does not apply
here (this is third-party published research data, not this project's own
capture). No payload bytes are stored or read at any point in this
project's pipeline regardless of source (`pcap_source.py` derives sizes
from header length fields only, spec 002).

## Decisions this project made

- **Used only as an anomaly/open-set test corpus**, never for its own supervised training (spec 004) — the class list above exists to support that evaluation, not a D4-native classifier.
- **The zero-PPI drop rate (36.5%) was accepted, not investigated further**: it reflects genuinely payload-free flows in three specific classes (MySQL/FTP/SMB), not a parsing bug — confirmed by the drop being concentrated in exactly the classes whose protocols are control-heavy.
