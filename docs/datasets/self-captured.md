# Self-captured demo traffic (D5)

- **Role:** dashboard demo only (spec 001) — not used for training, evaluation, or any reported result.
- **Status: not captured yet.** Live capture (the Npcap/Scapy sniffer) is explicitly **deferred to phase 5** (spec 016), per this plan's own scope decision: "it has no consumer until the inference service exists, and its only test is a manual smoke test." The *streaming interface* it will feed (`StreamTensorizer`, `data/features.py`) is already built and tested (plan T2), so phase 5 is a driver for existing code, not a rewrite.
- **Source:** the project owner's own devices only, captured live at demo time.
- **License / citation:** not applicable — this is not a published dataset, it is one person's own traffic, captured and discarded per run.
- **Size:** < 1 GB (spec 001).
- **Time structure:** live, ad hoc.
- **Classes:** ad hoc (spec 001) — whatever the demo session happens to generate, not a fixed taxonomy.

## Privacy notes — the entire point of this card

- **Only the owner's own devices**, never anyone else's traffic (spec 001).
- **Never committed to git.** No self-captured pcap or derived shard from this dataset is ever expected to appear in this repository's history; `.gitignore` already excludes `data/raw/` and `data/processed/` generally, and this is an additional, dataset-specific rule on top of that.
- **Deleted after the demo** (spec 001) — this data has no long-term retention purpose, unlike D1-D4.
- Subject to the same header-only, no-payload discipline as every other source in this project (`pcap_source.py` derives sizes from header length fields only, spec 002) — being self-captured does not relax that rule, since the *dashboard viewer* is still a third party relative to whatever traffic gets captured on a shared or work device.

## Decisions this project made

- **Deferred to phase 5**, not built now — recorded in `plans/phase-1-data-pipeline.md`'s scope-decisions table, not a gap discovered here.
- **This card exists now, ahead of the capture code**, so its privacy rules are fixed and reviewable before any live-capture code is written, the same reasoning behind writing `configs/splits/d1_main.yaml`'s period boundaries before the full D1 corpus existed.
