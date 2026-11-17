# Future work and enhancement proposals

Ideas that are **not** in the current plan, each with what was checked and a verdict, so the checking is not repeated. Written 2026-11-17. None of these starts before phases 4 to 7 are done unless the roadmap says otherwise.

## FEP-1 — Behaviour checks behind an address allow-list (the payment-API fraud case)

**The case.** In late 2026, Delhi Police's cyber unit reported that twelve non-bank lenders (NBFCs) in Delhi-NCR lost over Rs 65 crore in three months. Malware sat in a lender's public front end for about a week and learned the commands that release an approved loan. The attackers then sent those commands from servers rented on the same cloud platforms whose addresses the lender's firewall trusted, with the payee swapped for a mule account. One lender lost about Rs 12 crore in roughly 300 transactions from at least 165 addresses. News coverage of CERT-In's advisory on the campaign (CIAD-2026-0058, "Crafty Falcon") describes attackers invoking victims' own payment APIs with AI-assisted tooling. The advisory itself was not read; only coverage of it.

**The link to this project.** The firewall asked "is this address trusted?". This project asks "what does this encrypted connection behave like, decided in its first few packets, with 'unknown' as a possible answer?". That is the kind of check the case calls for.

**The idea.** Enrol a site's genuine client from a few hundred connections using the frozen pretrained PAT; score each new connection per K against that enrolment; hold anything that does not match before the request completes; add up evidence across connections at the protected endpoint.

**What it would need that does not exist:** (1) the unknown detector (phase 4); (2) examples of the genuine client's traffic; (3) a test with attacker traffic. After phase 4, a model trained on D1 would call *both* a lender's genuine traffic and an attacker's traffic "unknown", because it has seen neither. Enrolment is what separates them.

**Honest assessment** (judgement, not measurement):

| Question | Estimate |
|---|---|
| Can the model separate a genuine client from other tools (curl, Python, a browser)? | likely (~85%) |
| Will it clearly beat a handshake fingerprint (JA4), which already separates different tools? | unlikely (~20%) |
| Can it catch an attacker who uses the same software as the genuine client? | unlikely (~20%) |
| Would it become a result reviewers accept? | unlikely (~15%) |

Reviewers will also ask why the lender does not use proper keys (mutual TLS, signed requests), which is the real fix, and will note that lab traffic is not fraud traffic.

**Verdict.** Use the case as **motivation** in the report and the first slide. Do not claim the system detects such fraud. Optional, after phase 4, as a demo scene only (roadmap DP-6a):

- A mock loan-payment HTTPS server on a second machine or in containers kept outside the repository (traffic inside one Windows machine has no delay and does not split large messages into normal-sized packets).
- One genuine client; attacker clients built with curl, Python, Node, Java and a headless browser; one attacker using the genuine client's own library (the hard case).
- A few hundred runs each, varied amounts and waits, one capture file per client; export with `scripts/export_pcap.py`.
- Enrol on the genuine client only; score the rest at K = 3, 5, 10.
- Pass mark, fixed in advance: within 10 packets, at least 9 in 10 connections from unseen tools flagged, at most 1 in 100 genuine ones flagged, and better than a size-only baseline.
- About one day once phase 4 exists. Everything runs against the project's own mock server.

## FEP-2 — TQH-C2 as an unknown-rejection demo

TQH-C2 (Zenodo record 21523144, v1.2.0, July 2026, CC BY 4.0) holds raw PCAPs of command-and-control tools in a lab: Sliver over TLS (~33 GB), Merlin over QUIC (~4 GB), Mythic over HTTP with an encrypted body (~0.4 GB), Merlin over HTTP/2-TLS (~1.7 GB), with benign background traffic and per-flow labels.

Checked in its README: the authors' own random forest on size and timing features scores about 1.00 on the TLS and QUIC testbeds and 0.98 on the HTTP one; benign traffic is automated requests to a few public sites; the traffic was made in containers; it had three corrective releases in its first week. No peer-reviewed paper was found. The PCAPs themselves were not downloaded.

**Verdict.** Too easy to carry a research claim. Useful once, after phase 4: run the smallest archive (Merlin HTTP/2-TLS) through `scripts/export_pcap.py` and show how many packets the unknown detector needs to flag tools it never saw. Also run the dataset's benign flows; if those are flagged too, the result only shows a domain gap and must be reported that way. One to two days. A presentation slide, not a table in the paper.

## FEP-3 — Label-free adaptation under drift (spec 012 stretch)

Re-fit temperatures, the safety maps and the unknown quantiles on recent confidently-committed flows; refresh the backbone with a few SSL epochs on unlabelled recent traffic; cluster rejected flows to surface new services. Each step has an acceptance gate in spec 012. Needs GPU retraining and its own evaluation on the following weeks. **Verdict:** only if phases 4 to 6 finish early (roadmap DP-5a).

## FEP-4 — Depth exits and a compute budget

Spec 006's optional layer-wise early exit, with spec 011's "compute" budget metric. The streaming model's lazy cache fill is the one genuinely complex piece. **Verdict:** descoped (ADR-018); revisit only if efficiency results show compute, not packets, is the binding cost.

## FEP-5 — Cross-protocol and cross-dataset evaluation

D2 (CESNET-QUIC22) was never exported; D3 (ISCX VPN) was used only to validate the PCAP pipeline. Closing this needs one export-kernel run for D2, a split check, and a score pass. **Verdict:** a limitation in the paper by default (roadmap DP-5b); the cheapest real extension of contribution C4 if time allows.

## FEP-6 — Decisions across flows of one host

The fraud case and botnet beaconing both show attacks whose single connections look ordinary and whose pattern lives across many connections. That needs a model over sequences of flows, with evidence accumulating until a sequential test fires: "anytime" at a second time scale.

CESNET-CC25 (Zenodo 16753890, CNSM 2025) is the dataset built for this. Checked by downloading its botnet flow archive (4 MB): columns are `SRC_IP, SRC_PORT, DST_IP, DST_PORT, PROTOCOL, TIME_FIRST, TIME_LAST, PACKETS, PACKETS_REV, BYTES, BYTES_REV, TCP_FLAGS, TCP_FLAGS_REV, PPI_PKT_TIMES, PPI_PKT_LENGTHS, PPI_PKT_DIRECTIONS, PPI_PKT_FLAGS`, the same kind of per-packet data as D1 in a different layout (absolute timestamps, `|`-separated lists). Of 160,745 botnet flows: all TCP; 5 to ports 443 or 8443; about 9 in 10 have one or two payload packets; a few hundred distinct length-and-direction sequences. Benign traffic is 194M flows (14.9 GB). The authors state that single-flow classification is unreliable on it and that botnet and benign traffic were captured at different points, so inter-packet times can leak the label.

**Verdict.** Wrong dataset for this project's per-flow early classifier (ADR-015). The right dataset for a follow-on project on multi-flow detection.

## FEP-7 — Statistical guarantees for the thresholds

Use Learn-Then-Test to pick `θ_commit` and `θ_reject(k)` with a finite-sample guarantee on error or false-positive rate. Spec 000 lists formal guarantees as a non-goal for the headline claim. **Verdict:** an optional module and a paragraph in the paper; cheap once phase 4's score files exist.

## FEP-8 — Smaller items

| Idea | Note |
|---|---|
| ELECTS-style halting head (spec 009, policy A) | ~3 GPU-hours for three operating points; stretch in phase 4 |
| Live capture on Windows | Needs Npcap; deferred since phase 1 |
| ONNX Runtime and int8 quantisation | Optional in phase 5; appendix material |
| 30pktTCNET public weights as a baseline | Phase-7 A6, if they load within a day |
| A second machine or cloud VM for realistic demo traffic | Only with FEP-1 |

## Other datasets checked on 2026-11-17 and not used

| Dataset | Finding | Verdict |
|---|---|---|
| CIRA-CIC-DoHBrw-2020 | Encrypted DNS tunnels vs browsers; PCAPs available; about 90% malicious | Not about this problem; class balance unusable as is |
| BH-KSU23 | Seven command-and-control frameworks; the record shows flow-summary tables (CICFlowMeter features), per-packet data not confirmed | No |
| Cobalt Strike corpus built from malware-analysis blog posts | Labels derived from published address lists; mostly 2022 captures | Labels too noisy; slow manual collection |
| Stratosphere / CTU malware captures | Many real captures, mixed age and quality | Too much assembly work for the gain |
| A public packet-level "automated client vs human browser" dataset | None found | Would have to be recorded (FEP-1) |

## Sources

- TQH-C2: https://zenodo.org/records/21523144
- CESNET-CC25 flows: https://zenodo.org/records/16753890 ; paper: https://dl.ifip.org/db/conf/cnsm/cnsm2025/1571190192.pdf ; page: https://netmon.fit.cvut.cz/datasets/cesnet-cc25/
- DoHBrw-2020: https://www.unb.ca/cic/datasets/dohbrw-2020.html
- BH-KSU23: https://data.mendeley.com/datasets/wjxc69xj3n
- AgentPrint (fingerprinting LLM agents from encrypted traffic): https://arxiv.org/abs/2510.07176
- JA4-based bot detection: https://arxiv.org/abs/2602.09606
- Cobalt Strike detection from flow metadata: https://arxiv.org/pdf/2506.08922
- Unknown encrypted malicious traffic (HyperVision): https://arxiv.org/pdf/2301.13686
- News coverage of the CERT-In advisory: https://yourstory.com/2026/11/cert-in-warns-nbfcs-and-fintech-firms-of-ai-driven-attacks ; https://www.moneylife.in/article/crafty-falcon-cyber-threat-escalates-as-aidriven-attacks-target-banks-fintechs-and-nbfcs/82128.html
- The Hero FinCorp case: https://theprint.in/india/an-nbfc-317-transactions-rs-12-84-cr-gone-delhi-police-uncovers-cyber-scam-with-china-dubai-links/3045108/
