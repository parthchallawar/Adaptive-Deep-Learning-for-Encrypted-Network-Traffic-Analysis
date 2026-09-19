# Spec 003: Feature Representation and Preprocessing

- **Status:** partially implemented (PPI schema and flow statistics built; shard writer and tokeniser pending)
- **Owner:** Parth Challawar
- **Created:** 2026-09-17
- **Build step:** step 3 of 18
- **Depends on:** 001, 002. **Used by:** 005 to 011, 015, 016.

## Problem

Every model, baseline and the serving path must see exactly the same numbers. The representation must (a) be identical whether flows come from DataZoo or from our PCAP pipeline, (b) support prefixes of any length K with a mask, (c) be normalised with training-only statistics, (d) exclude shortcut features (IPs, ports, SNI, ASN), and (e) be compact enough to hold 10M flows in RAM on Kaggle.

## Goals

- One shard format (`.npz` or memory-mapped `.npy`) and one `FlowTensorizer` used everywhere.
- Deterministic, versioned normalisation with statistics stored next to the shards.
- Token-level representations that suit both embedding-based Transformers and continuous-input CNN/RNN baselines.

## Non-goals

- Learned tokenisers or byte-level inputs.
- Feature selection experiments beyond the ablations in spec 004.

## Shard schema

Per dataset and period, shards of up to 500k flows:

| Array | dtype | shape | Notes |
|---|---|---|---|
| `ppi` | int16 | [N, 30, 4] | raw `[ipt_ms, dir, size, push]`; zeros beyond `ppi_len` |
| `ppi_len` | int8 | [N] | number of recorded payload packets, 1..30 |
| `flowstats` | float32 | [N, 43] | raw (unscaled) values, order fixed in `FLOWSTATS_COLUMNS` |
| `label` | int16 | [N] | class index in the dataset's own label space; `-1` = unknown (test only) |
| `category` | int8 | [N] | coarse category (24 for D1) for hierarchical analysis |
| `session_id` | int32 | [N] | grouping key for leakage-safe splits |
| `ts` | int64 | [N] | flow start, unix ms; needed for time-ordered evaluation |
| `meta.json` | | | dataset, period, label map, columns, exporter git commit, manifest hash |

Size check: 10M flows x (240 + 1 + 172 + 2 + 1 + 4 + 8) bytes ≈ 4.3 GB; fits Kaggle RAM. If needed, `flowstats` is dropped from GPU runs that do not use it.

## Model-facing representation

### Per-packet token (Transformer, spec 006)

Each packet k becomes a sum of embeddings:

- `size_bin`: 64 log-spaced bins over [1, 1500] plus a bin for 0 (padding). Log spacing follows the observation (ECHO, 2024) that non-uniform bins are more efficient than uniform ones.
- `ipt_bin`: 32 log-spaced bins over [0, 32767] ms plus padding bin (32767, not DataZoo's 65535: our IPT shares a signed int16 array with direction, see spec 002).
- `dir`: 3 values {pad, +1, -1}.
- `push`: 3 values {pad, 0, 1} (not 2: a real packet with push=0 must not collide with the padding index, since both are 0 in the raw array).
- `pos`: learned positional embedding for k in 1..30.
- Optional continuous side channel: `[log1p(size)/7.3, log1p(ipt)/11.1, dir]` projected linearly and added (ablation flag `continuous_side=true`).

### Continuous sequence (CNN / RNN baselines, spec 005)

`x[k] = [log1p(size) * dir, log1p(ipt), dir, push]` standardised per channel with training mean/std; padding positions zero and masked.

### Flow statistics (XGBoost baseline, MM baselines)

Standardised with training mean/std after `log1p` on count-like fields (bytes, packets, duration); histograms normalised to proportions. The 43 columns follow the DataZoo `flowstats` set used by mm-CESNET-v2 so that the public baseline can be reused unchanged.

### Prefix construction

For any K, `prefix(x, K)` zeroes positions > K and sets the mask accordingly. `ppi_len < K` keeps `ppi_len`. During training, prefixes are generated on the fly (spec 008), not stored.

### Padding mask

`mask[k] = k <= min(K, ppi_len)`. The Transformer uses it as key-padding mask; CNN/RNN use it for masked pooling.

## Excluded features (shortcut prevention)

Never enter model input: source/destination IP, ports, SNI, JA3, ASN, destination prefix, absolute timestamps. They remain in side tables for auditing and grouping only. A test asserts the model input dimension equals the documented feature count.

## Normalisation discipline

- Statistics (`mean`, `std`, bin edges) are computed on the training periods only and stored as `stats.json` with a hash; validation, test and drift periods reuse them unchanged.
- Bin edges are fixed constants (not data-dependent) so that PCAP-derived flows and DataZoo flows tokenise identically.

## Augmentations (used by 007/008, defined here so they are shared)

| Name | Effect | Default parameters |
|---|---|---|
| `drop` | remove each packet with prob p, shift the rest left, recompute IPT as the sum of removed gaps | p in [0, 0.2] |
| `reorder` | swap adjacent packets with prob q | q = 0.05 |
| `ipt_jitter` | multiply IPT by exp(N(0, s)) | s = 0.2 |
| `size_jitter` | add N(0, 8) bytes, clip to [1, 1500] | |
| `crop` | random prefix K ~ U{2..30} (used for prefix supervision) | |

Augmentations act on raw integer PPI before binning so that they are backend-independent.

## Inputs and outputs

- Inputs: DataZoo dataframes (001) or `FlowRecord`s (002).
- Outputs: shards + `stats.json` under `data/processed/<dataset>/<period>/`; a `Dataset` class returning `(tokens[30,4] int64, cont[30,4] float32, mask[30] bool, flowstats[FLOWSTATS_DIM] float32, label, ppi_len)`.

## Edge cases

- Flows with `ppi_len == 0` (no payload packets): dropped at export and counted; they cannot be classified by definition.
- Sizes > 1500 or IPT > 32767: clipped; counters kept in `meta.json`.
- Labels missing in a period (class absent): allowed; per-class metrics report support.
- Unknown classes (label −1) must never appear in train shards; export asserts this.

## Performance considerations

- Tokenisation is vectorised NumPy (`np.digitize`) on int16 arrays: 10M flows in seconds.
- On Kaggle the whole shard set is loaded into RAM once; batches are sliced views, no DataLoader workers.

## Testing

- Unit tests for bin edges (monotone, cover range), prefix/mask consistency, augmentation invariants (drop keeps order, IPT sums preserved), and shortcut-feature exclusion.
- Round-trip test: DataZoo row → shard → `Dataset` item equals a hand-computed expectation for a fixture flow.
- Cross-backend test: the same fixture pcap through spec-002 backends yields identical tokens.

## Interactions

- Spec 004 uses `session_id` and `ts` for splits. Spec 006 consumes tokens. Spec 016 uses the same `FlowTensorizer` for streaming (one packet at a time).

## Success criteria

- Shards for D1 (weeks 11 to 52, one shard set per week) and D2 (4 weeks) produced with stats; total < 12 GB on disk.
- All tests pass; feature count documented in `meta.json` matches model configs.

## Open questions

- Whether to include the `push` channel for UDP/QUIC (always 0). Default: include for schema uniformity.
- Bin counts (64/32) are initial values; ablate in spec 004 if time permits.
