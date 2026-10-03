# Spec 006: Prefix-Aware Transformer Backbone (PAT)

- **Status:** draft, revised 2026-10-03 before build (see [Revisions](#revisions)). Not implemented. Plan: [phase 3](../plans/phase-3-pat-ssl-prefix-training.md), task T4.
- **Owner:** Parth Challawar
- **Created:** 2026-09-17
- **Build step:** step 8 of 18
- **Depends on:** 003. **Used by:** 007, 008, 009, 010, 011, 013, 016.

## Problem

Adaptive inference over packets requires a model whose output after k packets depends only on those k packets, that is cheap to evaluate incrementally as packets arrive, and that can be trained on all prefixes at once. A bidirectional encoder or a pooled CNN gives a different representation for every K and needs one pass per K (30x cost). A recurrent model is incremental but weak at long-range structure and awkward to pretrain with modern objectives. A **causal Transformer** gives both: one pass yields all 30 prefix outputs in training, and a KV-cache gives O(k) streaming inference.

## Goals

- Causal Transformer encoder over packet tokens with per-position outputs.
- Multiple heads on each position: class logits, commit-safety, next-packet prediction (SSL), and an embedding for unknown scoring.
- Size under 2M parameters; CPU streaming latency under 1 ms per packet step (spec 013 verifies).

## Non-goals

- Sequence lengths beyond 30 packets; relative-position schemes; MoE; Mamba.
- Depth-axis early exits. Deferred to spec 011's phase, the only consumer (see Revisions).

## Architecture

```
tokens[k] = E_size(size_bin_k) + E_ipt(ipt_bin_k) + E_dir(dir_k) + E_push(push_k) + E_pos(k)
            + W_side · cont_k                         # side channel, ON by default
h^0 = LayerNorm(Dropout(tokens))
h^l = TransformerBlock_l(h^{l-1}, causal_mask & key_padding_mask)    l = 1..L
z_k = LayerNorm_final(h^L_k)                     # prefix representation after k packets
```

- **Side channel, on by default.** `cont_k` is packet k's row of the existing standardised continuous view (`features.continuous` then `Standardizer.transform_continuous`, padded rows re-zeroed): `[log1p(size)·dir, log1p(ipt), dir, push]`, 4 values. It carries the **exact** packet size, which the size tokens do not. Measured on the D1 probe day (487,081 flows): 1,460 distinct exact sizes fall into only 53 size bins, and the bin around 1,400 bytes is 162 bytes wide. The CNN and GRU baselines read exact sizes, so a token-only PAT would start with less information than the models it is compared against. Switching the channel off is an ablation, not the default.
- `d_model = 128`, `L = 4`, heads = 4, **FFN = 512** (the usual 4x ratio), pre-LayerNorm, GELU, dropout 0.1. **A final LayerNorm** after the last block, which pre-LN Transformers need. Parameter count about 0.9M (about 0.6M with FFN 256); the parameter-budget test checks the real number. FFN ∈ {256, 512} and dropout ∈ {0, 0.1} are in the equal-budget search (plan phase 3, T9).
- A learned `[BOS]` token at position 0 so that K=0 has a representation (used for the prior and for the controller's initial state); packet k sits at position k. 31 positions in total; the saved per-K logits drop position 0 (`DenseLogits` with `effective` indexing, positions K=1..30).
- Causal mask: position k attends to 0..k. Key-padding mask from `ppi_len`.

### Heads (all applied to every position k)

| Head | Output | Used by |
|---|---|---|
| `cls` | logits over C known classes | classification, energy score |
| `safe` | scalar logit: P(argmax cls_k == y) | stopping policy (009) |
| `next` | three softmaxes for packet k+1: size token (`SIZE_VOCAB` = 65), ipt token (`IPT_VOCAB` = 33), dir token (`DIR_VOCAB` = 3); plus `end`, one logit: "packet k is the last one" | SSL (007); its entropy and `p_end` feed `safe` |
| `proj` | 128-d L2-normalised projection of z_k | contrastive SSL (007), Mahalanobis unknown score (010) |

- **`next` widths come from `features.py`'s vocabulary constants**, never literals. Targets are the next packet's token ids, with `ignore_index = PAD_INDEX` (0), so no padded position is ever a target.
- **`end`** is a binary logit per position. Target 1 at k = `ppi_len` when `ppi_len < 30` (the flow had no more payload packets), 0 at k < `ppi_len`, and ignored at k = `ppi_len` = 30 (truncated at the cap, so whether the flow continued is unknown). Measured on the D1 probe day, 83% of flows end before packet 30, so the target is real. It tells the stopping policy when waiting cannot bring more evidence.
- `safe` receives `[z_k, entropy(cls_k), margin(cls_k), entropy(next_k), p_end_k, k/30]` so that it can learn the value of waiting from the classifier's uncertainty, the predictor's uncertainty and the chance that no more packets are coming.

### Streaming inference

`PATStream` keeps per-flow KV caches (L x [k, d_model] keys and values). `step(packet) -> Outputs_k` runs the new token through all blocks attending to cached keys; cost O(k · d_model · L). For k <= 30 this is tiny; batching across flows is done by the service (spec 016) with padding to the max k in the batch.

Exactness test: `stream(x)[k] == full_forward(x)[k]` for all k, within fp32 tolerance.

## Inputs and outputs

- Inputs: `tokens [B,30,4] int64`, `cont [B,30,4]` (standardised continuous view; omitted only when the side channel is off), `mask [B,30]`, `ppi_len`.
- Outputs per position: `cls [B,31,C]`, `safe [B,31]`, `next {size:[B,31,65], ipt:[B,31,33], dir:[B,31,3], end:[B,31]}`, `proj [B,31,128]`. Position 0 is `[BOS]`.

## Cost model (used by the controller, spec 011)

`cost(flow) = Σ_k c_layer(k) · L` with `c_layer(k) ≈ a + b·k` measured once in spec 013 (attention over k cached keys plus constant FFN work). The default budget metric is simply E[K]; the compute metric uses this cost model.

## Edge cases

- Flows with `ppi_len < k`: positions beyond `ppi_len` are masked; outputs there are ignored (mask propagates to losses and metrics).
- Padding token bins (0) must never receive gradient through `next` targets: targets at padded positions are ignored (`ignore_index`).
- Truncated flows (`ppi_len` = 30, 16.7% of D1 probe flows): the `end` target at position 30 is ignored, never set to 1.
- fp16 overflow in attention logits: use PyTorch SDPA with fp32 softmax (`torch.nn.functional.scaled_dot_product_attention`) under autocast.

## Performance considerations

- Training: [4096, 31, 128] activations per block, trivial for a T4. Estimated about 40k flows/s, the same order as B3's measured 43k flows/s; plan phase 3 T4 replaces this with a measurement.
- ONNX export of the full-forward graph (fixed 31 positions) for CPU evaluation; the streaming path is exported separately with explicit cache inputs, or run in TorchScript.

## Testing

- Causality test: perturbing packet j > k (tokens **and** side channel) does not change any output at k.
- Stream/full equivalence test, with the side channel on.
- Parameter budget test (< 2M).
- `next` targets: padded positions and the truncated `end` position are ignored; head widths equal the vocabulary constants.
- `causal: false` (used by spec 007's MPM comparison) does change outputs at k when a later packet changes, so the switch is real.
- Side channel off: the model builds and trains without a `cont` input.

## Interactions

- Spec 007 pretrains `next` and `proj`; spec 008 trains `cls` and `safe`; spec 010 uses `proj` and `cls`; spec 011 uses the cost model; spec 016 uses `PATStream`.

## Success criteria

Compared like with like, because the training recipe matters as much as the architecture. The CNN (B2) trains on full flows only, while the GRU (B3) trains on every prefix, which costs it some full-flow accuracy (real closed-set D1 val macro-F1: CNN 0.8651, GRU 0.8124).

- **B5** (this backbone, full-flow training only, spec 005) within 1 point of the **CNN** at K=30. Both are full-flow-trained, so this isolates the architecture.
- **The prefix-trained PAT** (spec 008) has a higher `AUC_K` than both the CNN and the GRU, and better accuracy than both at every K <= 8.
- Streaming latency target met (spec 013).

## Open questions

- Whether to share `cls` weights across shallow and deep exits, once depth exits are built (spec 011's phase).

## Revisions

**2026-10-03, before build** (plan phase 3, findings F2 and F10-F14; measured on the D1 probe day):

- Side channel **on by default**, using the existing standardised continuous view rather than its own `log1p(x)/7.3` scaling. Reason: size tokens collapse 1,460 exact sizes into 53 bins, and the baselines see exact sizes.
- FFN 256 → **512**, the standard 4x ratio; still under the 2M-parameter cap. 256 stays in the search space.
- **Final LayerNorm** added (pre-LN needs it; the original text omitted it).
- `next` head widths taken from the vocabulary constants: `dir` is 3-way (pad, +1, −1), not 2. **`end`** logit added, and `p_end` added to `safe`'s inputs.
- **Depth exits removed** from this spec's deliverables, deferred to spec 011's phase. Streaming *latency* stays with spec 013; `PATStream` correctness is built here.
- Success criteria rewritten to compare like with like: B5 vs the CNN at K=30, not "within 1 point of the GRU", which would have let the PAT pass while 4 points behind the best baseline.
