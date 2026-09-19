# Spec 006: Prefix-Aware Transformer Backbone (PAT)

- **Status:** draft
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
- Optional depth-axis early exit (a second head set after layer 2).

## Non-goals

- Sequence lengths beyond 30 packets; relative-position schemes; MoE; Mamba.

## Architecture

```
tokens[k] = E_size(size_bin_k) + E_ipt(ipt_bin_k) + E_dir(dir_k) + E_push(push_k) + E_pos(k)
            (+ W_side · [log1p(size)/7.3, log1p(ipt)/11.1, dir]   if continuous_side)
h^0 = LayerNorm(Dropout(tokens))
h^l = TransformerBlock_l(h^{l-1}, causal_mask & key_padding_mask)    l = 1..L
z_k = h^L_k                                    # prefix representation after k packets
```

- `d_model = 128`, `L = 4`, heads = 4, FFN = 256, pre-LayerNorm, GELU, dropout 0.1. Parameter count about 0.9M (embeddings about 0.02M).
- A learned `[BOS]` token at position 0 so that K=0 has a representation (used for the prior and for the controller's initial state); packet k sits at position k.
- Causal mask: position k attends to 0..k. Key-padding mask from `ppi_len`.

### Heads (all applied to every position k)

| Head | Output | Used by |
|---|---|---|
| `cls` | logits over C known classes | classification, energy score |
| `safe` | scalar logit: P(argmax cls_k == y) | stopping policy (009) |
| `next` | three small softmaxes: size bin (65), ipt bin (33), dir (2) of packet k+1 | SSL (007); its entropy feeds `safe` |
| `proj` | 128-d L2-normalised projection of z_k | contrastive SSL (007), Mahalanobis unknown score (010) |

`safe` receives `[z_k, entropy(cls_k), margin(cls_k), entropy(next_k), k/30]` so that it can learn the value of waiting from both the classifier's and the predictor's uncertainty.

### Depth exits (optional, flag `depth_exits: [2]`)

A second `cls`/`safe` head pair after block 2. At inference, if the shallow `safe` at step k exceeds a depth threshold, blocks 3 to 4 are skipped for that step. Cost per step becomes `layers_evaluated ∈ {2, 4}`. The KV-cache for blocks 3 to 4 is then stale for that position; to keep it exact we recompute blocks 3 to 4 for skipped positions only when a later step needs them (lazy fill). This is the one piece of real complexity in the depth-exit path and is why it is optional.

### Streaming inference

`PATStream` keeps per-flow KV caches (L x [k, d_model] keys and values). `step(packet) -> Outputs_k` runs the new token through all blocks attending to cached keys; cost O(k · d_model · L). For k <= 30 this is tiny; batching across flows is done by the service (spec 016) with padding to the max k in the batch.

Exactness test: `stream(x)[k] == full_forward(x)[k]` for all k, within fp32 tolerance.

## Inputs and outputs

- Inputs: `tokens [B,30,4] int64`, optional `cont [B,30,3]`, `mask [B,30]`, `ppi_len`.
- Outputs per position: `cls [B,30,C]`, `safe [B,30]`, `next {size:[B,30,65], ipt:[B,30,33], dir:[B,30,2]}`, `proj [B,30,128]`, plus shallow-exit copies if enabled.

## Cost model (used by the controller, spec 011)

`cost(flow) = Σ_k layers_evaluated_k · c_layer(k)` with `c_layer(k) ≈ a + b·k` measured once in spec 013 (attention over k cached keys plus constant FFN work). The default budget metric is simply E[K]; the compute metric uses this cost model.

## Edge cases

- Flows with `ppi_len < k`: positions beyond `ppi_len` are masked; outputs there are ignored (mask propagates to losses and metrics).
- Padding token bins (0) must never receive gradient through `next` targets: targets at padded positions are ignored (`ignore_index`).
- fp16 overflow in attention logits: use PyTorch SDPA with fp32 softmax (`torch.nn.functional.scaled_dot_product_attention`) under autocast.

## Performance considerations

- Training: [4096, 31, 128] activations per block, trivial for a T4; throughput is bounded by the embedding gathers and the heads, about 40k flows/s.
- ONNX export of the full-forward graph (fixed 31 positions) for CPU evaluation; the streaming path is exported separately with explicit cache inputs, or run in TorchScript.

## Testing

- Causality test: perturbing packet j > k does not change any output at k.
- Stream/full equivalence test.
- Parameter budget test (< 2M).
- Depth-exit test: outputs with depth threshold = +inf equal the no-exit model.

## Interactions

- Spec 007 pretrains `next` and `proj`; spec 008 trains `cls` and `safe`; spec 010 uses `proj` and `cls`; spec 011 uses the cost model; spec 016 uses `PATStream`.

## Success criteria

- Fixed-K accuracy of B5 (this backbone, plain training) within 1 point of the GRU baseline at K=30 and better at K <= 8 after prefix supervision (spec 008).
- Streaming latency target met (spec 013).

## Open questions

- Continuous side channel on or off by default (ablation in phase 3).
- Whether to share `cls` weights across the shallow and deep exits (default: separate).
