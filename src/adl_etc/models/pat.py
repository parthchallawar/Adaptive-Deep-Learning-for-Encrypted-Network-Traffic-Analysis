"""The Prefix-Aware Transformer, PAT (spec 006; plan phase 3, T4).

A small causal Transformer over packet tokens: its output after packet k depends
on packets 1..k only, one forward pass yields all 30 prefix outputs, and
:class:`PATStream` produces the same outputs one packet at a time with a KV
cache.

Input per packet k (1..30): the sum of embeddings of its four tokens
(``features.tokenize``: ipt, dir, size, push; 0 = padding) and a learned
position, plus, with the side channel on (the default, spec 006 / plan F10), a
linear projection of the packet's standardised continuous row
(``[log1p(size)·dir, log1p(ipt), dir, push]``), which carries the exact size the
64 size bins cannot. A learned ``[BOS]`` vector sits at position 0 so that "no
packets yet" has a representation. ``h0 = LayerNorm(Dropout(x))``, then ``L``
pre-LN blocks, then a final LayerNorm.

**Output layout.** Every per-position output is returned for packet positions
1..30 only, as ``[B, 30, ...]`` aligned with ``batch["mask"]``: index ``k - 1``
is "after k packets", the convention every loss, metric and ``DenseLogits``
already uses. The ``[BOS]`` position's class logits and representation come back
separately as ``cls0`` and ``z0`` (the prior, for spec 011's controller).

Heads at every position:

- ``cls`` ``[B, 30, C]``: class logits.
- ``next_size``/``next_ipt``/``next_dir`` ``[B, 30, V]``: the next packet's
  tokens (widths are ``features.SIZE_VOCAB``/``IPT_VOCAB``/``DIR_VOCAB``; index
  0 is padding and is never a target). ``next_end`` ``[B, 30]``: one logit,
  "this packet is the flow's last".
- ``proj`` ``[B, 30, 128]``: L2-normalised projection (contrastive SSL, spec
  010's Mahalanobis score).
- ``safe`` ``[B, 30]``: commit-safety logit, from ``z_k`` and five *detached*
  summary features: normalised entropy and top-2 margin of ``cls``, mean
  normalised entropy of the three ``next`` heads, ``sigmoid(next_end)`` and
  ``k / 30``. Detached so that training ``safe`` never moves the heads it reads.
- ``z`` ``[B, 30, d_model]``: the prefix representation (probes, spec 007).

``causal=False`` drops the causal mask (every position sees every real packet)
for spec 007's masked-packet-modelling comparison; weights are interchangeable
with the causal model. ``layer_id``/``n_layers`` feed layer-wise LR decay.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from adl_etc.data import features as FT
from adl_etc.data import ppi as P

N_POSITIONS = P.K_MAX + 1  # [BOS] + 30 packets
SAFE_FEATURES = 5  # entropy(cls), margin(cls), entropy(next), p_end, k/30


def tokenizer_fingerprint() -> int:
    """A signed 64-bit fingerprint of what a token *means*: the size and ipt bin
    edges, the vocabulary sizes and ``K_MAX``. Stored in every PAT's state as an
    integer buffer, so a checkpoint trained under different tokens is refused
    even when every tensor shape still matches (spec 008's strict load)."""
    import hashlib

    digest = hashlib.sha256()
    digest.update(FT.SIZE_BIN_EDGES.tobytes())
    digest.update(FT.IPT_BIN_EDGES.tobytes())
    sizes = (FT.SIZE_VOCAB, FT.IPT_VOCAB, FT.DIR_VOCAB, FT.PUSH_VOCAB, P.K_MAX)
    digest.update(repr(sizes).encode())
    return int.from_bytes(digest.digest()[:8], "little", signed=True)


class Block(nn.Module):
    """Pre-LN Transformer block: ``h + attn(LN(h))``, then ``h + ffn(LN(h))``."""

    def __init__(self, d_model: int, n_heads: int, ffn: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} is not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.attn_dropout = dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, d_model)
        )
        self.drop = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, d] -> [B, H, T, dh]
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.d_head).transpose(1, 2)

    def _finish(self, h: torch.Tensor, attended: torch.Tensor) -> torch.Tensor:
        attended = attended.transpose(1, 2).reshape(h.shape)
        h = h + self.drop(self.out(attended))
        return h + self.drop(self.ff(self.norm2(h)))

    def forward(self, h: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        """``allowed`` ``[B, 1, T, T]`` bool: True where query i may attend key j."""
        q, k, v = (self._split(t) for t in self.qkv(self.norm1(h)).chunk(3, dim=-1))
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, dropout_p=self.attn_dropout if self.training else 0.0
        )
        return self._finish(h, attended)

    def step(
        self, x: torch.Tensor, cache: dict[str, torch.Tensor] | None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """One new position ``x`` ``[B, 1, d]`` attending to itself and every
        cached earlier position. Every cached key is a real, earlier packet, so no
        mask is needed."""
        q, k, v = (self._split(t) for t in self.qkv(self.norm1(x)).chunk(3, dim=-1))
        if cache is not None:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
        attended = F.scaled_dot_product_attention(q, k, v)
        return self._finish(x, attended), {"k": k, "v": v}


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of ``softmax(logits)`` over the last axis, divided by its maximum
    ``log(V)`` so every head's value is in [0, 1]."""
    logp = logits.float().log_softmax(-1)
    return -(logp.exp() * logp).sum(-1) / math.log(logits.shape[-1])


class PAT(nn.Module):
    causal: bool
    logits_key = "cls"  # what predict_dense reads from the dict of heads

    def __init__(
        self,
        n_classes: int,
        *,
        d_model: int = 128,
        layers: int = 4,
        heads: int = 4,
        ffn: int = 512,
        dropout: float = 0.1,
        side_channel: bool = True,
        causal: bool = True,
        proj_dim: int = 128,
        safe_hidden: int = 64,
    ) -> None:
        super().__init__()
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
        self.n_classes = n_classes
        self.n_layers = layers
        self.causal = causal
        self.side_channel = side_channel
        self.views = ("tokens", "continuous") if side_channel else ("tokens",)

        pad = FT.PAD_INDEX
        self.emb_size = nn.Embedding(FT.SIZE_VOCAB, d_model, padding_idx=pad)
        self.emb_ipt = nn.Embedding(FT.IPT_VOCAB, d_model, padding_idx=pad)
        self.emb_dir = nn.Embedding(FT.DIR_VOCAB, d_model, padding_idx=pad)
        self.emb_push = nn.Embedding(FT.PUSH_VOCAB, d_model, padding_idx=pad)
        self.emb_pos = nn.Embedding(N_POSITIONS, d_model)
        self.bos = nn.Parameter(torch.zeros(d_model))
        self.side = nn.Linear(FT.CONTINUOUS_CHANNELS, d_model) if side_channel else None
        self.emb_drop = nn.Dropout(dropout)
        self.emb_norm = nn.LayerNorm(d_model)

        self.blocks = nn.ModuleList(
            [Block(d_model, heads, ffn, dropout) for _ in range(layers)]
        )
        self.norm_f = nn.LayerNorm(d_model)

        self.head_cls = nn.Linear(d_model, n_classes)
        self.head_next_size = nn.Linear(d_model, FT.SIZE_VOCAB)
        self.head_next_ipt = nn.Linear(d_model, FT.IPT_VOCAB)
        self.head_next_dir = nn.Linear(d_model, FT.DIR_VOCAB)
        self.head_next_end = nn.Linear(d_model, 1)
        self.head_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, proj_dim)
        )
        self.head_safe = nn.Sequential(
            nn.Linear(d_model + SAFE_FEATURES, safe_hidden), nn.GELU(), nn.Linear(safe_hidden, 1)
        )
        self.register_buffer(
            "tokenizer_id", torch.tensor([tokenizer_fingerprint()], dtype=torch.int64)
        )
        self._init_embeddings()

    def _init_embeddings(self) -> None:
        for emb in (self.emb_size, self.emb_ipt, self.emb_dir, self.emb_push, self.emb_pos):
            nn.init.normal_(emb.weight, std=0.02)
            if emb.padding_idx is not None:
                with torch.no_grad():
                    emb.weight[emb.padding_idx].zero_()
        nn.init.normal_(self.bos, std=0.02)

    # --- layer-wise LR decay (training.loop.param_groups) -------------------------------

    def layer_id(self, name: str) -> int:
        """0 for embeddings, ``i + 1`` for block ``i``, ``n_layers + 1`` for the
        final norm and the heads."""
        if name.startswith(("emb_", "bos", "side")):
            return 0
        if name.startswith("blocks."):
            return int(name.split(".")[1]) + 1
        return self.n_layers + 1

    # --- the forward pass ------------------------------------------------------------------

    def _embed_packets(self, tokens: torch.Tensor, cont: torch.Tensor | None) -> torch.Tensor:
        x = (
            self.emb_ipt(tokens[..., P.IPT_POS])
            + self.emb_dir(tokens[..., P.DIR_POS])
            + self.emb_size(tokens[..., P.SIZE_POS])
            + self.emb_push(tokens[..., P.PUSH_POS])
        )
        if self.side is not None:
            if cont is None:
                raise ValueError("the side channel is on, so the batch needs its 'cont' view")
            x = x + self.side(cont.to(x.dtype))
        return x

    def _outputs(self, z: torch.Tensor, positions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Every head at every position of ``z`` ``[B, T, d]``; ``positions`` [T]."""
        cls = self.head_cls(z)
        nxt_size = self.head_next_size(z)
        nxt_ipt = self.head_next_ipt(z)
        nxt_dir = self.head_next_dir(z)
        nxt_end = self.head_next_end(z).squeeze(-1)
        with torch.no_grad():
            probs = cls.detach().float().softmax(-1)
            top2 = probs.topk(2, dim=-1).values
            features = torch.stack(
                [
                    _entropy(cls.detach()),
                    top2[..., 0] - top2[..., 1],
                    (
                        _entropy(nxt_size.detach())
                        + _entropy(nxt_ipt.detach())
                        + _entropy(nxt_dir.detach())
                    )
                    / 3,
                    torch.sigmoid(nxt_end.detach().float()),
                    (positions.float() / P.K_MAX).expand(z.shape[0], -1),
                ],
                dim=-1,
            )
        safe_in = torch.cat([z, features.to(z.dtype)], dim=-1)
        return {
            "cls": cls,
            "safe": self.head_safe(safe_in).squeeze(-1),
            "next_size": nxt_size,
            "next_ipt": nxt_ipt,
            "next_dir": nxt_dir,
            "next_end": nxt_end,
            "proj": F.normalize(self.head_proj(z).float(), dim=-1),
            "z": z,
        }

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        tokens = batch["tokens"]
        b = tokens.shape[0]
        x = self._embed_packets(tokens, batch.get("cont"))
        h = torch.cat([self.bos.to(x.dtype).expand(b, 1, -1), x], dim=1) + self.emb_pos.weight
        h = self.emb_norm(self.emb_drop(h))

        valid = torch.cat(
            [torch.ones(b, 1, dtype=torch.bool, device=tokens.device), batch["mask"].bool()], dim=1
        )
        allowed = valid[:, None, None, :].expand(b, 1, N_POSITIONS, N_POSITIONS)
        if self.causal:
            tril = torch.ones(N_POSITIONS, N_POSITIONS, dtype=torch.bool, device=tokens.device)
            allowed = allowed & tril.tril()
        for block in self.blocks:
            h = block(h, allowed)
        z = self.norm_f(h)

        out = self._outputs(z, torch.arange(N_POSITIONS, device=z.device))
        per_packet = {name: t[:, 1:] for name, t in out.items()}
        per_packet["cls0"] = out["cls"][:, 0]
        per_packet["z0"] = out["z"][:, 0]
        return per_packet


class PATStream:
    """Packet-by-packet inference with a KV cache (spec 006).

    ``start(batch_size)`` runs the ``[BOS]`` position and returns its outputs;
    each ``step(tokens_k, cont_k)`` then consumes packet k for every flow in the
    batch (``tokens_k`` ``[B, 4]``, ``cont_k`` ``[B, 4]`` standardised, or
    ``None`` without the side channel) and returns that position's outputs, each
    ``[B, ...]``. Equal to ``model(batch)`` at every position within fp32
    tolerance (tested). All flows advance together; serving flows at different
    k is batching work for spec 016.
    """

    def __init__(self, model: PAT) -> None:
        if not model.causal:
            raise ValueError("streaming needs a causal model")
        if model.training:
            raise ValueError("call model.eval() first: streaming is inference only")
        self.model = model
        self.position = -1
        self.caches: list[dict[str, torch.Tensor] | None] = []

    def _advance(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        m = self.model
        self.position += 1
        if self.position >= N_POSITIONS:
            raise ValueError(f"a flow has at most {P.K_MAX} packets")
        h = m.emb_norm(m.emb_drop(x + m.emb_pos.weight[self.position]))[:, None]
        for i, block in enumerate(m.blocks):
            assert isinstance(block, Block)
            h, self.caches[i] = block.step(h, self.caches[i])
        z = m.norm_f(h)
        out = m._outputs(z, torch.tensor([self.position], device=z.device))
        return {name: t[:, 0] for name, t in out.items()}

    @torch.no_grad()
    def start(self, batch_size: int, device: Any = "cpu") -> dict[str, torch.Tensor]:
        self.position = -1
        self.caches = [None] * self.model.n_layers
        bos = self.model.bos.to(device).expand(batch_size, -1)
        return self._advance(bos)

    @torch.no_grad()
    def step(
        self, tokens: torch.Tensor, cont: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if self.position < 0:
            raise ValueError("call start() before step()")
        x = self.model._embed_packets(tokens[:, None], None if cont is None else cont[:, None])
        return self._advance(x[:, 0])
