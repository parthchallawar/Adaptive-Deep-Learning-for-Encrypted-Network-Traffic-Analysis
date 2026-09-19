"""Tokeniser, continuous view, prefixes and augmentations (spec 003).

The contract between the data and every model. A CNN/RNN baseline reads
:func:`continuous`; the Prefix-Aware Transformer reads :func:`tokenize`; both
read the same raw ``ppi`` arrays from :mod:`adl_etc.data.tensors`, so a bug
here is a bug every model shares identically (which is the point: results stay
comparable) or a bug that corrupts every model identically (which is the risk,
hence the fixture-based tests).

**Padding convention, enforced in every channel:** index/token ``0`` always
means "no packet here". Real values start at ``1``. This used to not hold for
the ``push`` channel (2 raw values, colliding with the padding index); see the
spec-003 correction recorded in ``plans/phase-1-data-pipeline.md``.

**Bin edges are frozen literals, not computed at import time.** They were
generated once by :func:`_generate_size_edges`/:func:`_generate_ipt_edges` and
pasted in below. ``test_features.py`` asserts the literals still equal the
generators' output, so a NumPy version change that alters ``geomspace``'s
rounding is a loud test failure instead of every saved token silently moving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from adl_etc.data import ppi as P
from adl_etc.data.tensors import ShardSet

# --- token channel layout ---------------------------------------------------

PAD_INDEX = 0
"""The one rule every channel below obeys: index 0 is padding."""

SIZE_BINS = 64
IPT_BINS = 32

SIZE_BIN_EDGES = np.array(
    [
        1.121053724605064, 1.2567614534508866, 1.40889710833119, 1.579449350879985,
        1.7706475776290578, 1.9849910618639892, 2.2252816232103863, 2.4946602519952057,
        2.7966481671234327, 3.1351928441636496, 3.5147196153048035, 3.9401895156799274,
        4.417164132202806, 4.951878302597851, 5.551321614918322, 6.223329772884783,
        6.976687021338074, 7.8212409706748565, 8.768031321208774, 9.829434170094956,
        11.019323787145236, 12.353253974208346, 13.848661378778575, 15.525093419474022,
        17.40446380274292, 19.511338970818965, 21.873259225268537, 24.521098723739367,
        27.4894690556565, 30.81717167225937, 34.54770508498003, 38.729833462074176,
        43.41822405599209, 48.67416179370712, 54.56635037086487, 61.171810321362976,
        68.57688580159845, 76.87837324969807, 86.18478667315235, 96.61777610423032,
        108.31371776470554, 121.42549672594483, 136.12450536664045, 152.60288375129429,
        171.07603121486204, 191.78542198407328, 215.00176164019928, 241.02852568339557,
        270.20592645343794, 302.91536026098856, 339.584392860666, 380.6923484341989,
        426.77657514080744, 478.43946913579515, 536.3563488727527, 601.2842826193726,
        674.0719845769315, 755.6709089618962, 847.147687067428, 949.6980698775053,
        1064.6625584864175, 1193.5439266387552, 1338.0268644381297,
    ],
    dtype=np.float64,
)
"""63 strictly-interior edges -> 64 bins (token indices 1..64), log-spaced over
(0, SIZE_MAX). Deliberately excludes both endpoints: an edge placed exactly at
SIZE_MAX would make the top bin unreachable, since ``clip_size`` never
produces a value greater than SIZE_MAX for ``side="left"`` digitization to
fall into (see ``ppi.SIZE_HIST_EDGES`` for the same convention)."""

IPT_BIN_EDGES = np.array(
    [
        1.3839085621450118, 1.9152029083782738, 2.650465703149721, 3.668002180260598,
        5.076179623229213, 7.024968443572947, 9.721913977859115, 13.454239994396502,
        18.619437925399176, 25.767599567287476, 35.66000166709322, 49.35018163319571,
        68.29613890559105, 94.51561139289247, 130.80096386299456, 181.0165738268184,
        250.51038640908848, 346.6834686577931, 479.77822062965157, 663.9691874600737,
        918.872643526462, 1271.6357188970915, 1759.8275593111132, 2435.440427229407,
        3370.4268598368803, 4664.362589411787, 6455.051324435847, 8933.200796972264,
        12362.733070290553, 17108.892147488412, 23677.14233172476,
    ],
    dtype=np.float64,
)
"""31 strictly-interior edges -> 32 bins (token indices 1..32), log-spaced over
(0, IPT_MAX_MS). Same top-bin-reachability reasoning as SIZE_BIN_EDGES."""

SIZE_VOCAB = SIZE_BINS + 1  # + padding
IPT_VOCAB = IPT_BINS + 1
DIR_VOCAB = 3  # pad, +1, -1
PUSH_VOCAB = 3  # pad, 0, 1

CONTINUOUS_CHANNELS = 4
"""``[log1p(size) * dir, log1p(ipt), dir, push]``."""


def _generate_size_edges() -> np.ndarray:
    """Regenerates :data:`SIZE_BIN_EDGES`. Used only by the test that checks
    the frozen literal still matches; never called at import time.

    Generates ``SIZE_BINS + 1`` log-spaced points spanning ``[1, SIZE_MAX]``
    and drops both endpoints, leaving ``SIZE_BINS - 1`` strictly-interior
    edges. Including ``SIZE_MAX`` itself as an edge would make the top bin
    unreachable (see the module docstring on ``SIZE_BIN_EDGES``).
    """
    return np.geomspace(1, P.SIZE_MAX, SIZE_BINS + 1)[1:-1]


def _generate_ipt_edges() -> np.ndarray:
    """Regenerates :data:`IPT_BIN_EDGES`. Test-only, see :func:`_generate_size_edges`."""
    return np.geomspace(1, P.IPT_MAX_MS, IPT_BINS + 1)[1:-1]


# --- tokenisation ------------------------------------------------------------
#
# Both PPI columns being binned are bounded integers (size in [0, SIZE_MAX],
# ipt in [0, IPT_MAX_MS]), so the bin lookup is precomputed once as a table
# indexed by the raw value, rather than binary-searched per element with
# np.searchsorted. On 1M flows (30M packets) this is the difference between
# ~8s and ~0.3s for tokenize() -- searchsorted's per-element binary search
# does not vectorise as well as a gather does, and spec 003 requires 10M
# flows to tokenise "in seconds".

_SIZE_LUT = (np.searchsorted(SIZE_BIN_EDGES, np.arange(P.SIZE_MAX + 1), side="left") + 1).astype(
    np.int64
)
_IPT_LUT = (np.searchsorted(IPT_BIN_EDGES, np.arange(P.IPT_MAX_MS + 1), side="left") + 1).astype(
    np.int64
)


def _valid_mask(ppi_len: np.ndarray, k_max: int = P.K_MAX) -> np.ndarray:
    """[N, k_max] bool, True where position j holds a real (non-padding) packet."""
    positions = np.arange(k_max)
    return positions[None, :] < np.asarray(ppi_len)[:, None]


def tokenize(ppi: np.ndarray, ppi_len: np.ndarray) -> np.ndarray:
    """Raw int16 PPI ``[N, 30, 4]`` -> token indices ``[N, 30, 4]`` int64.

    Channel order matches :data:`adl_etc.data.ppi.PPI_COLUMNS`
    (``ipt, dir, size, push``). Padding positions (``j >= ppi_len[i]``) are 0
    in every channel, regardless of what the raw array holds there.
    """
    ppi = np.asarray(ppi)
    mask = _valid_mask(ppi_len, ppi.shape[1])

    # Defensively re-clip before indexing the lookup tables: every writer in
    # this codebase already clips via ppi.clip_ipt/clip_size, but an
    # out-of-range value here would otherwise be a silent wraparound (negative
    # index) or a hard crash (too-large index) instead of a clipped value.
    ipt = np.clip(ppi[..., P.IPT_POS], 0, P.IPT_MAX_MS).astype(np.int64)
    size = np.clip(ppi[..., P.SIZE_POS], 0, P.SIZE_MAX).astype(np.int64)
    direction = ppi[..., P.DIR_POS]
    push = ppi[..., P.PUSH_POS]

    ipt_bin = _IPT_LUT[ipt]
    size_bin = _SIZE_LUT[size]
    dir_tok = np.where(direction == P.DIR_REV, 2, direction)  # DIR_FWD(1)->1, DIR_REV(-1)->2
    push_tok = push + 1  # 0->1 ("no push"), 1->2 ("push"); pad forced to 0 below

    tokens = np.stack([ipt_bin, dir_tok, size_bin, push_tok], axis=-1).astype(np.int64)
    tokens *= mask[..., None]
    return tokens


def continuous(ppi: np.ndarray, ppi_len: np.ndarray) -> np.ndarray:
    """Raw int16 PPI ``[N, 30, 4]`` -> continuous view ``[N, 30, 4]`` float32:
    ``[log1p(size) * dir, log1p(ipt), dir, push]``. Padding rows are all zero.
    """
    ppi = np.asarray(ppi)
    mask = _valid_mask(ppi_len, ppi.shape[1])

    ipt = ppi[..., P.IPT_POS].astype(np.float64)
    size = ppi[..., P.SIZE_POS].astype(np.float64)
    direction = ppi[..., P.DIR_POS].astype(np.float64)
    push = ppi[..., P.PUSH_POS].astype(np.float64)

    out = np.stack(
        [np.log1p(size) * direction, np.log1p(ipt), direction, push], axis=-1
    ).astype(np.float32)
    out *= mask[..., None]
    return out


def padding_mask(ppi_len: np.ndarray, k: int = P.K_MAX) -> np.ndarray:
    """``[N, K_MAX]`` bool. ``mask[i, j] = j < min(k, ppi_len[i])``.

    Always ``K_MAX`` wide, regardless of ``k``: ``k`` is a *cutoff* on how many
    leading positions can be true, not the width of the returned array. This
    keeps the mask's shape aligned with ``ppi``/``tokens``/``cont``, which stay
    ``K_MAX``-wide with trailing positions zeroed rather than truncated (see
    :func:`prefix`).
    """
    ppi_len = np.asarray(ppi_len)
    effective = np.minimum(ppi_len, k)
    positions = np.arange(P.K_MAX)
    return positions[None, :] < effective[:, None]


def prefix(arrays: dict[str, np.ndarray], k: int) -> dict[str, np.ndarray]:
    """Truncate a batch of flow arrays to the first ``k`` packets.

    Zeroes PPI/token/continuous positions beyond ``k`` and returns a mask
    consistent with :func:`padding_mask`. ``ppi_len`` is left unchanged (it
    records the flow's true length, not the prefix length); the effective
    length at this prefix is ``min(ppi_len, k)``, which ``padding_mask``
    already accounts for.

    Recognises whichever of ``ppi``, ``tokens``, ``cont`` are present in
    ``arrays``; any other keys (``label``, ``flowstats``, ...) pass through
    unchanged.
    """
    if not (1 <= k <= P.K_MAX):
        raise ValueError(f"k must be in 1..{P.K_MAX}, got {k}")
    out = dict(arrays)
    ppi_len = arrays.get("ppi_len")
    for key in ("ppi", "tokens", "cont"):
        if key not in arrays:
            continue
        arr = arrays[key].copy()
        arr[:, k:] = 0
        out[key] = arr
    if ppi_len is not None:
        out["mask"] = padding_mask(ppi_len, k=k)
    return out


# --- flow-statistics standardisation -----------------------------------------

_LOG1P_PREFIXES = ("BYTES", "PACKETS", "DURATION", "PPI_LEN", "PPI_DURATION", "PPI_ROUNDTRIPS")
_HIST_PREFIXES = ("PSIZE_HIST", "IPT_HIST")


def _flowstats_groups() -> dict[str, list[int]]:
    """Column indices of :data:`adl_etc.data.ppi.FLOWSTATS_COLUMNS`, grouped by
    how they are treated: ``log1p`` counts, histogram proportions, or raw
    0/1 flags. Derived from the column names so the grouping cannot drift out
    of sync with a change to ``FLOWSTATS_COLUMNS``.
    """
    groups: dict[str, list[int]] = {"log1p": [], "hist": [], "flag": []}
    for i, name in enumerate(P.FLOWSTATS_COLUMNS):
        if name.startswith(_HIST_PREFIXES):
            groups["hist"].append(i)
        elif name.startswith(_LOG1P_PREFIXES):
            groups["log1p"].append(i)
        else:
            groups["flag"].append(i)
    return groups


def _normalize_histograms(flowstats: np.ndarray, groups: dict[str, list[int]]) -> np.ndarray:
    """Renormalise each 8-bin histogram block to proportions (sum <= 1), one
    block at a time so that e.g. PSIZE_HIST and PSIZE_HIST_REV are normalised
    independently."""
    out = flowstats.copy()
    hist_idx = np.array(groups["hist"])
    if len(hist_idx) == 0:
        return out
    for start in range(0, len(hist_idx), P.PHIST_BIN_COUNT):
        block = hist_idx[start : start + P.PHIST_BIN_COUNT]
        total = out[:, block].sum(axis=1, keepdims=True)
        safe_total = np.where(total > 0, total, 1.0)
        out[:, block] = out[:, block] / safe_total
    return out


@dataclass
class Standardizer:
    """Training-only normalisation statistics for the continuous view and the
    flow-statistics vector. Fit once on the training periods; every other
    split reuses the saved stats unchanged (spec 003's normalisation
    discipline).
    """

    cont_mean: np.ndarray  # [CONTINUOUS_CHANNELS]
    cont_std: np.ndarray
    flowstats_mean: np.ndarray  # [FLOWSTATS_DIM]
    flowstats_std: np.ndarray
    hash: str = ""

    @classmethod
    def fit(cls, shards: ShardSet) -> Standardizer:
        ppi = shards.column("ppi")
        ppi_len = shards.column("ppi_len")
        cont = continuous(ppi, ppi_len)
        mask = _valid_mask(ppi_len, cont.shape[1])
        flat = cont[mask]  # [n_valid_packets, CONTINUOUS_CHANNELS]
        cont_mean = flat.mean(axis=0).astype(np.float64)
        cont_std = flat.std(axis=0).astype(np.float64)
        cont_std = np.where(cont_std < 1e-8, 1.0, cont_std)

        groups = _flowstats_groups()
        fs = shards.column("flowstats").astype(np.float64).copy()
        log1p_idx = np.array(groups["log1p"])
        if len(log1p_idx):
            fs[:, log1p_idx] = np.log1p(np.clip(fs[:, log1p_idx], a_min=0, a_max=None))
        fs = _normalize_histograms(fs, groups)
        fs_mean = fs.mean(axis=0)
        fs_std = fs.std(axis=0)
        fs_std = np.where(fs_std < 1e-8, 1.0, fs_std)

        obj = cls(
            cont_mean=cont_mean,
            cont_std=cont_std,
            flowstats_mean=fs_mean,
            flowstats_std=fs_std,
        )
        obj.hash = obj._compute_hash()
        return obj

    def _compute_hash(self) -> str:
        from adl_etc.utils.provenance import stable_hash

        payload = {
            "cont_mean": self.cont_mean.tolist(),
            "cont_std": self.cont_std.tolist(),
            "flowstats_mean": self.flowstats_mean.tolist(),
            "flowstats_std": self.flowstats_std.tolist(),
        }
        return stable_hash(payload)

    def transform_continuous(self, cont: np.ndarray) -> np.ndarray:
        return ((cont - self.cont_mean) / self.cont_std).astype(np.float32)

    def transform_flowstats(self, flowstats: np.ndarray) -> np.ndarray:
        groups = _flowstats_groups()
        fs = flowstats.astype(np.float64).copy()
        log1p_idx = np.array(groups["log1p"])
        if len(log1p_idx):
            fs[:, log1p_idx] = np.log1p(np.clip(fs[:, log1p_idx], a_min=0, a_max=None))
        fs = _normalize_histograms(fs, groups)
        return ((fs - self.flowstats_mean) / self.flowstats_std).astype(np.float32)

    def save(self, path: str | Path) -> None:
        import json

        path = Path(path)
        payload = {
            "cont_mean": self.cont_mean.tolist(),
            "cont_std": self.cont_std.tolist(),
            "flowstats_mean": self.flowstats_mean.tolist(),
            "flowstats_std": self.flowstats_std.tolist(),
            "hash": self.hash,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Standardizer:
        import json

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            cont_mean=np.array(payload["cont_mean"], dtype=np.float64),
            cont_std=np.array(payload["cont_std"], dtype=np.float64),
            flowstats_mean=np.array(payload["flowstats_mean"], dtype=np.float64),
            flowstats_std=np.array(payload["flowstats_std"], dtype=np.float64),
            hash=payload["hash"],
        )


# --- augmentations ------------------------------------------------------------
# Act on raw integer PPI (single flow, shape [K_MAX, PPI_CHANNELS] plus its
# ppi_len), so they are backend-independent and can run before tokenisation.


def drop(
    ppi: np.ndarray, ppi_len: int, p: float, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """Remove each of the first ``ppi_len`` packets independently with
    probability ``p``. Surviving packets shift left; a removed packet's gap is
    folded into the next surviving packet's IPT so total elapsed time is
    preserved. The new first packet always has IPT 0, matching how a real
    capture that started mid-flow would look.
    """
    if ppi_len == 0:
        return ppi.copy(), 0
    keep = rng.random(ppi_len) >= p
    if not keep.any():
        keep[rng.integers(0, ppi_len)] = True  # never drop every packet

    used = ppi[:ppi_len].copy()
    kept_rows: list[np.ndarray] = []
    carried_ipt = 0
    for i in range(ppi_len):
        if keep[i]:
            row = used[i].copy()
            row[P.IPT_POS] = P.clip_ipt(int(row[P.IPT_POS]) + carried_ipt) if kept_rows else 0
            kept_rows.append(row)
            carried_ipt = 0
        else:
            carried_ipt += int(used[i, P.IPT_POS])

    new_ppi = P.empty_ppi()
    new_len = len(kept_rows)
    if new_len:
        new_ppi[:new_len] = np.stack(kept_rows)
    return new_ppi, new_len


def reorder(ppi: np.ndarray, ppi_len: int, q: float, rng: np.random.Generator) -> np.ndarray:
    """Swap each adjacent pair of the first ``ppi_len`` packets independently
    with probability ``q``. Preserves the IPT/size/dir/push of every packet
    (only their order changes), so the multiset of packets is exactly
    unchanged; ``ipt[0]`` may no longer be 0 after a swap, which is corrected
    back to 0 since position 0 is always "no prior packet" by definition.
    """
    out = ppi.copy()
    if ppi_len < 2:
        return out
    i = 0
    while i < ppi_len - 1:
        if rng.random() < q:
            out[i], out[i + 1] = out[i + 1].copy(), out[i].copy()
            i += 2
        else:
            i += 1
    out[0, P.IPT_POS] = 0
    return out


def ipt_jitter(ppi: np.ndarray, ppi_len: int, s: float, rng: np.random.Generator) -> np.ndarray:
    """Multiply each of the first ``ppi_len`` inter-packet times by
    ``exp(N(0, s))`` and re-clip. Position 0's IPT stays 0."""
    out = ppi.copy()
    if ppi_len == 0:
        return out
    factors = np.exp(rng.normal(0, s, size=ppi_len))
    new_ipt = out[:ppi_len, P.IPT_POS].astype(np.float64) * factors
    new_ipt[0] = 0
    out[:ppi_len, P.IPT_POS] = [P.clip_ipt(v) for v in new_ipt]
    return out


def size_jitter(ppi: np.ndarray, ppi_len: int, sd: float, rng: np.random.Generator) -> np.ndarray:
    """Add ``N(0, sd)`` bytes to each of the first ``ppi_len`` sizes, clipped
    to ``[1, SIZE_MAX]`` so a jittered packet never looks like padding."""
    out = ppi.copy()
    if ppi_len == 0:
        return out
    noise = rng.normal(0, sd, size=ppi_len)
    new_size = out[:ppi_len, P.SIZE_POS].astype(np.float64) + noise
    out[:ppi_len, P.SIZE_POS] = np.clip(np.round(new_size), 1, P.SIZE_MAX).astype(P.PPI_DTYPE)
    return out


def crop(ppi_len: int, rng: np.random.Generator) -> int:
    """Draw a random prefix length ``K ~ U{2..30}``, capped at ``ppi_len`` if
    the flow is shorter."""
    k = int(rng.integers(2, P.K_MAX + 1))
    return min(k, max(ppi_len, 1))


# --- streaming tensoriser ------------------------------------------------------


@dataclass
class StreamTensorizer:
    """Builds up tokens/continuous/mask one packet at a time, for the live
    inference path (spec 016). Its output at prefix ``k`` must equal
    :func:`tokenize`/:func:`continuous` applied to the offline array truncated
    at ``k`` (tested as the stream/offline equivalence invariant, spec 020).
    """

    _ppi: np.ndarray = field(default_factory=P.empty_ppi)
    _len: int = 0

    def reset(self) -> None:
        self._ppi = P.empty_ppi()
        self._len = 0

    def push(
        self, ipt_ms: int, direction: int, size: int, push_flag: int
    ) -> dict[str, np.ndarray | int]:
        """Append one payload-carrying packet (caller filters zero-payload
        packets before calling, matching spec 002). No-op if already at
        ``K_MAX``. Returns the current-prefix tokens/continuous/mask."""
        if self._len < P.K_MAX:
            self._ppi[self._len] = (
                P.clip_ipt(ipt_ms),
                direction,
                P.clip_size(size),
                push_flag,
            )
            self._len += 1
        return self.current()

    def current(self) -> dict[str, np.ndarray | int]:
        ppi_batch = self._ppi[None, ...]
        len_batch = np.array([self._len])
        return {
            "tokens": tokenize(ppi_batch, len_batch)[0],
            "cont": continuous(ppi_batch, len_batch)[0],
            "mask": padding_mask(len_batch)[0],
            "ppi_len": self._len,
        }
