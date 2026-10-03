"""Flow statistics as observed after K packets (plan phase 2 T4).

Built for spec 005's B1 (XGBoost), which was removed 2026-10-03. Kept because it
is the only leakage-safe early tabular feature in the project; it has no model
consumer at present.

The stored ``flowstats`` vector (:data:`~adl_etc.data.ppi.FLOWSTATS_COLUMNS`) is
computed from the **whole** flow: ``BYTES``, ``PACKETS`` and ``DURATION`` are
totals over every packet including pure ACKs and everything past the 30th
payload packet, and the FIN/RST flags are only ever set at the very end. A
tabular baseline that read those at "K=1" would be reading the future, would
look extraordinary at small K, and would invalidate the accuracy-vs-K curve the
project's earliness claim is built on.

So :func:`prefix_flowstats` recomputes **every** column from the first K PPI
entries alone and never touches the stored array. It is a pure function of the
PPI, which is also the principled reason to define it this way: the tabular
baseline then sees exactly the information the token and continuous views give
the deep models, and the comparison between them is about the model, not about
what each was allowed to see.

Consequences, deliberate and tested (``tests/data/test_prefix_stats.py``):

* Same 46 columns in the same order as the stored vector, so the layout is
  interchangeable, but the *meaning* is "as observed through packet K".
* ``prefix_flowstats(k=30)`` does **not** equal the stored vector. The PPI holds
  only payload-carrying packets, so ``PACKETS``/``BYTES``/``DURATION`` here count
  those (and payload sizes clipped to ``SIZE_MAX``), while the stored columns
  count every packet. Only the columns that are themselves derived from the PPI
  (``PPI_LEN``, ``PPI_DURATION``, ``PPI_ROUNDTRIPS`` and the four histograms)
  match exactly.
* The PPI carries a push flag but no other TCP flag, so ``FLAG_PSH`` is
  recomputed and the other five flag columns are **0**. Copying the stored
  flags would be the leak described above; dropping the columns would break the
  shared layout.
* Histograms are raw counts, like the stored ones; the ``Standardizer``
  renormalises them per group. Note the ``Standardizer`` is fit on *whole-flow*
  statistics, so its means and scales are the wrong ones for these values. That
  is harmless for tree models (a monotone rescaling changes nothing); a model
  that cares should fit its own standardiser on prefix features at the same K.
"""

from __future__ import annotations

import numpy as np

from adl_etc.data import ppi as P

assert len(P.SIZE_HIST_EDGES) + 1 == len(P.IPT_HIST_EDGES) + 1 == P.PHIST_BIN_COUNT

_COL = {name: i for i, name in enumerate(P.FLOWSTATS_COLUMNS)}


def _hist_columns(prefix: str) -> np.ndarray:
    return np.array([_COL[f"{prefix}_{i}"] for i in range(P.PHIST_BIN_COUNT)])


_SIZE_FWD, _SIZE_REV = _hist_columns("PSIZE_HIST"), _hist_columns("PSIZE_HIST_REV")
_IPT_FWD, _IPT_REV = _hist_columns("IPT_HIST"), _hist_columns("IPT_HIST_REV")

#: Columns that are themselves functions of the PPI alone, so a prefix at
#: ``k >= ppi_len`` must reproduce the stored value exactly. The rest describe
#: the whole flow and cannot.
PPI_DERIVED_COLUMNS: tuple[str, ...] = (
    "PPI_LEN",
    "PPI_DURATION",
    "PPI_ROUNDTRIPS",
    *(P.FLOWSTATS_COLUMNS[i] for i in (*_SIZE_FWD, *_SIZE_REV, *_IPT_FWD, *_IPT_REV)),
)


def _binned_counts(values: np.ndarray, edges: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``[N, PHIST_BIN_COUNT]`` counts of ``values[mask]`` per bin, per flow.

    Same binning as :func:`adl_etc.data.ppi.histogram` (``searchsorted``,
    ``side="left"``), but one ``bincount`` for the whole batch instead of a
    Python loop over flows.
    """
    n = values.shape[0]
    bins = np.searchsorted(edges, values, side="left")
    flat = (np.arange(n)[:, None] * P.PHIST_BIN_COUNT + bins)[mask]
    return np.bincount(flat, minlength=n * P.PHIST_BIN_COUNT).reshape(n, P.PHIST_BIN_COUNT)


def prefix_flowstats(ppi: np.ndarray, ppi_len: np.ndarray, k: int) -> np.ndarray:
    """Flow statistics using only the first ``min(k, ppi_len)`` PPI entries.

    Args:
        ppi: ``[N, K_MAX, PPI_CHANNELS]`` raw PPI (``ipt_ms, dir, size, push``).
        ppi_len: ``[N]`` number of real packets per flow, each in ``1..K_MAX``.
        k: prefix length in ``1..K_MAX``. A flow shorter than ``k`` uses all its
            packets, the same ``min(k, ppi_len)`` rule as
            :func:`adl_etc.evaluation.metrics.logits_at_k`.

    Returns:
        ``[N, FLOWSTATS_DIM]`` float32, unscaled, in ``FLOWSTATS_COLUMNS`` order.
        ``k=1`` is degenerate by construction (one packet, zero duration, empty
        timing histograms) and the resulting low accuracy is the honest result.
    """
    ppi = np.asarray(ppi)
    ppi_len = np.asarray(ppi_len)
    if ppi.ndim != 3 or ppi.shape[1:] != (P.K_MAX, P.PPI_CHANNELS):
        raise ValueError(f"ppi must be [N, {P.K_MAX}, {P.PPI_CHANNELS}], got {ppi.shape}")
    if ppi_len.shape != (ppi.shape[0],):
        raise ValueError(f"ppi_len must be [{ppi.shape[0]}], got {ppi_len.shape}")
    if isinstance(k, bool) or not isinstance(k, int | np.integer) or not 1 <= k <= P.K_MAX:
        raise ValueError(f"k must be an int in 1..{P.K_MAX}, got {k!r}")
    if ppi_len.size and (ppi_len.min() < 1 or ppi_len.max() > P.K_MAX):
        raise ValueError(
            f"ppi_len must be in 1..{P.K_MAX}; a zero-packet flow is dropped at write time "
            f"and cannot be classified (got min={ppi_len.min()}, max={ppi_len.max()})"
        )

    n = ppi.shape[0]
    effective = np.minimum(ppi_len, k)
    valid = np.arange(P.K_MAX)[None, :] < effective[:, None]

    ipt = ppi[:, :, P.IPT_POS].astype(np.int64)
    direction = ppi[:, :, P.DIR_POS]
    size = ppi[:, :, P.SIZE_POS].astype(np.int64)
    push = ppi[:, :, P.PUSH_POS]

    fwd = valid & (direction == P.DIR_FWD)
    rev = valid & (direction == P.DIR_REV)

    out = np.zeros((n, P.FLOWSTATS_DIM), dtype=np.float32)
    ppi_duration_ms = (ipt * valid).sum(axis=1)

    out[:, _COL["BYTES"]] = (size * fwd).sum(axis=1)
    out[:, _COL["BYTES_REV"]] = (size * rev).sum(axis=1)
    out[:, _COL["PACKETS"]] = fwd.sum(axis=1)
    out[:, _COL["PACKETS_REV"]] = rev.sum(axis=1)
    out[:, _COL["DURATION"]] = ppi_duration_ms / 1000.0  # stored DURATION is in seconds
    out[:, _COL["PPI_LEN"]] = effective
    out[:, _COL["PPI_DURATION"]] = ppi_duration_ms  # stored PPI_DURATION is in ms

    # Direction changes between consecutive real packets, halved: exchanges.
    both_valid = valid[:, 1:]  # a valid position implies the one before it is valid
    changes = ((direction[:, 1:] != direction[:, :-1]) & both_valid).sum(axis=1)
    out[:, _COL["PPI_ROUNDTRIPS"]] = changes // 2

    out[:, _SIZE_FWD] = _binned_counts(size, P.SIZE_HIST_EDGES, fwd)
    out[:, _SIZE_REV] = _binned_counts(size, P.SIZE_HIST_EDGES, rev)
    out[:, _IPT_FWD] = _binned_counts(ipt, P.IPT_HIST_EDGES, fwd)
    out[:, _IPT_REV] = _binned_counts(ipt, P.IPT_HIST_EDGES, rev)

    # The PPI carries a push flag and no other TCP flag: only FLAG_PSH is
    # recoverable. The other five stay 0 (see the module docstring).
    out[:, _COL["FLAG_PSH"]] = (valid & (push == 1)).any(axis=1)
    return out
