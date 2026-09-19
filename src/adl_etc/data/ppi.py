"""Per-packet information (PPI) schema.

The single source of truth for what a flow looks like numerically. Every backend
(spec 002) emits this layout and every model (spec 003) consumes it, so the
constants here must not be changed without re-exporting all shards.

Layout matches CESNET DataZoo's column order so that flows we extract from PCAP
and flows read from the published dataset are interchangeable::

    ppi[k] = (ipt_ms, direction, payload_size, push_flag)   for k in 0..K_MAX-1
"""

from __future__ import annotations

import numpy as np

# --- PPI array layout -------------------------------------------------------

K_MAX = 30
"""Packets recorded per flow. DataZoo's PPI_MAX_LEN; also our maximum prefix."""

IPT_POS, DIR_POS, SIZE_POS, PUSH_POS = 0, 1, 2, 3
PPI_CHANNELS = 4

PPI_COLUMNS = ("IPT", "DIR", "SIZE", "PUSH")

# Channels a model actually sees per transport. UDP/QUIC has no push flag.
TCP_CHANNELS = (IPT_POS, DIR_POS, SIZE_POS, PUSH_POS)
UDP_CHANNELS = (IPT_POS, DIR_POS, SIZE_POS)

# --- value ranges -----------------------------------------------------------

IPT_MAX_MS = 32_767
"""Inter-packet times are clipped here so the value fits in the *signed*
int16 that PPI_DTYPE uses for every channel, including direction (which needs
the sign bit for +1/-1). DataZoo's own IPT column is unsigned 16-bit and can
represent up to 65535; because our single array dtype is shared with a signed
column, our storable ceiling is half of that. A packet gap this long only
occurs with a non-default (long) inactive timeout, since the default 30 s
timeout closes a flow before any single gap could reach it, but the type must
still be correct rather than merely untriggered."""

SIZE_MAX = 1_500
"""Payload sizes above this are clipped; jumbo frames are rare and uninformative."""

DIR_FWD = 1
"""Client to server, where "client" is whoever sent the flow's first packet."""

DIR_REV = -1
"""Server to client."""

PPI_DTYPE = np.int16

# --- flow statistics --------------------------------------------------------

PHIST_BIN_COUNT = 8
"""Log-spaced histogram bins per direction, matching DataZoo's PHIST_BIN_COUNT."""

SIZE_HIST_EDGES = np.array([16, 32, 64, 128, 256, 512, 1024], dtype=np.int32)
"""Upper edges; 8 bins = 7 edges. Bin i counts sizes in (edges[i-1], edges[i]]."""

IPT_HIST_EDGES = np.array([1, 4, 16, 64, 256, 1024, 4096], dtype=np.int32)
"""Upper edges in milliseconds; 8 bins."""

FLOWSTATS_COLUMNS: tuple[str, ...] = (
    "BYTES",
    "BYTES_REV",
    "PACKETS",
    "PACKETS_REV",
    "DURATION",
    "PPI_LEN",
    "PPI_DURATION",
    "PPI_ROUNDTRIPS",
    *(f"PSIZE_HIST_{i}" for i in range(PHIST_BIN_COUNT)),
    *(f"PSIZE_HIST_REV_{i}" for i in range(PHIST_BIN_COUNT)),
    *(f"IPT_HIST_{i}" for i in range(PHIST_BIN_COUNT)),
    *(f"IPT_HIST_REV_{i}" for i in range(PHIST_BIN_COUNT)),
    "FLAG_SYN",
    "FLAG_FIN",
    "FLAG_RST",
    "FLAG_PSH",
    "FLAG_ACK",
    "FLAG_URG",
)
"""Flow-level features we compute ourselves from PCAP.

NOTE (spec 001, Path A verification): DataZoo ships 43 flow-statistics columns
whose exact names and order are only knowable from the published file. These are
our own, computed identically for every backend. Aligning the two is a task for
when the canonical HDF5 is first opened; until then, models that use flow
statistics must be trained and evaluated on one source at a time.
"""

FLOWSTATS_DIM = len(FLOWSTATS_COLUMNS)


def clip_ipt(ipt_ms: float) -> int:
    """Round an inter-packet time to whole milliseconds and clamp it.

    Rounding rather than truncating matters: capture timestamps are quantised to
    microseconds and carry float error, so a true 5 ms gap frequently arrives as
    4.999 ms. Truncation would bias every inter-packet time downward by up to a
    millisecond, which is a large fraction of the gaps that distinguish one
    application from another.

    Negative values occur with capture clock jumps and are treated as zero.
    """
    if ipt_ms <= 0:
        return 0
    return min(int(ipt_ms + 0.5), IPT_MAX_MS)


def clip_size(size: int) -> int:
    """Clamp a payload size to the storable range."""
    return min(max(int(size), 0), SIZE_MAX)


def histogram(values: list[int], edges: np.ndarray) -> np.ndarray:
    """Bucket ``values`` into ``len(edges) + 1`` log-spaced bins.

    Returns an int32 array of shape ``[PHIST_BIN_COUNT]``.
    """
    counts = np.zeros(len(edges) + 1, dtype=np.int32)
    if values:
        idx = np.searchsorted(edges, np.asarray(values, dtype=np.int32), side="left")
        np.add.at(counts, idx, 1)
    return counts


def empty_ppi() -> np.ndarray:
    """An all-zero PPI array of shape ``[K_MAX, PPI_CHANNELS]``."""
    return np.zeros((K_MAX, PPI_CHANNELS), dtype=PPI_DTYPE)


def padding_mask(ppi_len: int) -> np.ndarray:
    """Boolean mask of shape ``[K_MAX]``, True where a real packet sits."""
    mask = np.zeros(K_MAX, dtype=bool)
    mask[: min(ppi_len, K_MAX)] = True
    return mask
