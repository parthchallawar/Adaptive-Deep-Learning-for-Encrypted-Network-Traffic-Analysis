"""Packet-to-flow aggregation (spec 002).

Turns a stream of parsed packets into flow records carrying the PPI sequence and
flow statistics. Transport-agnostic and capture-agnostic: it never sees payload
bytes, only the header-derived facts in :class:`ParsedPacket`.

Two properties matter for the research and are enforced here:

* **Only payload-carrying packets enter the PPI.** Pure ACKs, bare SYNs and
  FINs without data are counted in the flow statistics but do not consume one of
  the 30 PPI slots. This matches how the CESNET dataset was produced, and it is
  the difference between a model that sees 30 useful packets and one that sees a
  handshake.
* **Direction is relative to the flow initiator**, never to an IP or a port, so
  the representation carries no addressing shortcut.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from adl_etc.data import ppi as P

# TCP flag bits (dpkt exposes the same values).
TH_FIN, TH_SYN, TH_RST, TH_PUSH, TH_ACK, TH_URG = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20

PROTO_TCP, PROTO_UDP = 6, 17

Endpoint = tuple[bytes, int]
"""An (IP bytes, port) pair. IPs stay as packed bytes; they never reach a model."""

FlowKey = tuple[Endpoint, Endpoint, int]
"""Direction-independent key: the two endpoints in sorted order, plus protocol."""


@dataclass(slots=True)
class ParsedPacket:
    """One packet, reduced to what flow construction needs."""

    ts: float
    """Capture timestamp in seconds."""

    src: Endpoint
    dst: Endpoint
    proto: int
    payload_len: int
    """L4 payload bytes, derived from header length fields, not captured bytes."""

    flags: int = 0
    """TCP flag bits; zero for UDP."""


class EndReason:
    FIN = "fin"
    RST = "rst"
    IDLE = "idle"
    ACTIVE = "active"
    EOF = "eof"


@dataclass(slots=True)
class FlowRecord:
    """A completed flow, ready to be tensorised (spec 003)."""

    key: FlowKey
    proto: int
    start_ts: float
    end_ts: float
    ppi: np.ndarray
    """int16 array of shape ``[K_MAX, PPI_CHANNELS]``, zero-padded."""

    ppi_len: int
    packets: int
    packets_rev: int
    bytes: int
    bytes_rev: int
    flags_seen: int
    end_reason: str
    session_id: int = 0
    label: int = -1

    @property
    def duration(self) -> float:
        return self.end_ts - self.start_ts

    def flowstats(self) -> np.ndarray:
        """Flow-level features, shape ``[FLOWSTATS_DIM]``, float32, unscaled."""
        used = self.ppi[: self.ppi_len]
        sizes_fwd = [int(r[P.SIZE_POS]) for r in used if r[P.DIR_POS] == P.DIR_FWD]
        sizes_rev = [int(r[P.SIZE_POS]) for r in used if r[P.DIR_POS] == P.DIR_REV]
        ipt_fwd = [int(r[P.IPT_POS]) for r in used if r[P.DIR_POS] == P.DIR_FWD]
        ipt_rev = [int(r[P.IPT_POS]) for r in used if r[P.DIR_POS] == P.DIR_REV]

        base = [
            float(self.bytes),
            float(self.bytes_rev),
            float(self.packets),
            float(self.packets_rev),
            float(self.duration),
            float(self.ppi_len),
            float(used[:, P.IPT_POS].sum()) if self.ppi_len else 0.0,
            float(self.ppi_roundtrips()),
        ]
        hists = np.concatenate(
            [
                P.histogram(sizes_fwd, P.SIZE_HIST_EDGES),
                P.histogram(sizes_rev, P.SIZE_HIST_EDGES),
                P.histogram(ipt_fwd, P.IPT_HIST_EDGES),
                P.histogram(ipt_rev, P.IPT_HIST_EDGES),
            ]
        ).astype(np.float32)
        flags = [
            float(bool(self.flags_seen & bit))
            for bit in (TH_SYN, TH_FIN, TH_RST, TH_PUSH, TH_ACK, TH_URG)
        ]
        out = np.concatenate(
            [np.asarray(base, dtype=np.float32), hists, np.asarray(flags, np.float32)]
        )
        assert out.shape == (P.FLOWSTATS_DIM,), out.shape
        return out

    def ppi_roundtrips(self) -> int:
        """Direction changes in the PPI sequence, halved: how many exchanges happened."""
        if self.ppi_len < 2:
            return 0
        dirs = self.ppi[: self.ppi_len, P.DIR_POS]
        changes = int(np.count_nonzero(dirs[1:] != dirs[:-1]))
        return changes // 2


@dataclass(slots=True)
class _FlowState:
    key: FlowKey
    initiator: Endpoint
    proto: int
    start_ts: float
    last_ts: float
    last_ppi_ts: float
    ppi: np.ndarray = field(default_factory=P.empty_ppi)
    ppi_len: int = 0
    packets: int = 0
    packets_rev: int = 0
    bytes: int = 0
    bytes_rev: int = 0
    flags_seen: int = 0
    fin_seen: int = 0
    """Bitmask: 1 = initiator sent FIN, 2 = responder sent FIN."""

    close_at: float | None = None


class FlowBuilder:
    """Aggregates packets into flows with ipfixprobe-style timeouts.

    Args:
        inactive_timeout: seconds of silence before a TCP flow is exported.
        active_timeout: maximum lifetime of a flow regardless of activity.
        fin_linger: grace period after both sides have sent FIN, so that a
            trailing ACK or retransmission still lands in the same flow.
        k_max: PPI slots per flow.
        udp_idle_timeout: seconds of silence before a UDP/QUIC flow is
            exported. Defaults to ``inactive_timeout`` when not given; spec
            002 documents UDP/QUIC as needing its own (typically shorter)
            value since there is no FIN/RST to close the flow explicitly.

    The TCP defaults follow the exporter settings used for the CESNET
    captures. Yielded records are ordered by expiry, not by flow start.
    """

    def __init__(
        self,
        inactive_timeout: float = 30.0,
        active_timeout: float = 300.0,
        fin_linger: float = 5.0,
        k_max: int = P.K_MAX,
        session_id: int = 0,
        udp_idle_timeout: float | None = None,
    ) -> None:
        self.inactive_timeout = inactive_timeout
        self.active_timeout = active_timeout
        self.fin_linger = fin_linger
        self.k_max = k_max
        self.session_id = session_id
        self.udp_idle_timeout = inactive_timeout if udp_idle_timeout is None else udp_idle_timeout
        self._flows: dict[FlowKey, _FlowState] = {}
        self.skipped_packets = 0
        self.clock_jumps = 0

        # A full expiry sweep is O(open flows); doing it on every packet made
        # it the single biggest cost in the pipeline (measured: 34% of wall
        # time on a real capture, ahead of all dpkt parsing combined), even
        # though only one flow -- the one this packet belongs to -- actually
        # needs an up-to-the-packet answer. That one is still always checked
        # exactly, inline in add() below; the sweep for *other*, untouched
        # flows is throttled to this interval instead, since a flow nobody is
        # sending packets to has no reader waiting on its exact export time.
        self._sweep_interval = max(
            min(inactive_timeout, active_timeout, fin_linger, self.udp_idle_timeout) / 4,
            1e-6,
        )
        self._last_sweep: float | None = None

    # -- public API ---------------------------------------------------------

    def add(self, pkt: ParsedPacket) -> Iterator[FlowRecord]:
        """Feed one packet; yields any flows that expired at this timestamp."""
        yield from self._sweep_if_due(pkt.ts)

        key = flow_key(pkt.src, pkt.dst, pkt.proto)
        state = self._flows.get(key)
        if state is not None:
            reason = self._expiry_reason(state, pkt.ts)
            if reason is not None:
                # Same 5-tuple reused after this specific flow timed out: it
                # must become a new flow (spec 002's "port reuse" edge case),
                # never silently absorbed into the stale one, regardless of
                # whether the throttled sweep above has caught up to it yet.
                yield self._close(key, reason)
                state = None
        if state is None:
            state = _FlowState(
                key=key,
                initiator=pkt.src,
                proto=pkt.proto,
                start_ts=pkt.ts,
                last_ts=pkt.ts,
                last_ppi_ts=pkt.ts,
            )
            self._flows[key] = state

        if pkt.ts < state.last_ts:
            self.clock_jumps += 1
        state.last_ts = max(state.last_ts, pkt.ts)

        forward = pkt.src == state.initiator
        direction = P.DIR_FWD if forward else P.DIR_REV
        if forward:
            state.packets += 1
            state.bytes += pkt.payload_len
        else:
            state.packets_rev += 1
            state.bytes_rev += pkt.payload_len
        state.flags_seen |= pkt.flags

        # Only payload-carrying packets occupy a PPI slot.
        if pkt.payload_len > 0 and state.ppi_len < self.k_max:
            ipt = 0 if state.ppi_len == 0 else P.clip_ipt((pkt.ts - state.last_ppi_ts) * 1000.0)
            state.ppi[state.ppi_len] = (
                ipt,
                direction,
                P.clip_size(pkt.payload_len),
                1 if pkt.flags & TH_PUSH else 0,
            )
            state.ppi_len += 1
            state.last_ppi_ts = pkt.ts

        if pkt.proto == PROTO_TCP:
            if pkt.flags & TH_RST:
                yield self._close(key, EndReason.RST)
                return
            if pkt.flags & TH_FIN:
                state.fin_seen |= 1 if forward else 2
                if state.fin_seen == 3:
                    state.close_at = pkt.ts + self.fin_linger

    def flush(self) -> Iterator[FlowRecord]:
        """Export every remaining flow. Call once the capture is exhausted.

        A flow whose two sides both sent FIN is reported as a clean close even
        though no later packet arrived to trip its linger timer.
        """
        for key in list(self._flows):
            closed = self._flows[key].fin_seen == 3
            yield self._close(key, EndReason.FIN if closed else EndReason.EOF)

    @property
    def open_flows(self) -> int:
        return len(self._flows)

    # -- internals ----------------------------------------------------------

    def _expiry_reason(self, state: _FlowState, now: float) -> str | None:
        """Which :class:`EndReason` applies to ``state`` at time ``now``, or
        ``None`` if it is still alive. The single source of truth for all
        three expiry conditions, shared by the per-packet check in
        :meth:`add` and the throttled sweep in :meth:`_expire`."""
        if state.close_at is not None and now >= state.close_at:
            return EndReason.FIN
        idle_timeout = self.udp_idle_timeout if state.proto == PROTO_UDP else self.inactive_timeout
        if now - state.last_ts >= idle_timeout:
            return EndReason.IDLE
        if now - state.start_ts >= self.active_timeout:
            return EndReason.ACTIVE
        return None

    def _sweep_if_due(self, now: float) -> Iterator[FlowRecord]:
        if self._last_sweep is not None and now - self._last_sweep < self._sweep_interval:
            return
        self._last_sweep = now
        yield from self._expire(now)

    def _expire(self, now: float) -> Iterator[FlowRecord]:
        for key, state in list(self._flows.items()):
            reason = self._expiry_reason(state, now)
            if reason is not None:
                yield self._close(key, reason)

    def _close(self, key: FlowKey, reason: str) -> FlowRecord:
        s = self._flows.pop(key)
        return FlowRecord(
            key=s.key,
            proto=s.proto,
            start_ts=s.start_ts,
            end_ts=s.last_ts,
            ppi=s.ppi,
            ppi_len=s.ppi_len,
            packets=s.packets,
            packets_rev=s.packets_rev,
            bytes=s.bytes,
            bytes_rev=s.bytes_rev,
            flags_seen=s.flags_seen,
            end_reason=reason,
            session_id=self.session_id,
        )


def flow_key(src: Endpoint, dst: Endpoint, proto: int) -> FlowKey:
    """Direction-independent key, so both halves of a conversation collide."""
    return (src, dst, proto) if src <= dst else (dst, src, proto)
