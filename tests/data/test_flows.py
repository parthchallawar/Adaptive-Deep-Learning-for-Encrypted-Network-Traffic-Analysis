"""Flow construction and PPI extraction (specs 002, 003).

The golden array in :func:`test_reference_flow_ppi` is the contract every other
backend must reproduce. If it changes, every exported shard is invalidated.
"""

from __future__ import annotations

import numpy as np
import pytest

from adl_etc.data import ppi as P
from adl_etc.data.flows import (
    PROTO_TCP,
    EndReason,
    FlowBuilder,
    ParsedPacket,
    flow_key,
)
from adl_etc.data.pcap_source import PcapStats, parse_pcap
from tests.conftest import (
    CLIENT_IP,
    CLIENT_PORT,
    SERVER_IP,
    SERVER_PORT,
    TH_ACK,
    TH_RST,
    Pkt,
    write_pcap,
)

CLIENT = (CLIENT_IP, CLIENT_PORT)
SERVER = (SERVER_IP, SERVER_PORT)

#: Expected PPI rows for ``REFERENCE_PACKETS``: (ipt_ms, direction, size, push).
EXPECTED_PPI = [
    (0, P.DIR_FWD, 517, 1),  # ClientHello; first packet so IPT is zero by definition
    (45, P.DIR_REV, 1460, 0),  # 30 ms -> 75 ms, the handshake ACKs in between skipped
    (5, P.DIR_REV, 1460, 0),
    (10, P.DIR_REV, 1460, 0),  # retransmission: an observer counts it
    (30, P.DIR_FWD, 80, 1),
    (380, P.DIR_REV, P.SIZE_MAX, 1),  # 2000 bytes clipped
]


def build_flows(path, **kwargs):
    """Run a capture end to end and return the finished flow records."""
    builder = FlowBuilder(**kwargs)
    records = []
    for pkt in parse_pcap(path):
        records.extend(builder.add(pkt))
    records.extend(builder.flush())
    return records


def test_reference_flow_ppi(reference_pcap):
    (flow,) = build_flows(reference_pcap)

    assert flow.ppi_len == len(EXPECTED_PPI)
    np.testing.assert_array_equal(
        flow.ppi[: flow.ppi_len],
        np.array(EXPECTED_PPI, dtype=P.PPI_DTYPE),
    )
    # Everything past ppi_len stays zero so padding is unambiguous.
    assert not flow.ppi[flow.ppi_len :].any()


def test_zero_payload_packets_never_consume_a_slot(reference_pcap):
    """The handshake and pure ACKs are counted but must not fill PPI slots."""
    (flow,) = build_flows(reference_pcap)

    assert flow.ppi_len == 6
    assert flow.packets + flow.packets_rev == 12  # every packet still counted
    assert (flow.ppi[: flow.ppi_len, P.SIZE_POS] > 0).all()


def test_first_packet_defines_the_forward_direction(reference_pcap):
    (flow,) = build_flows(reference_pcap)

    assert flow.ppi[0, P.DIR_POS] == P.DIR_FWD
    assert flow.ppi[0, P.IPT_POS] == 0
    assert set(np.unique(flow.ppi[: flow.ppi_len, P.DIR_POS])) <= {P.DIR_FWD, P.DIR_REV}


def test_byte_and_packet_counters_split_by_direction(reference_pcap):
    (flow,) = build_flows(reference_pcap)

    assert flow.packets == 5  # SYN, pure ACK, 517, 80, FIN
    assert flow.bytes == 517 + 80
    assert flow.packets_rev == 7  # SYN/ACK, pure ACK, three data, the 2000, FIN
    assert flow.bytes_rev == 1460 * 3 + 2000  # unclipped in the counters
    assert flow.end_reason == EndReason.FIN


def test_padding_mask_matches_ppi_len(reference_pcap):
    (flow,) = build_flows(reference_pcap)

    mask = P.padding_mask(flow.ppi_len)
    assert mask.sum() == flow.ppi_len
    assert mask[: flow.ppi_len].all()
    assert not mask[flow.ppi_len :].any()


def test_flowstats_shape_and_histograms(reference_pcap):
    (flow,) = build_flows(reference_pcap)
    stats = flow.flowstats()

    assert stats.shape == (P.FLOWSTATS_DIM,)
    assert stats.dtype == np.float32
    names = P.FLOWSTATS_COLUMNS
    assert stats[names.index("PPI_LEN")] == 6
    # Three forward and three reverse payload packets land in the size histograms.
    fwd = stats[names.index("PSIZE_HIST_0") : names.index("PSIZE_HIST_0") + 8]
    rev = stats[names.index("PSIZE_HIST_REV_0") : names.index("PSIZE_HIST_REV_0") + 8]
    assert fwd.sum() == 2
    assert rev.sum() == 4


def test_truncated_capture_gives_identical_sizes(tmp_path, reference_pcap):
    """A small snaplen must not change the PPI: sizes come from header fields."""
    full = build_flows(reference_pcap)[0]

    truncated = tmp_path / "truncated.pcap"
    import dpkt

    from tests.conftest import REFERENCE_PACKETS, _frame

    with open(truncated, "wb") as fh:
        writer = dpkt.pcap.Writer(fh)
        for pkt in REFERENCE_PACKETS:
            writer.writepkt(_frame(pkt)[:80], ts=1_700_000_000.0 + pkt.offset)

    short = build_flows(truncated)[0]
    np.testing.assert_array_equal(full.ppi, short.ppi)
    assert full.ppi_len == short.ppi_len


def test_ppi_stops_at_k_max_but_counters_continue(tmp_path):
    packets = [Pkt(0.001 * i, 100, i % 2 == 0) for i in range(P.K_MAX + 15)]
    path = write_pcap(tmp_path / "long.pcap", packets)

    (flow,) = build_flows(path)

    assert flow.ppi_len == P.K_MAX
    assert flow.packets + flow.packets_rev == P.K_MAX + 15


def test_flow_key_is_direction_independent():
    assert flow_key(CLIENT, SERVER, PROTO_TCP) == flow_key(SERVER, CLIENT, PROTO_TCP)


def test_idle_timeout_starts_a_new_flow():
    builder = FlowBuilder(inactive_timeout=30.0)
    first = ParsedPacket(ts=0.0, src=CLIENT, dst=SERVER, proto=PROTO_TCP, payload_len=100)
    later = ParsedPacket(ts=45.0, src=CLIENT, dst=SERVER, proto=PROTO_TCP, payload_len=100)

    assert list(builder.add(first)) == []
    expired = list(builder.add(later))

    assert len(expired) == 1
    assert expired[0].end_reason == EndReason.IDLE
    assert builder.open_flows == 1  # the second packet opened a fresh flow


def test_rst_closes_the_flow_immediately():
    builder = FlowBuilder()
    data = ParsedPacket(ts=0.0, src=CLIENT, dst=SERVER, proto=PROTO_TCP, payload_len=10)
    rst = ParsedPacket(
        ts=0.1, src=SERVER, dst=CLIENT, proto=PROTO_TCP, payload_len=0, flags=TH_RST | TH_ACK
    )

    list(builder.add(data))
    (record,) = list(builder.add(rst))

    assert record.end_reason == EndReason.RST
    assert builder.open_flows == 0


def test_clock_jump_clamps_ipt_to_zero():
    builder = FlowBuilder()
    forward = ParsedPacket(ts=10.0, src=CLIENT, dst=SERVER, proto=PROTO_TCP, payload_len=50)
    backward = ParsedPacket(ts=9.5, src=SERVER, dst=CLIENT, proto=PROTO_TCP, payload_len=50)

    list(builder.add(forward))
    list(builder.add(backward))
    (record,) = list(builder.flush())

    assert record.ppi[1, P.IPT_POS] == 0
    assert builder.clock_jumps == 1


def test_udp_flows_have_no_push_flag(tmp_path):
    builder = FlowBuilder()
    pkt = ParsedPacket(ts=0.0, src=CLIENT, dst=SERVER, proto=17, payload_len=1200)
    list(builder.add(pkt))
    (record,) = list(builder.flush())

    assert record.ppi[0, P.PUSH_POS] == 0
    assert record.ppi[0, P.SIZE_POS] == 1200


def test_udp_idle_timeout_is_independent_of_tcp():
    """spec 002: UDP/QUIC gets its own idle timeout, since there is no
    FIN/RST to close the flow explicitly. A UDP flow must expire on
    udp_idle_timeout even while inactive_timeout (TCP's) would not yet
    have fired, and vice versa."""
    builder = FlowBuilder(inactive_timeout=30.0, udp_idle_timeout=5.0)
    udp_pkt = ParsedPacket(ts=0.0, src=CLIENT, dst=SERVER, proto=17, payload_len=100)
    tcp_pkt = ParsedPacket(ts=0.0, src=CLIENT, dst=SERVER, proto=PROTO_TCP, payload_len=100)

    list(builder.add(udp_pkt))
    list(builder.add(tcp_pkt))
    # at t=10: past udp_idle_timeout (5s) but not inactive_timeout (30s)
    later = ParsedPacket(ts=10.0, src=CLIENT, dst=SERVER, proto=17, payload_len=1)
    expired = list(builder.add(later))

    assert len(expired) == 1
    assert expired[0].proto == 17
    assert expired[0].end_reason == EndReason.IDLE
    assert builder.open_flows == 2  # the new UDP flow, plus the still-open TCP one


def test_udp_idle_timeout_defaults_to_inactive_timeout():
    """Backward compatible: omitting udp_idle_timeout preserves the old
    single-timeout behaviour exactly."""
    builder = FlowBuilder(inactive_timeout=12.0)
    assert builder.udp_idle_timeout == 12.0


def test_key_reuse_is_caught_even_when_sweep_is_throttled():
    """Regression test for the throttled-sweep optimisation (plan T4: the
    old per-packet full scan of every open flow was 34% of wall time on a
    real capture). A same-key reuse after that *specific* flow's own timeout
    must always start a new flow, never get merged into the stale one, even
    when many other flows' traffic has kept the throttled sweep from having
    caught up to this one yet -- the realistic case this optimisation has to
    stay correct under, not just an idle laptop.
    """
    other = (SERVER_IP, 8080)  # a second flow, distinct from CLIENT/SERVER's
    builder = FlowBuilder(inactive_timeout=10.0, active_timeout=10_000.0, fin_linger=10_000.0)
    # sweep_interval = min(10, 10_000, 10_000, 10) / 4 = 2.5s

    def a_pkt(ts: float) -> ParsedPacket:
        return ParsedPacket(ts=ts, src=CLIENT, dst=SERVER, proto=PROTO_TCP, payload_len=10)

    def other_pkt(ts: float) -> ParsedPacket:
        return ParsedPacket(ts=ts, src=CLIENT, dst=other, proto=PROTO_TCP, payload_len=10)

    list(builder.add(a_pkt(0.0)))  # flow A starts, idle from here on
    for ts in range(1, 11):  # 1.0s cadence keeps last_sweep moving without ever
        list(builder.add(other_pkt(float(ts))))  # landing exactly on flow A's 10.0s boundary

    # Flow A is now individually past its 10s inactive_timeout, but the last
    # sweep tick (driven by "other"'s traffic) ran at t=9.0, where flow A had
    # only been idle 9s -- not yet expired then -- so nothing has closed it.
    expired = list(builder.add(a_pkt(10.1)))

    assert len(expired) == 1
    assert expired[0].end_reason == EndReason.IDLE
    assert expired[0].packets == 1  # the stale flow A alone, not merged with the new packet
    assert builder.open_flows == 2  # a fresh flow A, plus the still-open "other" flow


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-5.0, 0), (0.0, 0), (12.7, 13), (4.999, 5), (999_999.0, P.IPT_MAX_MS)],
)
def test_ipt_clipping(value, expected):
    """Rounds to nearest millisecond; see clip_ipt for why truncation is wrong."""
    assert P.clip_ipt(value) == expected


def test_non_ip_traffic_is_counted_not_dropped_silently(tmp_path):
    import dpkt

    path = tmp_path / "arp.pcap"
    arp = dpkt.ethernet.Ethernet(
        src=b"\x00\x11\x22\x33\x44\x55",
        dst=b"\xff" * 6,
        type=dpkt.ethernet.ETH_TYPE_ARP,
        data=dpkt.arp.ARP(),
    )
    with open(path, "wb") as fh:
        writer = dpkt.pcap.Writer(fh)
        writer.writepkt(bytes(arp), ts=1.0)

    stats = PcapStats()
    assert list(parse_pcap(path, stats)) == []
    assert stats.total_skipped == 1
