"""PCAP reading, the default backend (spec 002).

Pure Python via ``dpkt``: runs on Windows with no external tooling, which is why
it is the default rather than the Linux-only exporters. It parses headers only
and never copies payload bytes, so the privacy guarantee holds at the source.

Payload sizes come from IP and L4 header *length fields*, not from how many bytes
the capture actually stored. A capture taken with a small snaplen therefore
yields the same PPI as a full capture, which matters because several public
datasets are truncated.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from pathlib import Path

import dpkt

from adl_etc.data.flows import PROTO_TCP, PROTO_UDP, ParsedPacket

# libpcap link types we understand.
DLT_EN10MB = 1
DLT_RAW = 101
DLT_RAW_ALT = 12
DLT_LINUX_SLL = 113
DLT_NULL = 0

_PCAPNG_MAGIC = 0x0A0D0D0A
_PCAP_MAGICS = (0xA1B2C3D4, 0xD4C3B2A1, 0xA1B23C4D, 0x4D3CB2A1)


class PcapStats:
    """Counters for packets the parser could not use. Never silently discarded."""

    def __init__(self) -> None:
        self.non_ip = 0
        self.non_tcp_udp = 0
        self.malformed = 0
        self.tunnelled = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "non_ip": self.non_ip,
            "non_tcp_udp": self.non_tcp_udp,
            "malformed": self.malformed,
            "tunnelled": self.tunnelled,
        }

    @property
    def total_skipped(self) -> int:
        return self.non_ip + self.non_tcp_udp + self.malformed + self.tunnelled


def reader_for(fh) -> dpkt.pcap.Reader | dpkt.pcapng.Reader:
    """Wrap an open binary handle in the right dpkt reader, detected by magic bytes."""
    magic = struct.unpack(">I", fh.read(4))[0]
    fh.seek(0)
    if magic == _PCAPNG_MAGIC:
        return dpkt.pcapng.Reader(fh)
    if magic in _PCAP_MAGICS or struct.unpack("<I", struct.pack(">I", magic))[0] in _PCAP_MAGICS:
        return dpkt.pcap.Reader(fh)
    raise ValueError(f"not a pcap or pcapng file (magic {magic:#x})")


def parse_pcap(path: str | Path, stats: PcapStats | None = None) -> Iterator[ParsedPacket]:
    """Yield one :class:`ParsedPacket` per TCP/UDP packet, in capture order.

    Packets that cannot be used are counted in ``stats`` rather than dropped
    silently, so a capture that yields nothing is distinguishable from one the
    parser did not understand.
    """
    stats = stats if stats is not None else PcapStats()
    with open(path, "rb") as fh:
        reader = reader_for(fh)
        linktype = getattr(reader, "datalink", lambda: DLT_EN10MB)()
        for ts, buf in reader:
            pkt = parse_frame(ts, buf, linktype, stats)
            if pkt is not None:
                yield pkt


def parse_frame(ts: float, buf: bytes, linktype: int, stats: PcapStats) -> ParsedPacket | None:
    """Decode one link-layer frame. Returns None when the packet is not usable."""
    try:
        ip = _strip_link_layer(buf, linktype, stats)
    except (dpkt.UnpackError, IndexError, struct.error):
        stats.malformed += 1
        return None
    if ip is None:
        return None

    if isinstance(ip, dpkt.ip.IP):
        src, dst = ip.src, ip.dst
        l4_len = ip.len - (ip.hl * 4)
        proto = ip.p
        # Only the first fragment carries L4 headers; later ones are unusable.
        if ip.offset:
            stats.malformed += 1
            return None
    elif isinstance(ip, dpkt.ip6.IP6):
        src, dst = ip.src, ip.dst
        l4_len = ip.plen
        proto = ip.nxt
    else:
        stats.non_ip += 1
        return None

    l4 = ip.data
    if proto == PROTO_TCP and isinstance(l4, dpkt.tcp.TCP):
        payload_len = l4_len - (l4.off * 4)
        flags = l4.flags
    elif proto == PROTO_UDP and isinstance(l4, dpkt.udp.UDP):
        payload_len = l4_len - 8
        flags = 0
    else:
        stats.non_tcp_udp += 1
        return None

    if payload_len < 0:
        stats.malformed += 1
        return None

    return ParsedPacket(
        ts=float(ts),
        src=(bytes(src), int(l4.sport)),
        dst=(bytes(dst), int(l4.dport)),
        proto=proto,
        payload_len=int(payload_len),
        flags=int(flags),
    )


def _strip_link_layer(buf: bytes, linktype: int, stats: PcapStats):
    """Peel the link layer (and up to one VLAN tag) off, returning the IP header."""
    if linktype in (DLT_RAW, DLT_RAW_ALT):
        version = (buf[0] >> 4) if buf else 0
        return dpkt.ip.IP(buf) if version == 4 else dpkt.ip6.IP6(buf)

    if linktype == DLT_LINUX_SLL:
        return dpkt.sll.SLL(buf).data

    if linktype == DLT_NULL:
        return dpkt.loopback.Loopback(buf).data

    eth = dpkt.ethernet.Ethernet(buf)
    payload = eth.data
    # dpkt already unwraps 802.1Q into .data; a second tag is a tunnel we skip.
    if isinstance(payload, dpkt.ethernet.VLANtag8021Q):
        stats.tunnelled += 1
        return None
    if not isinstance(payload, dpkt.ip.IP | dpkt.ip6.IP6):
        stats.non_ip += 1
        return None
    return payload
