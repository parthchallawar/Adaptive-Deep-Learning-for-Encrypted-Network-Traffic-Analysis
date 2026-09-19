"""Shared fixtures (spec 020).

The synthetic capture here is the project's ground truth for flow construction:
its packets are chosen so that every rule in spec 002 is exercised and the
expected PPI can be written out by hand in the test.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from pathlib import Path

import dpkt
import pytest

CLIENT_IP = socket.inet_aton("10.0.0.5")
SERVER_IP = socket.inet_aton("93.184.216.34")
CLIENT_PORT = 51_234
SERVER_PORT = 443

TH_FIN, TH_SYN, TH_RST, TH_PUSH, TH_ACK = 0x01, 0x02, 0x04, 0x08, 0x10


@dataclass(frozen=True)
class Pkt:
    """A packet to synthesise: offset in seconds, payload size, direction, flags."""

    offset: float
    payload: int
    to_server: bool
    flags: int = TH_ACK


def _frame(pkt: Pkt) -> bytes:
    src_ip, dst_ip = (CLIENT_IP, SERVER_IP) if pkt.to_server else (SERVER_IP, CLIENT_IP)
    sport, dport = (CLIENT_PORT, SERVER_PORT) if pkt.to_server else (SERVER_PORT, CLIENT_PORT)
    tcp = dpkt.tcp.TCP(
        sport=sport,
        dport=dport,
        seq=1,
        ack=1,
        off=5,
        flags=pkt.flags,
        win=64_240,
        data=b"\x00" * pkt.payload,
    )
    ip = dpkt.ip.IP(src=src_ip, dst=dst_ip, p=dpkt.ip.IP_PROTO_TCP, ttl=64, data=tcp)
    ip.len = len(ip)
    eth = dpkt.ethernet.Ethernet(
        src=b"\x00\x11\x22\x33\x44\x55",
        dst=b"\x66\x77\x88\x99\xaa\xbb",
        type=dpkt.ethernet.ETH_TYPE_IP,
        data=ip,
    )
    return bytes(eth)


def write_pcap(path: Path, packets: list[Pkt], base_ts: float = 1_700_000_000.0) -> Path:
    """Write a pcap containing exactly ``packets``."""
    with open(path, "wb") as fh:
        writer = dpkt.pcap.Writer(fh)
        for pkt in packets:
            writer.writepkt(_frame(pkt), ts=base_ts + pkt.offset)
    return path


#: One TLS-like flow. Each entry is (offset s, payload bytes, to_server, flags).
#: Covers: handshake with no payload, a pure ACK, a retransmission, a PSH flag,
#: a reverse-direction burst, and a clean FIN close.
REFERENCE_PACKETS: list[Pkt] = [
    Pkt(0.000, 0, True, TH_SYN),  # handshake: no payload, not in PPI
    Pkt(0.010, 0, False, TH_SYN | TH_ACK),  # not in PPI
    Pkt(0.020, 0, True, TH_ACK),  # pure ACK, not in PPI
    Pkt(0.030, 517, True, TH_ACK | TH_PUSH),  # PPI 0: ClientHello
    Pkt(0.055, 0, False, TH_ACK),  # pure ACK, not in PPI
    Pkt(0.075, 1460, False, TH_ACK),  # PPI 1: server flight
    Pkt(0.080, 1460, False, TH_ACK),  # PPI 2: same burst, 5 ms later
    Pkt(0.090, 1460, False, TH_ACK),  # PPI 3: retransmission, still counted
    Pkt(0.120, 80, True, TH_ACK | TH_PUSH),  # PPI 4: client finished
    Pkt(0.500, 2000, False, TH_ACK | TH_PUSH),  # PPI 5: size clipped to 1500
    Pkt(0.600, 0, True, TH_FIN | TH_ACK),
    Pkt(0.610, 0, False, TH_FIN | TH_ACK),
]


@pytest.fixture
def reference_pcap(tmp_path: Path) -> Path:
    """A capture whose expected PPI is spelled out in ``test_flows``."""
    return write_pcap(tmp_path / "reference.pcap", REFERENCE_PACKETS)
