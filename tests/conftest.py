"""Shared fixtures (spec 020).

The synthetic capture here is the project's ground truth for flow construction:
its packets are chosen so that every rule in spec 002 is exercised and the
expected PPI can be written out by hand in the test.
"""

from __future__ import annotations

import http.server
import json
import socket
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
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


# --- local HTTP fixture (specs 001, 020) ------------------------------------
#
# Shared by every downloader test (test_download.py, test_ustc_download.py,
# test_iscx_download.py): a minimal HTTP server that can serve plain files
# (with Range support for resume), inject N failures before succeeding (for
# retry), and serve canned JSON (to stand in for GitHub's contents API).
# Runs on loopback only, so it's fast and needs no network access.


@dataclass
class HttpFixtureState:
    files: dict[str, bytes] = field(default_factory=dict)
    fail_count: dict[str, int] = field(default_factory=dict)
    json_routes: dict[str, object] = field(default_factory=dict)
    request_log: list[str] = field(default_factory=list)


class _FixtureHandler(http.server.BaseHTTPRequestHandler):
    state: HttpFixtureState

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass

    def handle_one_request(self) -> None:
        # A retry test's client can abandon a connection (e.g. after reading
        # a 500 and opening a fresh one) while this thread is still mid-write;
        # that shows up as ConnectionAbortedError/ConnectionResetError here,
        # not as a test failure, but the base class's default handle_error()
        # prints a full traceback to stderr for it regardless. Swallow just
        # those two, so a client hanging up early stays silent and anything
        # else still surfaces loudly.
        try:
            super().handle_one_request()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    def do_GET(self) -> None:  # noqa: N802
        self.state.request_log.append(self.path)

        if self.path in self.state.json_routes:
            body = json.dumps(self.state.json_routes[self.path]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        remaining = self.state.fail_count.get(self.path, 0)
        if remaining > 0:
            self.state.fail_count[self.path] = remaining - 1
            self.send_response(500)
            self.end_headers()
            return

        body = self.state.files.get(self.path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return

        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            start = int(range_header[len("bytes=") :].split("-")[0])
            chunk = body[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
        else:
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


@pytest.fixture
def http_fixture() -> Iterator[tuple[str, HttpFixtureState]]:
    state = HttpFixtureState()
    handler = type("_BoundFixtureHandler", (_FixtureHandler,), {"state": state})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield base_url, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
