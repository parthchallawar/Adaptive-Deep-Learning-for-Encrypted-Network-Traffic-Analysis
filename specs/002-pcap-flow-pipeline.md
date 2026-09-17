# Spec 002: PCAP and Live Capture to Flows to PPI

- **Status:** draft
- **Owner:** Parth Challawar
- **Created:** 2026-09-17
- **Depends on:** 001. **Used by:** 003, 016, 017.

## Problem

The system must accept PCAP files or live traffic and turn them into the same per-packet-information (PPI) sequences the primary dataset provides, without touching payload bytes. If our features differ from CESNET's exporter (which packets count, how direction is defined, what "inter-packet time" means), models trained on D1 will not transfer to our own captures, and the demo will silently disagree with the research results.

## Goals

- Feature parity with CESNET's `ipfixprobe` PPI definition: first 30 **payload-carrying** packets, direction relative to the flow initiator, IPT in milliseconds (integer, first packet 0), size = L4 payload bytes, TCP PSH flag.
- Three interchangeable backends behind one interface:
  1. `ipfixprobe` (Docker, Linux): reference backend, bit-exact with D1.
  2. NFStream (`splt_analysis=30`, Linux/macOS): fast Python alternative.
  3. Pure-Python `dpkt` flow builder: Windows fallback, unit-testable, slow.
- Streaming mode for live capture: emit a PPI update per packet so that the anytime model can decide before the flow ends.
- Session-level metadata kept out of the model input but kept for grouping (spec 004 leakage rules).

## Non-goals

- Payload parsing, TLS handshake parsing beyond SNI extraction for *labelling* purposes only (never as a feature).
- Full IPFIX export, DPDK, or high-speed (>1 Gbps) capture.

## How it works

### Flow key and direction

- Key: `(ip_a, ip_b, port_a, port_b, proto)` normalised so that the first observed packet defines the client (`dir=+1`). Reverse packets get `dir=-1`.
- TCP: a flow starts at the first packet seen (SYN if present); ends on FIN/RST with a 5 s linger, or after 300 s idle, or after 1800 s active timeout (ipfixprobe defaults: inactive 30 s / active 300 s are configurable; we set the values in `configs/data/pcap.yaml` and document them).
- UDP/QUIC: same key; idle timeout 60 s.

### Which packets enter PPI

- Only packets with L4 payload length > 0 (excludes pure ACKs, SYN, FIN without data), matching CESNET's PPI definition ("TCP payload only, excludes zero-payload ACK packets").
- Stop recording after 30 payload packets; the flow keeps being tracked for flow statistics.
- IPT for packet i = timestamp_i minus timestamp_{i-1} of the previous *recorded* packet, in ms, clipped to [0, 65535] to fit int16-compatible storage (DataZoo clips IPT similarly).
- Size = payload bytes, clipped to [0, 1500] (jumbo frames are rare and clipped).
- PUSH flag: TCP PSH bit for TCP, 0 for UDP.

### Flow statistics (spec 003 lists the 43 fields)

Bytes/packets in each direction, duration, PPI length, PPI duration, round trips, 8-bin log histograms of sizes and IPTs per direction, TCP flag presence, flow end reason.

### Backends

```python
class FlowSource(Protocol):
    def flows(self) -> Iterator[FlowRecord]: ...          # batch (pcap)
    def packets(self) -> Iterator[PacketEvent]: ...       # streaming (live)

FlowRecord = {flow_id, key, start_ts, label_hint, ppi: int16[30,4], ppi_len, flowstats: float32[43], session_id}
PacketEvent = {flow_id, k, ipt, dir, size, push, ts}      # k = index within PPI (1..30)
```

- `IpfixprobeSource`: runs `ipfixprobe -i "pcap;file=<f>" -p pstats -p basicplus -o "text;..."` inside the `docker/ipfixprobe` image; parses its CSV. Used to *generate* D3/D4 shards.
- `NFStreamSource`: `NFStreamer(source=f, splt_analysis=30, statistical_analysis=True, idle_timeout=..., active_timeout=...)`; converts `splt_*` arrays to PPI (note NFStream direction encoding 0/1 is mapped to +1/-1 and NFStream includes zero-payload packets unless filtered; we filter by `splt_ps > 0`... see edge cases).
- `DpktSource`: minimal reassembly-free parser: Ethernet/IP/IPv6/TCP/UDP headers only, no payload copy; handles VLAN tags; ignores fragments beyond the first.

### Labelling PCAP datasets

- ISCX and USTC: label = file-level label from `labels.csv`; `session_id` = pcap file name (all flows in one capture file are one session for split-grouping).
- Optional SNI extraction from ClientHello for D3 sanity checks (e.g. confirm "youtube" flows), stored in a separate side table, never in shards.

### Live capture

- Windows: Scapy `AsyncSniffer` with Npcap, feeding `DpktSource`-equivalent logic; Linux: ipfixprobe with `-i "raw;ifc=eth0"` and a UNIX socket/text output tailed by the service.
- Emits `PacketEvent`s to the inference service (spec 016) as soon as a payload packet is seen.

## Inputs and outputs

- Inputs: `.pcap`/`.pcapng` files or an interface name; `configs/data/pcap.yaml` (timeouts, backend).
- Outputs: shards in the spec-003 format under `data/processed/<dataset>/`, plus `flows.parquet` with keys, timestamps, session ids, and end reasons for auditing.

## Edge cases

- **Truncated captures (snaplen):** payload length is taken from IP/TCP headers, not captured bytes, so truncation does not change sizes.
- **Retransmissions / duplicates:** ipfixprobe counts them; NFStream and dpkt backends count them too unless `dedup=true`. Default: count (parity with D1, whose week-10 artefact was precisely about skipping retransmissions).
- **Out-of-order packets:** recorded in arrival order (what an observer sees), never re-sorted by sequence number.
- **IPv6, VLAN, GRE:** IPv6 and 802.1Q supported; tunnels beyond one VLAN tag are skipped and counted in a `skipped_packets` stat.
- **Flows with < 30 payload packets:** PPI padded with zeros and `ppi_len` recorded; padding mask derived from `ppi_len` (spec 003).
- **Very long flows:** PPI stops at 30, flowstats keep counting until timeout.
- **Clock jumps / negative IPT:** clipped to 0 and counted.
- **Port reuse after timeout:** a new flow_id (key + start_ts).

## Performance considerations

- ipfixprobe processes tens of thousands of packets per second per core from pcap; the 28 GB ISCX set is a few hours in Docker.
- dpkt backend is about 50k to 150k packets/s; acceptable for tests and small captures only.
- Streaming path target: under 1 ms per PacketEvent end-to-end into the model (measured in spec 013).

## Testing

- **Parity test (mandatory):** a synthetic pcap generated with Scapy (known sizes, directions, gaps, PSH flags, one retransmission, one pure ACK) must produce identical PPI from all three backends; expected arrays are hand-written in the test.
- **Golden test against D1:** not possible directly (no raw pcaps), but a tiny public pcap with a known ipfixprobe output is committed under `tests/fixtures/` to guard against parser regressions.
- Property tests: padding mask equals `ppi_len`; IPT[0] == 0; direction of first packet == +1; sizes > 0 for recorded packets.
- Live smoke test (manual): capture 60 s of browsing, confirm flows appear in the dashboard with growing K.

## Interactions

- Spec 003 defines the shard schema and normalisation (this spec writes raw integers only).
- Spec 016 consumes `PacketEvent`s.
- Spec 019 builds the ipfixprobe Docker image.

## Success criteria

- Parity test passes on all backends.
- ISCX and USTC shards produced with per-file session ids and audit parquet.
- Live demo shows a first decision within the first 10 payload packets of a browsing flow.

## Open questions

- Owner's development OS is Windows 11: confirm WSL2 or Docker Desktop is acceptable for ipfixprobe/NFStream. Default: Docker Desktop.
- Timeout values: adopt ipfixprobe defaults or CESNET's production settings (documented in the Year22 paper)? Default: the paper's settings, recorded in `configs/data/pcap.yaml`.
