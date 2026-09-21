"""Download/extraction primitives (spec 001, plan T3).

Resume/retry/discovery paths run against the local ``http_fixture`` in
``conftest.py``, not the network, so they're fast and deterministic. Marked
``integration`` because they open real sockets (loopback only) rather than
being pure in-process unit tests; still safe and fast enough to run every
time.
"""

from __future__ import annotations

import zipfile

import py7zr
import pytest

from adl_etc.data.download import (
    DownloadError,
    discover_links,
    download_file,
    extract_7z,
    extract_zip,
    fetch_text,
    list_github_dir,
)

pytestmark = pytest.mark.integration


def test_download_file_simple(http_fixture, tmp_path):
    base_url, state = http_fixture
    state.files["/a.bin"] = b"0123456789" * 100

    result = download_file(f"{base_url}/a.bin", tmp_path / "a.bin")

    assert (tmp_path / "a.bin").read_bytes() == state.files["/a.bin"]
    assert result.bytes == len(state.files["/a.bin"])
    assert result.resumed is False


def test_download_file_resumes_partial(http_fixture, tmp_path):
    base_url, state = http_fixture
    full = b"A" * 500 + b"B" * 500
    state.files["/a.bin"] = full
    dest = tmp_path / "a.bin"
    dest.write_bytes(full[:500])  # simulate a prior partial download

    result = download_file(f"{base_url}/a.bin", dest)

    assert dest.read_bytes() == full
    assert result.resumed is True


def test_download_file_already_complete_skips_request(http_fixture, tmp_path):
    base_url, state = http_fixture
    full = b"X" * 50
    state.files["/a.bin"] = full
    dest = tmp_path / "a.bin"
    dest.write_bytes(full)

    download_file(f"{base_url}/a.bin", dest, expected_bytes=len(full))

    assert state.request_log == []  # no request was made at all


def test_download_file_retries_on_5xx_then_succeeds(http_fixture, tmp_path):
    base_url, state = http_fixture
    state.files["/flaky.bin"] = b"hello"
    state.fail_count["/flaky.bin"] = 2  # fail twice, succeed on the 3rd attempt

    result = download_file(
        f"{base_url}/flaky.bin", tmp_path / "flaky.bin", max_retries=5, sleep=lambda _: None
    )

    assert result.attempts == 3
    assert (tmp_path / "flaky.bin").read_bytes() == b"hello"


def test_download_file_exhausts_retries(http_fixture, tmp_path):
    base_url, state = http_fixture
    state.files["/flaky.bin"] = b"hello"
    state.fail_count["/flaky.bin"] = 100  # never recovers

    with pytest.raises(DownloadError):
        download_file(
            f"{base_url}/flaky.bin", tmp_path / "flaky.bin", max_retries=3, sleep=lambda _: None
        )


def test_download_file_404_makes_exactly_one_request(http_fixture, tmp_path):
    base_url, state = http_fixture
    with pytest.raises(DownloadError):
        download_file(
            f"{base_url}/nope.bin", tmp_path / "a.bin", max_retries=5, sleep=lambda _: None
        )
    assert state.request_log == ["/nope.bin"]


def test_fetch_text(http_fixture):
    base_url, state = http_fixture
    state.files["/hello.txt"] = b"hello, world"
    assert fetch_text(f"{base_url}/hello.txt") == "hello, world"


def test_list_github_dir(http_fixture):
    base_url, state = http_fixture
    state.json_routes["/repos/owner/repo/contents/Benign?ref=master"] = [
        {"name": "a.pcap", "size": 10, "download_url": "http://x/a.pcap", "type": "file"},
        {"name": "sub", "size": 0, "download_url": None, "type": "dir"},
    ]

    files = list_github_dir("owner", "repo", "Benign", api_base=base_url)

    assert len(files) == 1  # the "dir" entry is excluded
    assert files[0].name == "a.pcap"
    assert files[0].bytes == 10


def test_list_github_dir_raises_on_missing(http_fixture):
    base_url, _state = http_fixture
    with pytest.raises(DownloadError):
        list_github_dir("owner", "repo", "missing", api_base=base_url)


def test_discover_links_filters_by_suffix():
    html = """
    <a href="a.pcap">a</a>
    <a href='b.PCAP'>b</a>
    <a href="readme.txt">readme</a>
    <a href="../">up</a>
    """
    links = discover_links(html, suffix=".pcap")
    assert links == ["a.pcap", "b.PCAP"]


# --- 7z extraction: no network needed, archives built on the fly -----------


def test_extract_7z_single_top_level_file(tmp_path):
    archive = tmp_path / "one.7z"
    src = tmp_path / "Shifu.pcap"
    src.write_bytes(b"fake pcap bytes")
    with py7zr.SevenZipFile(archive, mode="w") as z:
        z.write(src, arcname="Shifu.pcap")

    out = extract_7z(archive, tmp_path / "out")

    assert [p.name for p in out] == ["Shifu.pcap"]
    assert out[0].read_bytes() == b"fake pcap bytes"


def test_extract_7z_nested_multiple_files(tmp_path):
    archive = tmp_path / "multi.7z"
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "SMB-1.pcap").write_bytes(b"one")
    (src_dir / "SMB-2.pcap").write_bytes(b"two")
    with py7zr.SevenZipFile(archive, mode="w") as z:
        z.write(src_dir / "SMB-1.pcap", arcname="SMB/SMB-1.pcap")
        z.write(src_dir / "SMB-2.pcap", arcname="SMB/SMB-2.pcap")

    out = extract_7z(archive, tmp_path / "out")

    assert sorted(p.name for p in out) == ["SMB-1.pcap", "SMB-2.pcap"]


def test_extract_7z_no_matching_member_raises(tmp_path):
    archive = tmp_path / "empty.7z"
    src = tmp_path / "readme.txt"
    src.write_bytes(b"not a pcap")
    with py7zr.SevenZipFile(archive, mode="w") as z:
        z.write(src, arcname="readme.txt")

    with pytest.raises(DownloadError):
        extract_7z(archive, tmp_path / "out")


def test_extract_7z_collision_raises(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "Shifu.pcap").write_bytes(b"already here")

    archive = tmp_path / "one.7z"
    src = tmp_path / "Shifu.pcap.src"
    src.write_bytes(b"new content")
    with py7zr.SevenZipFile(archive, mode="w") as z:
        z.write(src, arcname="Shifu.pcap")

    with pytest.raises(DownloadError):
        extract_7z(archive, dest)


# --- zip extraction: matches ISCX's real archive shape (flat, no subfolder) --


def test_extract_zip_flat_multiple_files(tmp_path):
    # Matches the real ISCX archive shape (VPN-PCAPS-01.zip, plan T3.1):
    # several files directly at the top level, no subfolder.
    archive = tmp_path / "VPN-PCAPs-01.zip"
    with zipfile.ZipFile(archive, mode="w") as z:
        z.writestr("vpn_aim_chat1a.pcap", b"one")
        z.writestr("vpn_bittorrent.pcap", b"two")

    out = extract_zip(archive, tmp_path / "out")

    assert sorted(p.name for p in out) == ["vpn_aim_chat1a.pcap", "vpn_bittorrent.pcap"]
    assert (tmp_path / "out" / "vpn_aim_chat1a.pcap").read_bytes() == b"one"


def test_extract_zip_matches_pcapng_by_default(tmp_path):
    # Real finding (NonVPN-PCAPs-01.zip): 11 of 23 members are .pcapng, the
    # newer capture format pcap_source.py already parses (spec 002). The
    # default suffix must match both, or extraction silently drops them.
    archive = tmp_path / "NonVPN-PCAPs-01.zip"
    with zipfile.ZipFile(archive, mode="w") as z:
        z.writestr("aim_chat_3a.pcap", b"one")
        z.writestr("AIMchat1.pcapng", b"two")
        z.writestr("readme.txt", b"not a capture")

    out = extract_zip(archive, tmp_path / "out")

    assert sorted(p.name for p in out) == ["AIMchat1.pcapng", "aim_chat_3a.pcap"]


def test_extract_zip_flattens_subfolders(tmp_path):
    archive = tmp_path / "nested.zip"
    with zipfile.ZipFile(archive, mode="w") as z:
        z.writestr("SMB/SMB-1.pcap", b"one")
        z.writestr("SMB/SMB-2.pcap", b"two")

    out = extract_zip(archive, tmp_path / "out")

    assert sorted(p.name for p in out) == ["SMB-1.pcap", "SMB-2.pcap"]


def test_extract_zip_no_matching_member_raises(tmp_path):
    archive = tmp_path / "empty.zip"
    with zipfile.ZipFile(archive, mode="w") as z:
        z.writestr("readme.txt", b"not a pcap")

    with pytest.raises(DownloadError):
        extract_zip(archive, tmp_path / "out")


def test_extract_zip_collision_raises(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "a.pcap").write_bytes(b"already here")

    archive = tmp_path / "one.zip"
    with zipfile.ZipFile(archive, mode="w") as z:
        z.writestr("a.pcap", b"new content")

    with pytest.raises(DownloadError):
        extract_zip(archive, dest)
