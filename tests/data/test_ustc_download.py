"""USTC-TFC2016 orchestration (spec 001, D4, plan T3).

Runs the whole plan()/run() pipeline against the local http_fixture, standing
in for both the GitHub contents API and raw.githubusercontent.com.
"""

from __future__ import annotations

import csv
import io

import py7zr
import pytest

from adl_etc.data import manifest as M
from adl_etc.data.ustc_download import GITHUB_DIRS, plan, run

pytestmark = pytest.mark.integration


def _seed_github_dirs(state, base_url: str) -> None:
    state.json_routes["/repos/davidyslu/USTC-TFC2016/contents/Benign?ref=master"] = [
        {
            "name": "BitTorrent.pcap",
            "size": 4,
            "download_url": f"{base_url}/raw/Benign/BitTorrent.pcap",
            "type": "file",
        },
    ]
    state.json_routes["/repos/davidyslu/USTC-TFC2016/contents/Malware?ref=master"] = [
        {
            "name": "Shifu.7z",
            "size": 0,  # filled in by the caller once the archive bytes exist
            "download_url": f"{base_url}/raw/Malware/Shifu.7z",
            "type": "file",
        },
    ]


def _make_7z_bytes(inner_name: str, content: bytes) -> bytes:
    buf = io.BytesIO()
    with py7zr.SevenZipFile(buf, mode="w") as z:
        z.writestr(content, inner_name)
    return buf.getvalue()


def test_plan_lists_both_directories(http_fixture):
    base_url, state = http_fixture
    _seed_github_dirs(state, base_url)

    items = plan(api_base=base_url)

    assert {i.gh_file.name for i in items} == {"BitTorrent.pcap", "Shifu.7z"}
    by_name = {i.gh_file.name: i for i in items}
    assert by_name["BitTorrent.pcap"].category == GITHUB_DIRS["Benign"]
    assert by_name["BitTorrent.pcap"].class_name == "BitTorrent"
    assert by_name["Shifu.7z"].category == GITHUB_DIRS["Malware"]
    assert by_name["Shifu.7z"].class_name == "Shifu"


def test_plan_respects_files_glob(http_fixture):
    base_url, state = http_fixture
    _seed_github_dirs(state, base_url)

    items = plan(files_glob="Bit*", api_base=base_url)

    assert [i.gh_file.name for i in items] == ["BitTorrent.pcap"]


def test_run_dry_run_makes_no_download_requests(http_fixture, tmp_path):
    base_url, state = http_fixture
    _seed_github_dirs(state, base_url)

    n = run(
        out_dir=tmp_path / "out",
        manifest_path=tmp_path / "m.json",
        dry_run=True,
        api_base=base_url,
        log=lambda _: None,
    )

    assert n == 0
    assert not (tmp_path / "out").exists()


def test_run_downloads_pcap_and_extracts_7z(http_fixture, tmp_path):
    base_url, state = http_fixture
    _seed_github_dirs(state, base_url)
    state.files["/raw/Benign/BitTorrent.pcap"] = b"pcap"

    archive_bytes = _make_7z_bytes("Shifu.pcap", b"malware pcap bytes")
    state.files["/raw/Malware/Shifu.7z"] = archive_bytes
    # fix up the size the plan reports so download_file's expected_bytes check passes
    state.json_routes["/repos/davidyslu/USTC-TFC2016/contents/Malware?ref=master"][0]["size"] = len(
        archive_bytes
    )

    out_dir = tmp_path / "out"
    manifest_path = tmp_path / "m.json"
    n = run(out_dir=out_dir, manifest_path=manifest_path, api_base=base_url, log=lambda _: None)

    assert n == 2
    pcap_dir = out_dir / "pcap"
    assert (pcap_dir / "BitTorrent.pcap").read_bytes() == b"pcap"
    assert (pcap_dir / "Shifu.pcap").read_bytes() == b"malware pcap bytes"
    assert not (out_dir / "_archives").exists()  # cleaned up after extraction

    with open(out_dir / "labels.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_file = {r["file_name"]: r for r in rows}
    assert by_file["BitTorrent.pcap"]["class_name"] == "BitTorrent"
    assert by_file["BitTorrent.pcap"]["category"] == "benign"
    assert by_file["Shifu.pcap"]["class_name"] == "Shifu"
    assert by_file["Shifu.pcap"]["category"] == "malware"

    entry = M.require("ustc-tfc2016", manifest_path=manifest_path)
    assert len(entry["files"]) == 2
    assert M.verify("ustc-tfc2016", manifest_path=manifest_path) is True


def test_run_is_idempotent_for_extracted_archives(http_fixture, tmp_path):
    """A second run must not re-download an archive whose extracted output
    is already on disk."""
    base_url, state = http_fixture
    state.json_routes["/repos/davidyslu/USTC-TFC2016/contents/Benign?ref=master"] = []
    archive_bytes = _make_7z_bytes("Shifu.pcap", b"content")
    state.json_routes["/repos/davidyslu/USTC-TFC2016/contents/Malware?ref=master"] = [
        {
            "name": "Shifu.7z",
            "size": len(archive_bytes),
            "download_url": f"{base_url}/raw/Malware/Shifu.7z",
            "type": "file",
        },
    ]
    state.files["/raw/Malware/Shifu.7z"] = archive_bytes

    out_dir = tmp_path / "out"
    run(out_dir=out_dir, manifest_path=tmp_path / "m1.json", api_base=base_url, log=lambda _: None)
    assert state.request_log.count("/raw/Malware/Shifu.7z") == 1

    run(out_dir=out_dir, manifest_path=tmp_path / "m2.json", api_base=base_url, log=lambda _: None)
    assert state.request_log.count("/raw/Malware/Shifu.7z") == 1  # unchanged: no second download
