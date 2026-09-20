"""ISCX VPN-nonVPN 2016 orchestration (spec 001, D3, plan T3).

Runs discover_files()/run() against the local http_fixture standing in for
the post-registration file-listing page. RegistrationRequired (the "you
haven't registered yet" path) needs no server at all.
"""

from __future__ import annotations

import csv

import pytest

from adl_etc.data import manifest as M
from adl_etc.data.iscx_download import RegistrationRequired, discover_files, run

pytestmark = pytest.mark.integration


def test_run_without_base_url_or_files_from_raises():
    with pytest.raises(RegistrationRequired, match="unb.ca/cic/datasets/vpn.html"):
        run(base_url=None, files_from=None, out_dir=None, log=lambda _: None)  # type: ignore[arg-type]


def test_discover_files_finds_pcap_links(http_fixture):
    base_url, state = http_fixture
    state.files["/index.html"] = (
        b'<a href="youtube1.pcap">youtube1.pcap</a>'
        b'<a href="vpn_skype_chat1a.pcap">vpn_skype_chat1a.pcap</a>'
        b'<a href="readme.txt">readme</a>'
    ).replace(b"\n", b"")

    urls = discover_files(f"{base_url}/index.html")

    assert urls == [f"{base_url}/youtube1.pcap", f"{base_url}/vpn_skype_chat1a.pcap"]


def test_run_downloads_and_labels(http_fixture, tmp_path):
    base_url, state = http_fixture
    state.files["/index.html"] = b'<a href="youtube1.pcap">a</a><a href="unknownapp1.pcap">b</a>'
    state.files["/youtube1.pcap"] = b"yt-bytes"
    state.files["/unknownapp1.pcap"] = b"unk-bytes"

    out_dir = tmp_path / "out"
    manifest_path = tmp_path / "m.json"
    n = run(
        base_url=f"{base_url}/index.html",
        files_from=None,
        out_dir=out_dir,
        manifest_path=manifest_path,
        log=lambda _: None,
    )

    assert n == 2
    assert (out_dir / "pcap" / "youtube1.pcap").read_bytes() == b"yt-bytes"

    with open(out_dir / "labels.csv", newline="", encoding="utf-8") as fh:
        rows = {r["file_name"]: r for r in csv.DictReader(fh)}
    assert rows["youtube1.pcap"]["class_name"] == "streaming_nonvpn"
    assert rows["youtube1.pcap"]["label_confidence"] == "heuristic"
    assert rows["unknownapp1.pcap"]["class_name"] == ""  # honestly unresolved, not guessed

    assert M.verify("iscx-vpn-2016", manifest_path=manifest_path) is True


def test_run_files_glob_restricts_selection(http_fixture, tmp_path):
    base_url, state = http_fixture
    state.files["/index.html"] = b'<a href="youtube1.pcap">a</a><a href="netflix1.pcap">b</a>'
    state.files["/youtube1.pcap"] = b"a"
    state.files["/netflix1.pcap"] = b"b"

    out_dir = tmp_path / "out"
    n = run(
        base_url=f"{base_url}/index.html",
        files_from=None,
        out_dir=out_dir,
        manifest_path=tmp_path / "m.json",
        files_glob="*netflix*",
        log=lambda _: None,
    )

    assert n == 1
    assert not (out_dir / "pcap" / "youtube1.pcap").exists()
    assert (out_dir / "pcap" / "netflix1.pcap").exists()


def test_run_files_from_local_list(http_fixture, tmp_path):
    base_url, state = http_fixture
    state.files["/a/youtube1.pcap"] = b"yt"

    files_from = tmp_path / "urls.txt"
    files_from.write_text(f"{base_url}/a/youtube1.pcap\n\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    n = run(
        base_url=None,
        files_from=files_from,
        out_dir=out_dir,
        manifest_path=tmp_path / "m.json",
        log=lambda _: None,
    )

    assert n == 1
    assert (out_dir / "pcap" / "youtube1.pcap").read_bytes() == b"yt"


def test_run_dry_run_makes_no_download_requests(http_fixture, tmp_path):
    base_url, state = http_fixture
    state.files["/index.html"] = b'<a href="youtube1.pcap">a</a>'

    n = run(
        base_url=f"{base_url}/index.html",
        files_from=None,
        out_dir=tmp_path / "out",
        manifest_path=tmp_path / "m.json",
        dry_run=True,
        log=lambda _: None,
    )

    assert n == 0
    assert "/youtube1.pcap" not in state.request_log
    assert not (tmp_path / "out").exists()


def test_run_no_links_found_raises(http_fixture, tmp_path):
    base_url, state = http_fixture
    state.files["/index.html"] = b"<p>nothing here</p>"

    with pytest.raises(RegistrationRequired):
        run(
            base_url=f"{base_url}/index.html",
            files_from=None,
            out_dir=tmp_path / "out",
            manifest_path=tmp_path / "m.json",
            log=lambda _: None,
        )
