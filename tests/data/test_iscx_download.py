"""ISCX VPN-nonVPN 2016 orchestration (spec 001, D3, plan T3, revised T3.1).

Runs discover_files()/run() against the local http_fixture standing in for
the post-registration file-listing page. RegistrationRequired (the "you
haven't registered yet" path) needs no server at all.

The real post-registration listing serves **zip archives**, not individual
pcaps directly -- confirmed against a real 640 MB ISCX archive
(``VPN-PCAPS-01.zip``) after the user registered. The fixtures below build
small in-memory zips with :func:`_zip_bytes`, matching that real shape
(flat top-level members, no subfolders).
"""

from __future__ import annotations

import csv
import io
import zipfile

import pytest

from adl_etc.data import manifest as M
from adl_etc.data.iscx_download import RegistrationRequired, discover_files, run

pytestmark = pytest.mark.integration


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, content in members.items():
            z.writestr(name, content)
    return buf.getvalue()


def test_run_without_base_url_or_files_from_raises():
    with pytest.raises(RegistrationRequired, match="unb.ca/cic/datasets/vpn.html"):
        run(base_url=None, files_from=None, out_dir=None, log=lambda _: None)  # type: ignore[arg-type]


def test_discover_files_finds_zip_links(http_fixture):
    base_url, state = http_fixture
    state.files["/index.html"] = (
        b'<a href="VPN-PCAPs-01.zip">VPN-PCAPs-01.zip</a>'
        b'<a href="NonVPN-PCAPs-01.zip">NonVPN-PCAPs-01.zip</a>'
        b'<a href="readme.txt">readme</a>'
    ).replace(b"\n", b"")

    urls = discover_files(f"{base_url}/index.html")

    assert urls == [f"{base_url}/VPN-PCAPs-01.zip", f"{base_url}/NonVPN-PCAPs-01.zip"]


def test_run_downloads_extracts_and_labels(http_fixture, tmp_path):
    base_url, state = http_fixture
    state.files["/index.html"] = b'<a href="archive.zip">a</a>'
    state.files["/archive.zip"] = _zip_bytes(
        {"youtube1.pcap": b"yt-bytes", "unknownapp1.pcap": b"unk-bytes"}
    )

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
    # The archive itself is kept, not deleted (see run()'s docstring/comment).
    assert (out_dir / "_archives" / "archive.zip").exists()

    with open(out_dir / "labels.csv", newline="", encoding="utf-8") as fh:
        rows = {r["file_name"]: r for r in csv.DictReader(fh)}
    assert rows["youtube1.pcap"]["class_name"] == "streaming_nonvpn"
    assert rows["youtube1.pcap"]["label_confidence"] == "heuristic"
    assert rows["unknownapp1.pcap"]["class_name"] == ""  # honestly unresolved, not guessed

    assert M.verify("iscx-vpn-2016", manifest_path=manifest_path) is True


def test_discover_files_via_base_url_ignores_bare_pcap_links(http_fixture, tmp_path):
    # discover_files only looks for .zip links (the real listing's shape) --
    # a bare .pcap href is invisible to it, so run() raises the same "nothing
    # found" error as an empty page. Direct pcap support is for --files-from
    # (test_run_files_from_local_list below), a hand-written list, not page
    # discovery.
    base_url, state = http_fixture
    state.files["/index.html"] = b'<a href="youtube1.pcap">a</a>'
    state.files["/youtube1.pcap"] = b"yt-bytes"

    with pytest.raises(RegistrationRequired):
        run(
            base_url=f"{base_url}/index.html",
            files_from=None,
            out_dir=tmp_path / "out",
            manifest_path=tmp_path / "m.json",
            log=lambda _: None,
        )


def test_run_files_glob_restricts_selection_by_archive_name(http_fixture, tmp_path):
    base_url, state = http_fixture
    state.files["/index.html"] = (
        b'<a href="VPN-PCAPs-01.zip">a</a><a href="NonVPN-PCAPs-01.zip">b</a>'
    )
    state.files["/VPN-PCAPs-01.zip"] = _zip_bytes({"vpn_youtube1.pcap": b"a"})
    state.files["/NonVPN-PCAPs-01.zip"] = _zip_bytes({"netflix1.pcap": b"b"})

    out_dir = tmp_path / "out"
    n = run(
        base_url=f"{base_url}/index.html",
        files_from=None,
        out_dir=out_dir,
        manifest_path=tmp_path / "m.json",
        files_glob="NonVPN*",
        log=lambda _: None,
    )

    assert n == 1
    assert not (out_dir / "pcap" / "vpn_youtube1.pcap").exists()
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
    state.files["/index.html"] = b'<a href="archive.zip">a</a>'

    n = run(
        base_url=f"{base_url}/index.html",
        files_from=None,
        out_dir=tmp_path / "out",
        manifest_path=tmp_path / "m.json",
        dry_run=True,
        log=lambda _: None,
    )

    assert n == 0
    assert "/archive.zip" not in state.request_log
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
