"""ISCX VPN-nonVPN 2016 acquisition (spec 001, D3): PCAP-pipeline validation
and category-level transfer corpus.

**Needs one manual step this project does not automate.** The dataset is
gated behind a registration form at
https://www.unb.ca/cic/datasets/vpn.html (name, email, institution, job
title, country); the form's ``action="insert.php"`` posts that information
before granting access. This is a real correction against spec 001, which
said "direct HTTP from cicresearch.ca" — the form was fetched and inspected
while building this module, and there is no unauthenticated direct link.
Automating a submission of someone else's personal identifying information
without their explicit, per-use consent is not something this project does
on the user's behalf; the user registers once, in their own browser, with
their own information, and gets back a URL to the real file listing.

Once that URL is in hand: the real post-registration listing (checked
2026-09-20 against the actual page, not assumed) serves **zip archives**,
not individual ``*.pcap`` files directly — this module's first version
assumed a flat directory-index page of pcaps, matching USTC's mirror, and
was wrong. The real ``/PCAPs`` folder lists ``VPN-PCAPs-01/02.zip`` and
``NonVPN-PCAPs-01/02/03.zip``; each zip holds its member pcaps flat, no
subfolders (confirmed against a real 640 MB archive, ``VPN-PCAPS-01.zip``,
14 members). This module discovers every ``*.zip`` link on the page,
downloads each with resume/retry (exactly like
:mod:`adl_etc.data.ustc_download`), and extracts it with
:func:`adl_etc.data.download.extract_zip` before labelling the extracted
pcaps by file name, the same as before.
"""

from __future__ import annotations

import csv
import fnmatch
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urljoin

from adl_etc.data import manifest as M
from adl_etc.data.download import discover_links, download_file, extract_zip, fetch_text
from adl_etc.data.iscx_labels import infer_label
from adl_etc.utils.provenance import file_sha256

REGISTRATION_URL = "https://www.unb.ca/cic/datasets/vpn.html"
LABELS_COLUMNS = ("file_name", "class_name", "category", "condition", "label_confidence")


class RegistrationRequired(RuntimeError):
    """Raised when neither --base-url nor --files-from was given."""


def discover_files(base_url: str) -> list[str]:
    """GETs ``base_url`` and returns every ``*.zip`` link found on it,
    resolved to absolute URLs against ``base_url`` -- the real post-
    registration listing serves zip archives (``VPN-PCAPs-01.zip`` and
    similar), not individual pcaps directly (checked against a real page,
    see the module docstring)."""
    html = fetch_text(base_url)
    return [urljoin(base_url, link) for link in discover_links(html, suffix=".zip")]


def run(
    *,
    base_url: str | None,
    files_from: Path | None,
    out_dir: Path,
    manifest_path: Path = M.DEFAULT_MANIFEST_PATH,
    files_glob: str | None = None,
    dry_run: bool = False,
    log: Callable[[str], None] = print,
) -> int:
    """Returns the number of pcap files present in ``out_dir/pcap`` on exit
    (0 on a dry run). Raises :class:`RegistrationRequired` if neither
    ``base_url`` nor ``files_from`` was given.

    ``files_glob`` matches against each discovered *archive* name (e.g.
    ``VPN-*.zip``), not individual pcap names -- a pcap's own class can only
    be known once its containing zip is downloaded and opened, so per-class
    subsetting happens later, at export time (``export_pcap.py --files``),
    same as it already does for D4.
    """
    if files_from is not None:
        urls = [
            line.strip()
            for line in files_from.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif base_url is not None:
        urls = discover_files(base_url)
    else:
        raise RegistrationRequired(
            "ISCX VPN-nonVPN 2016 needs one manual step first: register at "
            f"{REGISTRATION_URL} (this script does not submit your details "
            "for you), then pass the file-listing URL you're given as "
            "--base-url, or a local file of URLs as --files-from."
        )

    if files_glob:
        urls = [u for u in urls if fnmatch.fnmatch(Path(u).name, files_glob)]
    if not urls:
        raise RegistrationRequired(
            "no .zip (or .pcap) links found at --base-url (or --files-from was empty)"
        )

    log(f"ISCX VPN-nonVPN 2016: {len(urls)} archive(s)/file(s)")
    for u in urls:
        name = Path(u).name
        # Individual pcap names (e.g. from a hand-written --files-from list)
        # can be previewed now; a zip's members are unknown until extracted.
        if name.lower().endswith(".pcap"):
            label = infer_label(name)
            log(f"  {name:35s} -> {label.class_name or 'UNRESOLVED'}")
        else:
            log(f"  {name:35s} -> (zip: members classified after extraction)")

    if dry_run:
        return 0

    pcap_dir = out_dir / "pcap"
    pcap_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = out_dir / "_archives"

    labels: list[dict[str, str]] = []
    manifest_files: list[dict[str, object]] = []
    unresolved: list[str] = []
    for u in urls:
        name = Path(u).name
        if name.lower().endswith(".zip"):
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_dest = archive_dir / name
            log(f"downloading {name} ...")
            download_file(u, archive_dest)
            log(f"extracting {name} ...")
            # The archive is kept (not deleted after extraction, unlike
            # USTC's .7z path): unlike GitHub, this URL comes from the
            # user's own gated, non-repeatable registration session, so
            # losing the raw zip would mean asking them to fetch it again.
            pcaps = extract_zip(archive_dest, pcap_dir)
        else:
            dest = pcap_dir / name
            log(f"downloading {name} ...")
            download_file(u, dest)
            pcaps = [dest]

        for p in pcaps:
            label = infer_label(p.name)
            if label.class_name is None:
                unresolved.append(p.name)
            labels.append(
                {
                    "file_name": p.name,
                    "class_name": label.class_name or "",
                    "category": label.category or "",
                    "condition": label.condition,
                    "label_confidence": label.confidence,
                }
            )
            manifest_files.append(
                {"name": p.name, "bytes": p.stat().st_size, "sha256": file_sha256(p)}
            )

    if unresolved:
        shown = ", ".join(unresolved[:10]) + (" ..." if len(unresolved) > 10 else "")
        log(
            f"warning: {len(unresolved)} file(s) could not be classified from their "
            f"name; recorded with an empty class_name in labels.csv: {shown}"
        )

    labels_path = out_dir / "labels.csv"
    with open(labels_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(LABELS_COLUMNS))
        writer.writeheader()
        writer.writerows(labels)

    M.register(
        "iscx-vpn-2016",
        source_url=base_url or str(files_from),
        files=manifest_files,
        extracted_to=str(pcap_dir),
        manifest_path=manifest_path,
    )
    log(f"done: {len(labels)} pcaps, labels at {labels_path} ({len(unresolved)} unresolved)")
    return len(labels)
