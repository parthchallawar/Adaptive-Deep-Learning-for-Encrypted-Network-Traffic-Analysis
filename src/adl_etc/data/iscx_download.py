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

Once that URL is in hand, this module is fully generic: it discovers every
``*.pcap`` link on the page (assuming a standard directory-index page — the
common case for this kind of academic mirror) and downloads them with
resume/retry, exactly like :mod:`adl_etc.data.ustc_download`.
"""

from __future__ import annotations

import csv
import fnmatch
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urljoin

from adl_etc.data import manifest as M
from adl_etc.data.download import discover_links, download_file, fetch_text
from adl_etc.data.iscx_labels import infer_label
from adl_etc.utils.provenance import file_sha256

REGISTRATION_URL = "https://www.unb.ca/cic/datasets/vpn.html"
LABELS_COLUMNS = ("file_name", "class_name", "category", "condition", "label_confidence")


class RegistrationRequired(RuntimeError):
    """Raised when neither --base-url nor --files-from was given."""


def discover_files(base_url: str) -> list[str]:
    """GETs ``base_url`` and returns every ``*.pcap`` link found on it,
    resolved to absolute URLs against ``base_url``."""
    html = fetch_text(base_url)
    return [urljoin(base_url, link) for link in discover_links(html, suffix=".pcap")]


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
    ``base_url`` nor ``files_from`` was given."""
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
        raise RegistrationRequired("no .pcap links found at --base-url (or --files-from was empty)")

    log(f"ISCX VPN-nonVPN 2016: {len(urls)} files")
    unresolved: list[str] = []
    for u in urls:
        name = Path(u).name
        label = infer_label(name)
        log(f"  {name:35s} -> {label.class_name or 'UNRESOLVED'}")
        if label.class_name is None:
            unresolved.append(name)
    if unresolved:
        shown = ", ".join(unresolved[:10]) + (" ..." if len(unresolved) > 10 else "")
        log(
            f"warning: {len(unresolved)} file(s) could not be classified from their "
            f"name; recorded with an empty class_name in labels.csv: {shown}"
        )

    if dry_run:
        return 0

    pcap_dir = out_dir / "pcap"
    pcap_dir.mkdir(parents=True, exist_ok=True)

    labels: list[dict[str, str]] = []
    manifest_files: list[dict[str, object]] = []
    for u in urls:
        name = Path(u).name
        dest = pcap_dir / name
        log(f"downloading {name} ...")
        download_file(u, dest)
        label = infer_label(name)
        labels.append(
            {
                "file_name": name,
                "class_name": label.class_name or "",
                "category": label.category or "",
                "condition": label.condition,
                "label_confidence": label.confidence,
            }
        )
        manifest_files.append(
            {"name": name, "bytes": dest.stat().st_size, "sha256": file_sha256(dest)}
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
