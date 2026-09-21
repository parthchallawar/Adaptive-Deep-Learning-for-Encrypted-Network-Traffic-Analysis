"""Generic download and extraction primitives (spec 001).

Shared by every dataset downloader (``scripts/download_iscx.py``,
``scripts/download_ustc.py``): resumable HTTP with retry-with-backoff, GitHub
directory listing via the contents API, and single-archive 7z/zip
extraction. Dataset-specific knowledge (which files, what class each one
is) lives in the scripts that call these functions, not here.

Only the standard library's ``urllib`` is used for HTTP: the resume/retry
logic this project needs is a few dozen lines, not a reason to add
``requests`` as a dependency.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import py7zr

USER_AGENT = "adl-etc-downloader/1.0 (+https://github.com/)"


class DownloadError(RuntimeError):
    """Raised when a download exhausts its retries or hits a non-retriable
    HTTP error (4xx: the URL or request is wrong, retrying won't help)."""


@dataclass(slots=True)
class DownloadResult:
    path: Path
    bytes: int
    resumed: bool
    attempts: int


def download_file(
    url: str,
    dest: str | Path,
    *,
    expected_bytes: int | None = None,
    max_retries: int = 5,
    chunk_size: int = 1 << 20,
    timeout: float = 30.0,
    backoff_base: float = 1.0,
    sleep: Callable[[float], None] | None = None,
) -> DownloadResult:
    """Download ``url`` to ``dest``, resuming a partial file if one exists.

    Retries with exponential backoff (capped at 30s) on 5xx responses and
    connection-level errors. A 4xx response is not retried: the request
    itself is wrong (bad URL, moved file, forbidden), and retrying would just
    waste time and hide the real problem. If ``dest`` already exists and is
    at least ``expected_bytes`` long, returns immediately without a request.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sleep_fn = sleep or time.sleep
    resumed_any = False
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        existing = dest.stat().st_size if dest.exists() else 0
        if expected_bytes is not None and existing >= expected_bytes:
            return DownloadResult(dest, existing, resumed_any, attempt - 1)

        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        if existing:
            req.add_header("Range", f"bytes={existing}-")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                resumed = existing > 0 and resp.status == 206
                if existing and not resumed:
                    existing = 0  # server ignored Range; restart from scratch
                resumed_any = resumed_any or resumed
                mode = "ab" if resumed else "wb"
                written = existing
                with open(dest, mode) as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        written += len(chunk)
            return DownloadResult(dest, written, resumed_any, attempt)
        except urllib.error.HTTPError as e:
            if e.code < 500:
                raise DownloadError(f"{url}: HTTP {e.code} (not retriable)") from e
            last_error = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last_error = e

        if attempt < max_retries:
            sleep_fn(min(backoff_base * (2 ** (attempt - 1)), 30.0))

    raise DownloadError(f"{url}: failed after {max_retries} attempts: {last_error}") from last_error


def fetch_text(url: str, *, timeout: float = 30.0) -> str:
    """A single non-resumable GET, for small text/JSON/HTML responses."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


@dataclass(slots=True)
class GithubFile:
    name: str
    bytes: int
    download_url: str


GITHUB_API_BASE = "https://api.github.com"


def list_github_dir(
    owner: str,
    repo: str,
    path: str,
    *,
    ref: str = "master",
    api_base: str = GITHUB_API_BASE,
) -> list[GithubFile]:
    """Lists files in one directory of a public GitHub repo via the contents
    API (no auth token needed for public repos at low request volume).
    Raises :class:`DownloadError` on a non-200 response (rate limit, repo
    renamed, path missing). ``api_base`` is overridable so tests can point
    this at a local fixture instead of the real GitHub API."""
    url = f"{api_base}/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    try:
        raw = fetch_text(url)
    except urllib.error.HTTPError as e:
        raise DownloadError(f"listing {owner}/{repo}/{path}: HTTP {e.code}") from e
    entries = json.loads(raw)
    return [
        GithubFile(name=e["name"], bytes=e["size"], download_url=e["download_url"])
        for e in entries
        if e["type"] == "file"
    ]


_HREF_RE = re.compile(r'href=["\']([^"\'>]+)["\']', re.IGNORECASE)


def discover_links(html: str, *, suffix: str) -> list[str]:
    """Extracts every ``href`` ending in ``suffix`` (case-insensitive) from an
    HTML directory listing. Used for hosts that serve a plain Apache/nginx
    autoindex rather than an API; returns relative or absolute URLs exactly
    as written in the page, unresolved against a base URL."""
    return [h for h in _HREF_RE.findall(html) if h.lower().endswith(suffix.lower())]


def extract_7z(
    archive_path: str | Path, dest_dir: str | Path, *, suffix: str = ".pcap"
) -> list[Path]:
    """Extracts every ``suffix`` member of a 7z archive into ``dest_dir``
    (flattened: subfolders inside the archive are not preserved, only the
    member's own file name). Returns the extracted paths, sorted.

    Archives in the wild are not uniform even within one dataset mirror: some
    hold a single top-level file named after the archive, others hold a
    subfolder with several numbered files (both patterns are real, observed
    in USTC-TFC2016's own archives). This handles both without assuming
    either shape, and raises if extraction produces zero matching files
    rather than silently succeeding with nothing to show for it.
    """
    archive_path = Path(archive_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        names = [n for n in z.getnames() if n.lower().endswith(suffix.lower())]
        if not names:
            raise DownloadError(f"{archive_path}: no {suffix} member found")
        with tempfile.TemporaryDirectory() as tmp:
            z.extract(path=tmp, targets=names)
            out: list[Path] = []
            for name in names:
                extracted = Path(tmp) / name
                target = dest_dir / extracted.name
                if target.exists():
                    raise DownloadError(
                        f"{archive_path}: extracted name {target.name!r} collides with "
                        f"an existing file in {dest_dir}"
                    )
                shutil.move(str(extracted), target)
                out.append(target)
    return sorted(out)


def extract_zip(
    archive_path: str | Path,
    dest_dir: str | Path,
    *,
    suffix: str | tuple[str, ...] = (".pcap", ".pcapng"),
) -> list[Path]:
    """Extracts every member of a zip archive ending in ``suffix`` (one
    string, or a tuple to match several) into ``dest_dir`` (flattened, same
    contract as :func:`extract_7z`). Uses the standard library's
    ``zipfile`` -- no extra dependency, unlike ``.7z``.

    Confirmed against two real ISCX VPN-nonVPN 2016 archives, not assumed
    uniform with USTC's `.7z` shape or with each other:
    ``VPN-PCAPS-01.zip`` (640 MB, 14 flat top-level ``.pcap`` members, no
    subfolders) and ``NonVPN-PCAPs-01.zip`` (800 MB, 23 members, 11 of them
    ``.pcapng`` -- the newer capture format, which ``pcap_source.py``
    already parses via `dpkt.pcapng.Reader`, spec 002 -- mixed in with
    plain ``.pcap``). The default matches both suffixes for exactly this
    reason: an ISCX-specific single-suffix default would have silently
    dropped almost half of this real archive's files."""
    suffixes = (suffix,) if isinstance(suffix, str) else suffix
    archive_path = Path(archive_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as z:
        lower_suffixes = tuple(s.lower() for s in suffixes)
        names = [n for n in z.namelist() if n.lower().endswith(lower_suffixes)]
        if not names:
            raise DownloadError(f"{archive_path}: no {suffixes} member found")
        with tempfile.TemporaryDirectory() as tmp:
            out: list[Path] = []
            for name in names:
                extracted = Path(z.extract(name, path=tmp))
                target = dest_dir / Path(name).name
                if target.exists():
                    raise DownloadError(
                        f"{archive_path}: extracted name {target.name!r} collides with "
                        f"an existing file in {dest_dir}"
                    )
                shutil.move(str(extracted), target)
                out.append(target)
    return sorted(out)
