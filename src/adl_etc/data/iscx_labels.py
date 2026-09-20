"""ISCX VPN-nonVPN 2016 file-name label inference (spec 001, D3).

The dataset has no per-flow label column (spec 001: "none usable" time
structure) — the ground truth is entirely encoded in each pcap's *file name*
(e.g. ``facebook_chat_4a.pcap``, ``vpn_youtube_A.pcap``). This module infers
the 7-category x VPN/non-VPN class (14 total, matching spec 001's D3 row)
from that name.

**This is a heuristic, not a verified mapping.** ISCX VPN-nonVPN 2016 is
gated behind a registration form (name/email/institution) at
https://www.unb.ca/cic/datasets/vpn.html — this project has not automated
that registration (see ``scripts/download_iscx.py``'s module docstring for
why), so the exact real file names have not been seen. The category grouping
below comes from the dataset's own published description (spec 001) and from
the VPN-substring detection rule used by an independent third-party analysis
of this dataset (Mr-Pepe/iscx-analysis on GitHub); the activity-keyword rules
for apps that span multiple categories (Skype, Facebook, Hangouts each appear
in Chat *and* VoIP, and Skype also in File Transfer) are this project's own
best-effort reconstruction from the widely-documented ISCX naming convention
and have not been checked against a real file listing.

Every label this module produces is written with ``confidence="heuristic"``
in ``labels.csv``, and ``scripts/download_iscx.py`` prints a summary of any
file it could not classify at all, so a human reviews this once real files
are in hand rather than silently trusting it.
"""

from __future__ import annotations

from dataclasses import dataclass

# Ordered (name, category) rules. Order matters: more specific tokens
# (an app name *and* an activity keyword) are checked before generic,
# single-app rules, so "skype_chat1a" resolves via the Skype activity rules
# rather than falling through to a generic "skype -> ???" rule that doesn't
# exist (Skype has no single category — see the module docstring).
_FILE_TRANSFER_APPS = ("ftps", "sftp", "scp")
_STREAMING_APPS = ("youtube", "vimeo", "netflix", "spotify")
_MAIL_APPS = ("email", "gmail")
_CHAT_ONLY_APPS = ("icq", "aim")
_VOIP_ONLY_APPS = ("voipbuster",)
_BROWSING_APPS = ("firefox", "chrome")
_MULTI_CATEGORY_APPS = ("skype", "facebook", "hangout", "hangouts")

CATEGORIES = (
    "browsing",
    "chat",
    "streaming",
    "mail",
    "voip",
    "p2p",
    "file_transfer",
)


@dataclass(frozen=True, slots=True)
class IscxLabel:
    file_name: str
    app: str | None
    category: str | None
    condition: str  # "vpn" or "nonvpn"
    class_name: str | None  # f"{category}_{condition}", or None if unresolved
    confidence: str  # always "heuristic" here; see module docstring


def _condition(stem: str) -> str:
    return "vpn" if "vpn" in stem else "nonvpn"


def _multi_category_app(stem: str) -> tuple[str, str | None] | None:
    """Resolves Skype/Facebook/Hangouts, which span more than one category,
    using an activity keyword in the file name. Returns (app, category) or
    None if no activity keyword is recognised."""
    for app in _MULTI_CATEGORY_APPS:
        if app not in stem:
            continue
        canonical = "hangouts" if app == "hangout" else app
        if "chat" in stem:
            return canonical, "chat"
        if "file" in stem and canonical == "skype":
            return canonical, "file_transfer"
        if any(kw in stem for kw in ("audio", "voip", "call", "video")):
            return canonical, "voip"
        return canonical, None  # a known multi-category app, activity unresolved
    return None


def infer_label(file_name: str) -> IscxLabel:
    """Best-effort classification of one pcap file name. Never raises: an
    unrecognised name comes back with ``category=None``/``class_name=None``
    rather than guessing, so the caller can report it instead of mislabelling
    it silently.
    """
    stem = file_name.rsplit(".", 1)[0].lower()
    condition = _condition(stem)

    multi = _multi_category_app(stem)
    if multi is not None:
        app, category = multi
        class_name = f"{category}_{condition}" if category else None
        return IscxLabel(file_name, app, category, condition, class_name, "heuristic")

    rule_groups = (
        (_FILE_TRANSFER_APPS, "file_transfer"),
        (_STREAMING_APPS, "streaming"),
        (_MAIL_APPS, "mail"),
        (_CHAT_ONLY_APPS, "chat"),
        (_VOIP_ONLY_APPS, "voip"),
        (_BROWSING_APPS, "browsing"),
    )
    for apps, category in rule_groups:
        for app in apps:
            if app in stem:
                class_name = f"{category}_{condition}"
                return IscxLabel(file_name, app, category, condition, class_name, "heuristic")

    # "torrent"/"bittorrent"/"utorrent" all indicate P2P; kept separate from
    # the loop above since none of them is the literal app-name token.
    if any(kw in stem for kw in ("torrent", "p2p")):
        return IscxLabel(file_name, "torrent", "p2p", condition, f"p2p_{condition}", "heuristic")

    return IscxLabel(file_name, None, None, condition, None, "heuristic")
