"""ISCX file-name label inference (spec 001, D3, plan T3).

These are exact expectations for hand-picked file names following the
documented ISCX naming convention (app[_activity][_index].pcap, optionally
vpn_-prefixed). See ``iscx_labels.py``'s module docstring for why this is a
heuristic rather than a verified mapping.
"""

from __future__ import annotations

import pytest

from adl_etc.data.iscx_labels import CATEGORIES, infer_label


@pytest.mark.parametrize(
    ("file_name", "expected_class"),
    [
        ("facebook_chat_4a.pcap", "chat_nonvpn"),
        ("facebook_audio1a.pcap", "voip_nonvpn"),
        ("facebookvideo1b.pcap", "voip_nonvpn"),
        ("skype_chat1a.pcap", "chat_nonvpn"),
        ("skype_audio1a.pcap", "voip_nonvpn"),
        ("skype_file1.pcap", "file_transfer_nonvpn"),
        ("hangouts_chat_4a.pcap", "chat_nonvpn"),
        ("hangout_audio1a.pcap", "voip_nonvpn"),
        ("vpn_youtube_A.pcap", "streaming_vpn"),
        ("youtube1.pcap", "streaming_nonvpn"),
        ("netflix1.pcap", "streaming_nonvpn"),
        ("spotify1.pcap", "streaming_nonvpn"),
        ("vimeo1.pcap", "streaming_nonvpn"),
        ("email1a.pcap", "mail_nonvpn"),
        ("gmailchat1.pcap", "mail_nonvpn"),
        ("icq_chat_3a.pcap", "chat_nonvpn"),
        ("aim_chat_3a.pcap", "chat_nonvpn"),
        ("voipbuster1a.pcap", "voip_nonvpn"),
        ("vpn_voipbuster1a.pcap", "voip_vpn"),
        ("ftps_down_1a.pcap", "file_transfer_nonvpn"),
        ("sftp1.pcap", "file_transfer_nonvpn"),
        ("scp1.pcap", "file_transfer_nonvpn"),
        ("torrent01.pcap", "p2p_nonvpn"),
        ("bittorrent.pcap", "p2p_nonvpn"),
        ("firefox.pcap", "browsing_nonvpn"),
        ("chrome.pcap", "browsing_nonvpn"),
    ],
)
def test_infer_label_known_names(file_name, expected_class):
    label = infer_label(file_name)
    assert label.class_name == expected_class
    assert label.confidence == "heuristic"


def test_infer_label_unresolved_activity_keeps_app():
    """Skype/Facebook/Hangouts with no recognised activity keyword: the app
    is still known, but the category is honestly reported as unresolved
    rather than guessed."""
    label = infer_label("skype_somethingweird.pcap")
    assert label.app == "skype"
    assert label.category is None
    assert label.class_name is None


def test_infer_label_unknown_name():
    label = infer_label("totally_unrecognised_capture.pcap")
    assert label.app is None
    assert label.category is None
    assert label.class_name is None
    assert label.condition == "nonvpn"  # still determinable, independent of app


def test_infer_label_case_insensitive():
    a = infer_label("VPN_YouTube_A.PCAP")
    b = infer_label("vpn_youtube_a.pcap")
    assert a.class_name == b.class_name == "streaming_vpn"


def test_categories_match_spec_count():
    """Spec 001, D3: '14 (7 categories x VPN/non-VPN)'."""
    assert len(CATEGORIES) == 7
