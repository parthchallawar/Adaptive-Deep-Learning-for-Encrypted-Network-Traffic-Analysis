"""Spec 018's privacy rule: no address or port is ever stored.

A flow's key (its 5-tuple, for a PCAP) is written only as a salted SHA-1. The salt is random per
:class:`KeyHasher` (one per service process) and never written anywhere, so a stored hash cannot
be linked back to an address after the service stops.
"""

from __future__ import annotations

import hashlib
import secrets


class KeyHasher:
    def __init__(self, salt: bytes | None = None) -> None:
        self._salt = secrets.token_bytes(16) if salt is None else salt

    def __call__(self, key: str) -> str:
        return hashlib.sha1(self._salt + key.encode("utf-8"), usedforsecurity=False).hexdigest()
