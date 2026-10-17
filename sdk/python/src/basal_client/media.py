"""Inline media inputs. Helpers never fetch URLs or read files."""

from base64 import b64encode
from typing import Literal

from .types import MediaInput


def _media(kind: Literal["image", "audio"], data: bytes | str) -> MediaInput:
    if isinstance(data, bytes):
        data = b64encode(data).decode("ascii")
    if not isinstance(data, str) or not data.strip() or data[:8].lower().startswith(("http://", "https://")):
        raise ValueError("media data must be nonempty bytes, base64 or a data URL")
    return {"type": kind, "data": data}


def image(data: bytes | str) -> MediaInput:
    """An image encoded as bytes, base64 or a data URL; server validates the format."""
    return _media("image", data)


def audio(data: bytes | str) -> MediaInput:
    """Audio encoded as bytes, base64 or a data URL; server validates the format."""
    return _media("audio", data)
