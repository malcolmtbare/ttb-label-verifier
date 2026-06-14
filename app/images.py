"""Normalize uploads so iPhone photos work everywhere.

iPhones save HEIC/HEIF, which none of the vision models or Azure OCR accept, and
which Chrome/Firefox can't render in an <img>. We convert HEIC to JPEG at the upload
boundary, so every downstream path — extraction, OCR, the browser preview, and the
locate/straighten overlay — is unchanged and sees an ordinary JPEG.

EXIF orientation is baked in during conversion so the bytes the model/OCR see match
what the browser shows; otherwise the highlight boxes would be offset on rotated
phone photos.
"""

from __future__ import annotations

import io

# ISO-BMFF brand codes that indicate HEIF/HEIC containers.
_HEIC_BRANDS = {b"heic", b"heix", b"heif", b"mif1", b"msf1", b"hevc", b"heim", b"heis", b"hevx"}


def _looks_heic(data: bytes, content_type: str | None, filename: str | None) -> bool:
    ct = (content_type or "").lower()
    if "heic" in ct or "heif" in ct:
        return True
    if (filename or "").lower().endswith((".heic", ".heif")):
        return True
    # Magic bytes: "....ftyp<brand>" near the start of the file.
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return data[8:12].lower() in _HEIC_BRANDS
    return False


def normalize(data: bytes, content_type: str | None, filename: str | None) -> tuple[bytes, str]:
    """Return (bytes, media_type). Converts HEIC/HEIF to JPEG; otherwise passthrough."""
    if not _looks_heic(data, content_type, filename):
        return data, (content_type or "image/jpeg")
    from PIL import Image, ImageOps
    from pillow_heif import register_heif_opener

    register_heif_opener()
    img = Image.open(io.BytesIO(data))
    try:
        img = ImageOps.exif_transpose(img)  # bake in orientation
    except Exception:
        pass
    img = img.convert("RGB")
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue(), "image/jpeg"