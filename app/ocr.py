"""Dedicated OCR for on-image field highlighting.

The extraction model gives us *what* each field says; it does not give reliable
pixel coordinates. A dedicated OCR engine (Azure AI Vision Read) does — it returns
every word with an accurate bounding polygon. We then align each extracted field
value to the run of OCR words that spells it, and hand the frontend normalized boxes
to highlight.

This is intentionally decoupled from extraction: the UI shows the model's fast result
immediately, then calls this asynchronously to light up the image a moment later. If
Vision isn't configured, `locate()` returns `configured=False` and the UI simply
doesn't show the locate affordance — nothing else changes.

Azure AI Vision is a first-party Azure service, so this stays inside the same
compliance boundary as the in-boundary extraction model (no new external egress).
"""

from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass

API_PATH = "/computervision/imageanalysis:analyze"
API_VERSION = "2024-02-01"
MATCH_THRESHOLD = 0.6  # min similarity to accept a word-run as a field's location


@dataclass
class Word:
    text: str
    box: tuple[float, float, float, float]  # normalized (x, y, w, h), 0..1
    angle: float = 0.0  # text angle in degrees, from the word's top edge


def _configured() -> tuple[str, str] | None:
    ep = (os.getenv("AZURE_VISION_ENDPOINT") or "").strip().rstrip("/")
    key = (os.getenv("AZURE_VISION_KEY") or "").strip()
    return (ep, key) if ep and key else None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _poly_to_box(poly: list[dict], w: int, h: int) -> tuple[float, float, float, float]:
    xs = [p.get("x", 0) for p in poly]
    ys = [p.get("y", 0) for p in poly]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return (x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h)


def _poly_angle(poly: list[dict]) -> float:
    """Angle of the word's top edge (point0 -> point1), in degrees. ~0 for upright
    horizontal text, ~90 for text rotated a quarter turn (common on cans)."""
    if len(poly) < 2:
        return 0.0
    import math
    dx = poly[1].get("x", 0) - poly[0].get("x", 0)
    dy = poly[1].get("y", 0) - poly[0].get("y", 0)
    return math.degrees(math.atan2(dy, dx))


def read_words(image_bytes: bytes, media_type: str) -> list[Word] | None:
    """Call Azure AI Vision Read; return normalized word boxes, or None if not configured."""
    cfg = _configured()
    if not cfg:
        return None
    endpoint, key = cfg
    import requests

    url = f"{endpoint}{API_PATH}"
    resp = requests.post(
        url,
        params={"api-version": API_VERSION, "features": "read"},
        headers={"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/octet-stream"},
        data=image_bytes,
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    meta = body.get("metadata", {}) or {}
    w, h = meta.get("width") or 1, meta.get("height") or 1
    words: list[Word] = []
    for block in (body.get("readResult", {}) or {}).get("blocks", []) or []:
        for line in block.get("lines", []) or []:
            for word in line.get("words", []) or []:
                poly = word.get("boundingPolygon") or []
                if word.get("text") and poly:
                    words.append(Word(word["text"], _poly_to_box(poly, w, h), _poly_angle(poly)))
    return words


def _best_run(words: list[Word], value: str) -> tuple[int, int, float] | None:
    """Find the contiguous run of OCR words best matching `value`.
    Returns (start_index, size, score) or None."""
    target = _norm(value)
    if not target or not words:
        return None
    wnorm = [_norm(w.text) for w in words]
    n = len(target.split())
    best, best_score = None, 0.0
    # try a few window sizes — OCR may split or merge tokens vs the model's value
    for size in {max(1, n - 1), n, n + 1}:
        for i in range(0, len(words) - size + 1):
            cand = " ".join(t for t in wnorm[i:i + size] if t).strip()
            if not cand:
                continue
            score = difflib.SequenceMatcher(None, cand, target).ratio()
            if score > best_score:
                best, best_score = (i, size), score
    return (best[0], best[1], best_score) if best else None


def _locate_value(words: list[Word], value: str) -> list[tuple[float, float, float, float]]:
    """Find the contiguous run of OCR words best matching `value`; return their boxes."""
    r = _best_run(words, value)
    if r and r[2] >= MATCH_THRESHOLD:
        i, size, _ = r
        return [words[j].box for j in range(i, i + size) if _norm(words[j].text)]
    return []


def _locate_region(words: list[Word], text: str) -> dict | None:
    """Locate a long passage (e.g. the government warning or a multi-line address) by
    collecting every OCR word that belongs to it and unioning them. Robust to the
    jumbled reading order Vision returns for rotated or multi-line text, where matching
    a long string in order fails. Returns the union box, dominant angle, member boxes."""
    import statistics as _st
    tokens = {t for t in _norm(text).split() if len(t) >= 3}
    if len(tokens) < 2 or not words:
        return None
    members = [w for w in words if _norm(w.text) in tokens]
    # require a reasonable fraction of the passage's distinctive words to be present
    if len(members) < max(2, len(tokens) // 3):
        return None
    xs0 = [w.box[0] for w in members]; ys0 = [w.box[1] for w in members]
    xs1 = [w.box[0] + w.box[2] for w in members]; ys1 = [w.box[1] + w.box[3] for w in members]
    union = [min(xs0), min(ys0), max(xs1) - min(xs0), max(ys1) - min(ys0)]
    angle = _st.median([w.angle for w in members])
    return {"box": union, "angle": round(angle, 1), "boxes": [list(w.box) for w in members]}


def _orient_close(a: float, b: float, tol: float) -> bool:
    """True if two text angles share an orientation (mod 180°, within tol degrees).
    Lets vertical warning text (~±90°) be told apart from horizontal label text (~0°)."""
    d = abs((a % 180) - (b % 180))
    return min(d, 180 - d) <= tol


def _near(a: Word, b: Word, gap: float) -> bool:
    """True if two word boxes are within `gap` (normalized units) on both axes."""
    ax0, ay0, aw, ah = a.box; ax1, ay1 = ax0 + aw, ay0 + ah
    bx0, by0, bw, bh = b.box; bx1, by1 = bx0 + bw, by0 + bh
    gx = max(0.0, bx0 - ax1, ax0 - bx1)
    gy = max(0.0, by0 - ay1, ay0 - by1)
    return gx <= gap and gy <= gap


def _grow(seed: list[Word], candidates: list[Word], gap: float) -> list[Word]:
    """Greedily grow a cluster outward from `seed`, adding any candidate word within
    `gap` of a word already in the cluster. Connected-component over near-by words."""
    cluster = list(seed)
    ids = {id(w) for w in cluster}
    changed = True
    while changed:
        changed = False
        for w in candidates:
            if id(w) in ids:
                continue
            if any(_near(w, c, gap) for c in cluster):
                cluster.append(w); ids.add(id(w)); changed = True
    return cluster


def _area(box: list[float]) -> float:
    return max(0.0, box[2]) * max(0.0, box[3])


def _union(ws: list[Word]) -> list[float]:
    xs0 = [w.box[0] for w in ws]; ys0 = [w.box[1] for w in ws]
    xs1 = [w.box[0] + w.box[2] for w in ws]; ys1 = [w.box[1] + w.box[3] for w in ws]
    return [min(xs0), min(ys0), max(xs1) - min(xs0), max(ys1) - min(ys0)]


def _locate_warning(words: list[Word], value: str) -> dict | None:
    """Locate the government warning by its reliable anchor ("GOVERNMENT WARNING"),
    then grow the region to cover the whole warning block.

    The body text is often tiny and — on cans — rotated a quarter turn, so matching the
    full passage word-for-word is unreliable. Instead we anchor on the header (which OCRs
    cleanly), take its angle, and union it with every word that shares the warning's
    orientation and is spatially connected to it. Because label text like the brand,
    "PLEASE RECYCLE", and the barcode run horizontally while a rotated warning runs
    vertically, the orientation filter cleanly separates the warning from the rest.
    Carries the real `angle` through so the UI can straighten the crop for reading."""
    import statistics as _st
    r = _best_run(words, "government warning")
    if not r or r[2] < MATCH_THRESHOLD:
        return None
    i, size, _ = r
    seed = [words[j] for j in range(i, i + size)]
    angle = _st.median([w.angle for w in seed])

    # words OCR'd cleanly enough to match the warning's own tokens
    tokens = {t for t in _norm(value).split() if len(t) >= 3}
    members = [w for w in words if _norm(w.text) in tokens]

    # for a rotated warning, grow across the block by orientation + proximity (the body
    # words are too garbled to token-match, but they share the header's angle)
    o = abs(angle) % 180
    rotated = not (o < 20 or o > 160)
    cluster: list[Word] = []
    if rotated:
        scale = _st.median([max(w.box[2], w.box[3]) for w in seed]) or 0.02
        aligned = [w for w in words if _orient_close(w.angle, angle, 22)]
        cluster = _grow(seed, aligned, gap=2.0 * scale)

    # de-dup seed ∪ members ∪ cluster
    chosen: list[Word] = []
    seen: set[int] = set()
    for w in seed + members + cluster:
        if id(w) not in seen:
            chosen.append(w); seen.add(id(w))

    # safety: if growth ran away (e.g. covers most of the image), back off to the
    # token members, then to the anchor alone, rather than highlight everything.
    box = _union(chosen)
    if _area(box) > 0.55:
        chosen = [w for w in seed + members if True] or seed
        box = _union(chosen)
    if _area(box) > 0.55:
        chosen, box = seed, _union(seed)

    return {"box": box, "angle": round(angle, 1), "boxes": [list(w.box) for w in chosen]}


def locate(image_bytes: bytes, media_type: str, fields: dict[str, str]) -> dict:
    """Read the image once and locate each provided field value on it.

    Short fields use a precise contiguous-run match; long multi-word fields (the
    government warning, the name & address) fall back to a membership union that
    tolerates multiple lines and rotation. The warning additionally returns a region
    with a text `angle` so the UI can straighten a rotated warning for reading.
    Returns {"configured": bool, "boxes": {field: [[x,y,w,h]]}, "regions": {field: {...}}}.
    """
    words = read_words(image_bytes, media_type)
    if words is None:
        return {"configured": False, "boxes": {}, "regions": {}}
    boxes: dict = {}
    regions: dict = {}
    for field, value in fields.items():
        value = value or ""
        if field == "government_warning":
            # Anchor on "GOVERNMENT WARNING" and grow the region (handles tiny, rotated
            # warning text where matching the full passage word-for-word fails).
            reg = _locate_warning(words, value)
            if not reg:
                # Last resort: token-membership union (works for clean horizontal text).
                reg = _locate_region(words, value)
            if reg:
                regions[field] = reg
                boxes[field] = reg["boxes"]
            continue
        located = _locate_value(words, value)
        # Long fields (e.g. a multi-line bottler address) often fail a strict in-order
        # match; fall back to the membership union so they still highlight.
        if not located and len(value.split()) >= 4:
            reg = _locate_region(words, value)
            if reg:
                located = [tuple(b) for b in reg["boxes"]]
        if located:
            boxes[field] = [list(b) for b in located]
    return {"configured": True, "boxes": boxes, "regions": regions}