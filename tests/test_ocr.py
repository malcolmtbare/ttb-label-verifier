"""Tests for the government-warning locator (app/ocr._locate_warning).

These pin the two defects seen on a rotated, small-print can warning:
  1. the located region must cover the whole warning block, not just the header;
  2. it must carry the real text angle so the zoom can straighten a rotated warning.

They use synthetic OCR words, so no Azure Vision call is needed.
"""

from app.ocr import _locate_warning, Word

WARNING = ("GOVERNMENT WARNING: (1) ACCORDING TO THE SURGEON GENERAL, WOMEN SHOULD NOT "
           "DRINK ALCOHOLIC BEVERAGES DURING PREGNANCY BECAUSE OF THE RISK OF BIRTH "
           "DEFECTS. (2) CONSUMPTION OF ALCOHOLIC BEVERAGES IMPAIRS YOUR ABILITY TO "
           "DRIVE A CAR OR OPERATE MACHINERY, AND MAY CAUSE HEALTH PROBLEMS.")

# Distinctive words that appear down the warning column, in reading order.
_BODY = ["ACCORDING", "SURGEON", "GENERAL", "WOMEN", "ALCOHOLIC", "BEVERAGES",
         "PREGNANCY", "BIRTH", "DEFECTS", "CONSUMPTION", "MACHINERY", "PROBLEMS"]


def _vertical_warning_words():
    """A vertical (rotated -90°) warning column on the right of the image, plus
    horizontal noise elsewhere (brand, recycle text, barcode digits)."""
    words = []
    # Header anchor, then body, stacked top->bottom in a narrow column at x~0.80.
    col = ["GOVERNMENT", "WARNING"] + _BODY
    y = 0.28
    for t in col:
        words.append(Word(t, (0.80, y, 0.04, 0.05), angle=-90.0))
        y += 0.052  # small gap so the cluster stays connected
    # Horizontal noise — different orientation, should be excluded by the angle filter.
    words.append(Word("DAISY", (0.10, 0.10, 0.10, 0.04), angle=0.0))
    words.append(Word("CUTTER", (0.22, 0.10, 0.10, 0.04), angle=0.0))
    words.append(Word("PLEASE", (0.78, 0.95, 0.08, 0.03), angle=0.0))
    words.append(Word("RECYCLE", (0.87, 0.95, 0.08, 0.03), angle=0.0))
    words.append(Word("50438", (0.82, 0.90, 0.06, 0.03), angle=0.0))
    return words


def test_rotated_warning_carries_real_angle():
    """Defect 2: a sideways warning must report a rotated angle (not 0), so the
    zoom straightens it."""
    reg = _locate_warning(_vertical_warning_words(), WARNING)
    assert reg is not None
    assert abs(reg["angle"]) > 45          # clearly rotated, not the old hardcoded 0.0


def test_rotated_warning_covers_whole_block_not_just_header():
    """Defect 1: the region must span the full warning column, not just the
    'GOVERNMENT WARNING' header."""
    reg = _locate_warning(_vertical_warning_words(), WARNING)
    assert reg is not None
    # The column runs from y~0.28 down past y~0.9; a header-only box would be ~0.05 tall.
    assert reg["box"][3] > 0.30
    # Most of the column's words should be captured.
    assert len(reg["boxes"]) >= 10


def test_warning_excludes_horizontal_label_text():
    """The brand/recycle/barcode run horizontally and must not be pulled into the
    warning region by the orientation filter."""
    reg = _locate_warning(_vertical_warning_words(), WARNING)
    assert reg is not None
    # Brand sits at x~0.10; the warning column at x~0.80. If the brand were included,
    # the union would stretch left across the image.
    assert reg["box"][0] > 0.5


def test_horizontal_warning_still_located():
    """No regression for a normal upright warning: anchor + token members, angle ~0."""
    words = [Word("GOVERNMENT", (0.10, 0.80, 0.12, 0.03), angle=0.0),
             Word("WARNING", (0.23, 0.80, 0.10, 0.03), angle=0.0)]
    x = 0.10
    for t in _BODY:
        words.append(Word(t, (x, 0.84, 0.07, 0.03), angle=0.0)); x += 0.072
    words.append(Word("DAISY", (0.10, 0.10, 0.10, 0.04), angle=0.0))  # far-away brand
    reg = _locate_warning(words, WARNING)
    assert reg is not None
    assert abs(reg["angle"]) < 20
    assert len(reg["boxes"]) >= 6


# --- Per-field rotation (not just the warning) -----------------------------

def test_any_rotated_field_gets_an_angle():
    """A non-warning field whose words are rotated (e.g. a vertical name/address)
    must come back with a region carrying that angle, so the zoom can straighten it."""
    import app.ocr as ocr
    # vertical name/address: each word rotated -90, stacked in a column
    words = [
        ocr.Word("MILLER", (0.10, 0.30, 0.04, 0.08), angle=-90.0),
        ocr.Word("BREWING", (0.10, 0.39, 0.04, 0.09), angle=-90.0),
        ocr.Word("CO", (0.10, 0.49, 0.04, 0.04), angle=-90.0),
        ocr.Word("GOLDEN", (0.10, 0.54, 0.04, 0.08), angle=-90.0),
    ]
    run = ocr._locate_word_run(words, "MILLER BREWING CO GOLDEN")
    assert len(run) == 4
    import statistics
    angle = round(statistics.median([w.angle for w in run]), 1)
    assert abs(angle) > 45      # rotated, so the UI will straighten it — not 0