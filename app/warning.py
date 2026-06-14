"""Government health warning verification (Jenny's requirement).

Unlike the other fields, the warning is NOT compared against the application
form. It is checked against the fixed text mandated by the Alcoholic Beverage
Labeling Act of 1988 (27 CFR part 16). Source: TTB, ttb.gov.

What the regulation actually requires (and what we therefore check):
  1. WORDING: the statement must contain the exact required words. Compared
     case- and punctuation-insensitively, because the body's letter-case is NOT
     regulated and OCR/vision routinely varies punctuation. (Real labels print
     the whole warning in ALL CAPS and are fully compliant.)
  2. PREFIX CAPS: the words "GOVERNMENT WARNING" must appear in capital letters.
     This is the case rule that *does* exist, and it is what catches a title-case
     "Government Warning". Derivable from the characters, so high-confidence.
  3. BOLD: "GOVERNMENT WARNING" must be bold. A vision model gives only a visual
     heuristic, not a font-weight measurement, so this is ADVISORY and never
     hard-fails on its own. Documented as a known PoC limitation.

History note: an earlier version compared the body case-sensitively and flagged a
compliant all-caps label (Half Acre Daisy Cutter) as a mismatch — a false positive.
This version enforces only what the rule governs.
"""

from __future__ import annotations

import difflib
import re

from .schema import FieldResult, Status

# The exact statement required on the label. Source of truth for the wording.
CANONICAL_WARNING = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not "
    "drink alcoholic beverages during pregnancy because of the risk of birth defects. "
    "(2) Consumption of alcoholic beverages impairs your ability to drive a car or "
    "operate machinery, and may cause health problems."
)

PREFIX_WORDS = "GOVERNMENT WARNING"   # must appear capitalized


def _ws(s: str) -> str:
    """Collapse whitespace / trim. 'Continuous paragraph' + OCR line breaks mean
    whitespace differences are not meaningful."""
    return re.sub(r"\s+", " ", s).strip()


def _words(s: str) -> str:
    """Case- and punctuation-insensitive token form, for comparing WORDING.
    Body letter-case and incidental punctuation are not compliance signals."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)     # drop punctuation
    return re.sub(r"\s+", " ", s).strip()


def verify_warning(extracted: str | None, bold_visual: bool | None) -> FieldResult:
    label = "Government warning"

    if not extracted or not extracted.strip():
        return FieldResult(
            field="government_warning", label=label,
            expected=CANONICAL_WARNING, detected=None, status=Status.FAIL,
            reason="No government warning statement detected. It is mandatory on "
                   "all alcohol beverages.",
            rule="strict-canonical",
        )

    detected = _ws(extracted)
    prefix_present = detected.lower().startswith(PREFIX_WORDS.lower())
    prefix_caps_ok = detected.startswith(PREFIX_WORDS)        # all-caps satisfies this
    wording_ok = _words(detected) == _words(CANONICAL_WARNING)

    # --- compliant path -----------------------------------------------------
    if wording_ok and prefix_caps_ok:
        reason = "Required wording present and 'GOVERNMENT WARNING' is capitalized."
        if detected == detected.upper():
            reason += " (Label uses all-capitals, which is permitted.)"
        status = Status.PASS
        if bold_visual is False:
            status = Status.REVIEW
            reason += " Advisory: 'GOVERNMENT WARNING' may not appear bold — confirm by eye " \
                      "(bold detection is a known limitation)."
        return FieldResult(
            field="government_warning", label=label,
            expected=CANONICAL_WARNING, detected=extracted,
            status=status, reason=reason, score=1.0, rule="strict-canonical",
        )

    # --- failure path: say exactly what is wrong ----------------------------
    problems: list[str] = []
    if not prefix_present:
        problems.append("the statement does not begin with 'GOVERNMENT WARNING'")
    elif not prefix_caps_ok:
        problems.append("'GOVERNMENT WARNING' is not in capital letters")
    if not wording_ok:
        problems.append(_first_word_difference(_words(detected), _words(CANONICAL_WARNING)))
    if bold_visual is False:
        problems.append("'GOVERNMENT WARNING' may not be bold (advisory)")

    ratio = difflib.SequenceMatcher(None, _words(detected), _words(CANONICAL_WARNING)).ratio()
    return FieldResult(
        field="government_warning", label=label,
        expected=CANONICAL_WARNING, detected=extracted, status=Status.FAIL,
        reason="Warning does not meet requirements: " + "; ".join(problems) + ".",
        score=round(ratio, 3), rule="strict-canonical",
    )


def _first_word_difference(detected_words: str, canonical_words: str) -> str:
    """Point at the first real wording divergence (comparison is already
    case/punctuation-normalized, so this reflects words, not formatting)."""
    sm = difflib.SequenceMatcher(None, canonical_words, detected_words)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        exp = canonical_words[i1:i2].strip() or "(nothing)"
        got = detected_words[j1:j2].strip() or "(nothing)"
        exp = (exp[:40] + "…") if len(exp) > 40 else exp
        got = (got[:40] + "…") if len(got) > 40 else got
        return f"wording differs — expected “{exp}” but found “{got}”"
    return "wording differs from the required text"
