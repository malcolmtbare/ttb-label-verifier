"""Field matching (Dave's requirement).

Matching is done in deterministic Python, NOT by the model. In a compliance
setting an agent who rejects a label needs a defensible, reproducible reason —
"the comparison rule found a difference at character N" is auditable; "the model
decided it didn't match" is not. The model extracts; this module decides.

- Brand / class / name-address: fuzzy (normalize case + punctuation, then compare;
  fall back to a similarity ratio for graceful degradation). This treats Dave's
  "STONE'S THROW" vs "Stone's Throw" as a match while still flagging real diffs.
- Alcohol content: parse the numeric percentage and compare within a tolerance.
- Net contents: parse quantity + unit, normalize to a common unit, compare.
"""

from __future__ import annotations

import difflib
import re
from typing import Optional

from .schema import FieldResult, Status

# Default similarity threshold for a fuzzy "pass" when strings aren't identical
# after normalization. Between this and 1.0 we call it REVIEW, not a hard pass.
FUZZY_PASS = 0.92
FUZZY_REVIEW = 0.80


def normalize(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. This is the layer that
    makes 'STONE'S THROW' and "Stone's Throw" identical."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)      # drop punctuation/apostrophes
    s = re.sub(r"\s+", " ", s)
    return s


def compare_fuzzy(field: str, label: str, expected: Optional[str], detected: Optional[str]) -> FieldResult:
    if expected is None and detected is None:
        return _na(field, label, expected, detected, "Not present on form or label.")
    if not detected:
        return FieldResult(field=field, label=label, expected=expected, detected=detected,
                           status=Status.FAIL, reason="Expected on the label but not detected.",
                           rule="fuzzy")
    if not expected:
        return FieldResult(field=field, label=label, expected=expected, detected=detected,
                           status=Status.REVIEW,
                           reason="Detected on the label but no value on the application to compare against.",
                           rule="fuzzy")

    if normalize(expected) == normalize(detected):
        return FieldResult(field=field, label=label, expected=expected, detected=detected,
                           status=Status.PASS, reason="Match (ignoring case and punctuation).",
                           score=1.0, rule="fuzzy")

    ratio = difflib.SequenceMatcher(None, normalize(expected), normalize(detected)).ratio()
    if ratio >= FUZZY_PASS:
        return FieldResult(field=field, label=label, expected=expected, detected=detected,
                           status=Status.PASS,
                           reason=f"Close match ({ratio:.0%} similar) — minor surface differences only.",
                           score=round(ratio, 3), rule="fuzzy")
    if ratio >= FUZZY_REVIEW:
        return FieldResult(field=field, label=label, expected=expected, detected=detected,
                           status=Status.REVIEW,
                           reason=f"Partial match ({ratio:.0%} similar) — needs a human to confirm.",
                           score=round(ratio, 3), rule="fuzzy")
    return FieldResult(field=field, label=label, expected=expected, detected=detected,
                       status=Status.FAIL,
                       reason=f"Does not match ({ratio:.0%} similar).",
                       score=round(ratio, 3), rule="fuzzy")


_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def compare_abv(expected: Optional[str], detected: Optional[str], tolerance: float = 0.0) -> FieldResult:
    field, label = "alcohol_content", "Alcohol content"
    if expected is None and detected is None:
        return _na(field, label, expected, detected, "Not present on form or label.")
    e, d = _parse_pct(expected), _parse_pct(detected)
    if d is None:
        return FieldResult(field=field, label=label, expected=expected, detected=detected,
                           status=Status.FAIL, reason="No alcohol percentage detected on the label.",
                           rule="numeric")
    if e is None:
        return FieldResult(field=field, label=label, expected=expected, detected=detected,
                           status=Status.REVIEW, reason="No percentage on the application to compare against.",
                           rule="numeric")
    if abs(e - d) <= tolerance:
        return FieldResult(field=field, label=label, expected=expected, detected=detected,
                           status=Status.PASS, reason=f"ABV matches ({d:g}%).", score=1.0, rule="numeric")
    return FieldResult(field=field, label=label, expected=expected, detected=detected,
                       status=Status.FAIL,
                       reason=f"ABV differs: application says {e:g}%, label shows {d:g}%.",
                       rule="numeric")


def _parse_pct(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    m = _PCT.search(s)
    return float(m.group(1)) if m else None


_QTY = re.compile(r"(\d+(?:\.\d+)?)\s*(ml|milliliter|millilitre|l|liter|litre|fl\.?\s*oz|oz)", re.I)
_TO_ML = {"ml": 1.0, "milliliter": 1.0, "millilitre": 1.0,
          "l": 1000.0, "liter": 1000.0, "litre": 1000.0,
          "floz": 29.5735, "oz": 29.5735}


def compare_net_contents(expected: Optional[str], detected: Optional[str], tolerance_ml: float = 1.0) -> FieldResult:
    field, label = "net_contents", "Net contents"
    if expected is None and detected is None:
        return _na(field, label, expected, detected, "Not present on form or label.")
    e, d = _parse_volume(expected), _parse_volume(detected)
    if e is not None and d is not None:
        if abs(e - d) <= tolerance_ml:
            return FieldResult(field=field, label=label, expected=expected, detected=detected,
                               status=Status.PASS, reason="Volume matches.", score=1.0, rule="volume")
        return FieldResult(field=field, label=label, expected=expected, detected=detected,
                           status=Status.FAIL,
                           reason=f"Volume differs: form ≈ {e:g} mL, label ≈ {d:g} mL.", rule="volume")
    # Fall back to fuzzy text comparison if we can't parse a unit.
    return compare_fuzzy(field, label, expected, detected)


def _parse_volume(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    m = _QTY.search(s)
    if not m:
        return None
    qty = float(m.group(1))
    unit = re.sub(r"[.\s]", "", m.group(2).lower())
    return qty * _TO_ML.get(unit, 0) or None


def _na(field, label, expected, detected, reason) -> FieldResult:
    return FieldResult(field=field, label=label, expected=expected, detected=detected,
                       status=Status.NA, reason=reason, rule="n/a")


# --- Country of origin inference -------------------------------------------
# Country of origin is required for imports. A U.S. address on the label means
# the product is domestic, so we can infer the country without it being printed.
# Done in deterministic code (not the model) so the derivation is auditable.

US_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC",
}
# "City, ST" — comma then exactly two uppercase letters at a boundary. The comma
# requirement avoids matching words like "in"/"or" or a "Co." suffix.
_STATE_RE = re.compile(r",\s*([A-Z]{2})\b")
_US_WORDS_RE = re.compile(r"\b(u\.?\s?s\.?\s?a\.?|u\.?\s?s\.?|united states(?: of america)?)\b", re.I)
_US_CANON = {"usa", "us", "united states", "united states of america", "america"}


def infer_us_country(name_address: Optional[str]) -> bool:
    """True if the address indicates a U.S. location (a state code in a
    'City, ST' pattern, or an explicit USA / United States mention)."""
    if not name_address:
        return False
    if _US_WORDS_RE.search(name_address):
        return True
    return any(m.group(1) in US_STATES for m in _STATE_RE.finditer(name_address))


def canonical_country(s: Optional[str]) -> str:
    if not s:
        return ""
    t = re.sub(r"\s+", " ", s.strip().lower()).replace(".", "").strip()
    return "united states" if t in _US_CANON else t


def compare_country(expected: Optional[str], detected: Optional[str], inferred: bool = False) -> FieldResult:
    field, label = "country_of_origin", "Country of origin"
    note = " (inferred from U.S. address)" if inferred else ""
    if not expected and not detected:
        return _na(field, label, expected, detected, "Not present on form or label.")
    if not detected:
        return FieldResult(field=field, label=label, expected=expected, detected=detected,
                           status=Status.FAIL, reason="Expected on the label but not detected.", rule="country")
    if not expected:
        return FieldResult(field=field, label=label, expected=expected, detected=detected,
                           status=Status.REVIEW,
                           reason=f"Detected{note} but no application value to compare against.", rule="country")
    if canonical_country(expected) == canonical_country(detected):
        return FieldResult(field=field, label=label, expected=expected, detected=detected,
                           status=Status.PASS, reason=f"Country matches{note}.", score=1.0, rule="country")
    return FieldResult(field=field, label=label, expected=expected, detected=detected,
                       status=Status.FAIL,
                       reason=f"Country differs: application says '{expected}', label shows '{detected}'.", rule="country")