"""Unit tests for the deterministic comparison logic.

These cover the exact stakeholder scenarios from the interviews:
  - Dave: "STONE'S THROW" vs "Stone's Throw" must be a match.
  - Jenny: the warning must match the canonical text, "GOVERNMENT WARNING" must be
    all caps, and a title-case "Government Warning" must fail.
"""

from app import matching
from app.schema import Status
from app.warning import CANONICAL_WARNING, verify_warning


# --- Fuzzy matching (Dave) -------------------------------------------------

def test_brand_case_and_punctuation_is_a_match():
    r = matching.compare_fuzzy("brand_name", "Brand name", "Stone's Throw", "STONE'S THROW")
    assert r.status is Status.PASS

def test_brand_clear_mismatch_fails():
    r = matching.compare_fuzzy("brand_name", "Brand name", "Old Tom Distillery", "Acme Lager")
    assert r.status is Status.FAIL

def test_minor_ocr_typo_is_review_or_pass():
    r = matching.compare_fuzzy("brand_name", "Brand name", "Toppling Goliath", "Topplng Goliath")
    assert r.status in (Status.PASS, Status.REVIEW)

def test_missing_detected_fails():
    r = matching.compare_fuzzy("brand_name", "Brand name", "Old Tom", None)
    assert r.status is Status.FAIL


# --- ABV -------------------------------------------------------------------

def test_abv_match_across_formats():
    r = matching.compare_abv("45% Alc./Vol. (90 Proof)", "45% ALC/VOL")
    assert r.status is Status.PASS

def test_abv_mismatch():
    r = matching.compare_abv("40% Alc./Vol.", "45% Alc./Vol.")
    assert r.status is Status.FAIL


# --- Net contents ----------------------------------------------------------

def test_net_contents_unit_normalization():
    r = matching.compare_net_contents("750 mL", "0.75 L")
    assert r.status is Status.PASS


# --- Country of origin inference -------------------------------------------

def test_infers_us_from_city_state():
    assert matching.infer_us_country("Brewed & Canned by Half Acre Beer Co. Chicago, IL")

def test_infers_us_from_explicit_usa():
    assert matching.infer_us_country("Produced in the USA")

def test_does_not_infer_us_from_foreign_address():
    assert not matching.infer_us_country("Bordeaux, France")
    assert not matching.infer_us_country("Toronto, ON")        # Ontario is not a US state

def test_us_origin_from_city_in_marketing_copy():
    # A real U.S. city wrapped in marketing copy, no state code — should still infer U.S.
    assert matching.infer_us_country("SAN FRANCISCO BORN & BREWED")
    assert matching.infer_us_country("Distilled in Chicago")
    assert matching.infer_us_country("St. Louis, proudly brewed since 1850")

def test_us_origin_from_spelled_out_state():
    assert matching.infer_us_country("Brewed in Oregon")
    assert matching.infer_us_country("A Vermont creamery product")

def test_us_origin_does_not_overfire_on_ambiguous_names():
    # "London Dry Gin" is a style, not a U.S. origin; must not infer U.S.
    assert not matching.infer_us_country("London Dry Gin")
    assert not matching.infer_us_country("Hamburg, Germany")
    assert not matching.infer_us_country("Product of Dublin, Ireland")
    # "Georgia" is also a country and a wine origin — deliberately not inferred from the bare name
    assert not matching.infer_us_country("Saperavi from Georgia")

def test_country_usa_variants_match():
    r = matching.compare_country("USA", "United States")
    assert r.status is Status.PASS

def test_verify_infers_us_country_and_matches_application():
    from app.schema import ApplicationData, LabelExtraction
    from app.verifier import verify
    ext = LabelExtraction(is_alcohol_label=True, brand_name="Half Acre",
                          name_address="Half Acre Beer Co. Chicago, IL",
                          government_warning_text=CANONICAL_WARNING)
    rep = verify(ext, ApplicationData(country_of_origin="USA"), "test", 0)
    country = next(f for f in rep.fields if f.field == "country_of_origin")
    assert country.status is Status.PASS
    assert "inferred" in country.reason.lower()
    assert ext.country_of_origin == "United States"   # written back for the UI


# --- Government warning (Jenny) --------------------------------------------

def test_exact_warning_passes():
    r = verify_warning(CANONICAL_WARNING, bold_visual=True)
    assert r.status is Status.PASS

def test_warning_whitespace_differences_still_pass():
    spaced = CANONICAL_WARNING.replace(" (2)", "\n(2)")
    r = verify_warning(spaced, bold_visual=True)
    assert r.status is Status.PASS

def test_all_caps_body_is_compliant():
    # Regression: real label (Half Acre Daisy Cutter) prints the whole warning in
    # caps. Body case is not regulated, so this must PASS, not fail.
    r = verify_warning(CANONICAL_WARNING.upper(), bold_visual=True)
    assert r.status is Status.PASS

def test_all_caps_with_missing_commas_still_passes():
    # The exact Gemini extraction from the Half Acre can: all caps, commas dropped.
    half_acre = ("GOVERNMENT WARNING: (1) ACCORDING TO THE SURGEON GENERAL WOMEN SHOULD "
                 "NOT DRINK ALCOHOLIC BEVERAGES DURING PREGNANCY BECAUSE OF THE RISK OF "
                 "BIRTH DEFECTS. (2) CONSUMPTION OF ALCOHOLIC BEVERAGES IMPAIRS YOUR "
                 "ABILITY TO DRIVE A CAR OR OPERATE MACHINERY, AND MAY CAUSE HEALTH PROBLEMS.")
    r = verify_warning(half_acre, bold_visual=True)
    assert r.status is Status.PASS

def test_title_case_prefix_fails():
    # Jenny's actual catch: the PREFIX in title case is a real violation.
    bad = CANONICAL_WARNING.replace("GOVERNMENT WARNING:", "Government Warning:")
    r = verify_warning(bad, bold_visual=True)
    assert r.status is Status.FAIL
    assert "capital" in r.reason.lower()

def test_missing_warning_fails():
    r = verify_warning(None, bold_visual=None)
    assert r.status is Status.FAIL

def test_reworded_warning_fails_with_diff():
    # Wrong WORDS must still fail, regardless of case.
    bad = CANONICAL_WARNING.replace("birth defects", "health issues").upper()
    r = verify_warning(bad, bold_visual=True)
    assert r.status is Status.FAIL
    assert r.score is not None and r.score < 1.0

def test_not_bold_downgrades_to_review():
    r = verify_warning(CANONICAL_WARNING, bold_visual=False)
    assert r.status is Status.REVIEW


# --- Multiple-SKU handling (graceful edge case) ----------------------------

def test_multiple_products_flagged_not_fabricated():
    """A photo with several distinct SKUs is flagged, with a clear message, rather
    than silently producing one blended verdict."""
    from app.schema import LabelExtraction, ApplicationData
    from app.verifier import verify
    ex = LabelExtraction(is_alcohol_label=True, multiple_products=True, product_count=4,
                         brand_name="Koval", class_type="Bourbon Whiskey")
    rep = verify(ex, ApplicationData(), "test", 0)
    assert rep.multiple_products is True
    assert rep.product_count == 4
    assert rep.message and "multiple products" in rep.message.lower()

def test_single_product_not_flagged():
    from app.schema import LabelExtraction, ApplicationData
    from app.verifier import verify
    rep = verify(LabelExtraction(is_alcohol_label=True, brand_name="Koval"),
                 ApplicationData(), "test", 0)
    assert rep.multiple_products is False
    assert rep.message is None