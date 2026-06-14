"""Turn a `LabelExtraction` + `ApplicationData` into a `VerificationReport`.

This is model-agnostic on purpose: every adapter produces a `LabelExtraction`,
and this same function scores all of them, which is what makes the model picker
and the eval harness share one spine.
"""

from __future__ import annotations

from . import matching
from .schema import ApplicationData, FieldResult, LabelExtraction, Status, VerificationReport
from .warning import verify_warning


def _rollup(fields: list[FieldResult]) -> Status:
    statuses = {f.status for f in fields}
    if Status.FAIL in statuses:
        return Status.FAIL
    if Status.REVIEW in statuses:
        return Status.REVIEW
    return Status.PASS


def verify(extraction: LabelExtraction, expected: ApplicationData,
           model_used: str, latency_ms: int) -> VerificationReport:
    # Guardrail: not a label → don't fabricate a comparison.
    if not extraction.is_alcohol_label:
        return VerificationReport(
            is_alcohol_label=False, overall=Status.FAIL, model_used=model_used,
            latency_ms=latency_ms, fields=[], extraction=extraction,
            message="This image does not appear to be an alcohol beverage label. "
                    "Please upload a clearer photo of the label.",
        )

    # Country of origin: if the label doesn't state it but the address is clearly
    # U.S. (e.g. "Chicago, IL"), infer "United States". We write it back onto the
    # extraction so the UI and report reflect the derived value.
    inferred_us = False
    detected_country = extraction.country_of_origin
    if not detected_country and matching.infer_us_country(extraction.name_address):
        detected_country = "United States"
        extraction.country_of_origin = detected_country
        inferred_us = True

    fields = [
        matching.compare_fuzzy("brand_name", "Brand name", expected.brand_name, extraction.brand_name),
        matching.compare_fuzzy("class_type", "Class / type", expected.class_type, extraction.class_type),
        matching.compare_abv(expected.alcohol_content, extraction.alcohol_content),
        matching.compare_net_contents(expected.net_contents, extraction.net_contents),
        matching.compare_fuzzy("name_address", "Name & address", expected.name_address, extraction.name_address),
        matching.compare_country(expected.country_of_origin, detected_country, inferred_us),
        verify_warning(extraction.government_warning_text, extraction.government_warning_bold_visual),
    ]

    multi_msg = None
    if extraction.multiple_products:
        n = extraction.product_count or "several"
        multi_msg = (f"This image appears to contain multiple products ({n}). For each field "
                     "the tool shows the clearest value it could read, which may come from "
                     "different bottles — so the values below don't necessarily describe one "
                     "product. Upload one label per image for an accurate result.")

    return VerificationReport(
        is_alcohol_label=True, overall=_rollup(fields), model_used=model_used,
        latency_ms=latency_ms, fields=fields, extraction=extraction,
        multiple_products=extraction.multiple_products, product_count=extraction.product_count,
        message=multi_msg,
    )