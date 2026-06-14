"""Shared data contract.

Every model adapter returns a `LabelExtraction`. The verifier turns an
extraction plus the expected application data into a `VerificationReport`.
Keeping this in one place is what lets the model picker and the eval harness
run off the same spine: swap the adapter, the rest of the pipeline is identical.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --- What the model extracts from the image -------------------------------

class LabelExtraction(BaseModel):
    """Structured fields pulled from a single label image.

    `is_alcohol_label` is the guardrail for junk uploads (a photo of a dog,
    a blank page). When it is False, the rest of the fields are expected to be
    null and the verifier short-circuits with a clear message instead of
    pretending to compare nothing.
    """

    is_alcohol_label: bool = Field(
        description="True only if the image actually shows an alcohol beverage label."
    )
    multiple_products: bool = Field(
        default=False,
        description="True if the image shows more than one distinct product/label "
                    "(e.g. several different bottles), not a single label or a front/back pair.",
    )
    product_count: Optional[int] = Field(
        default=None, description="If multiple_products, how many distinct products are visible."
    )
    detected_beverage_type: Optional[str] = Field(
        default=None,
        description="Best guess: beer / wine / distilled spirits / seltzer / unknown.",
    )

    brand_name: Optional[str] = None
    class_type: Optional[str] = Field(
        default=None, description="Class/type designation, e.g. 'Kentucky Straight Bourbon Whiskey'."
    )
    alcohol_content: Optional[str] = Field(
        default=None, description="As printed, e.g. '45% Alc./Vol. (90 Proof)'."
    )
    net_contents: Optional[str] = Field(default=None, description="As printed, e.g. '750 mL'.")
    name_address: Optional[str] = Field(
        default=None, description="Name and address of the bottler / producer / importer."
    )
    country_of_origin: Optional[str] = Field(
        default=None, description="Only for imports; null if not present."
    )

    # The government warning is handled specially (see warning.py): it is checked
    # against the fixed legal text, not against the application form.
    government_warning_text: Optional[str] = Field(
        default=None,
        description="The full health warning statement exactly as it appears on the label, "
        "or null if no warning is present.",
    )
    government_warning_bold_visual: Optional[bool] = Field(
        default=None,
        description="ADVISORY visual judgment only: does the phrase 'GOVERNMENT WARNING' "
        "appear heavier/bolder than the surrounding text? This is a heuristic, not a "
        "font-weight measurement, and must be treated as low-confidence.",
    )

    notes: Optional[str] = Field(
        default=None, description="Anything ambiguous the reviewer should know (glare, crop, etc.)."
    )


# --- What the agent already has (the COLA application data) ----------------

class ApplicationData(BaseModel):
    """The 'expected' side. In production this comes from the COLA form; for the
    prototype it comes from a real TTB record or a hand-entered test case."""

    brand_name: Optional[str] = None
    class_type: Optional[str] = None
    alcohol_content: Optional[str] = None
    net_contents: Optional[str] = None
    name_address: Optional[str] = None
    country_of_origin: Optional[str] = None


# --- What verification produces -------------------------------------------

class Status(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    REVIEW = "review"   # matched loosely / low confidence — surface to a human
    NA = "not_applicable"


class FieldResult(BaseModel):
    field: str
    label: str                       # human label for the UI, e.g. "Brand name"
    expected: Optional[str]
    detected: Optional[str]
    status: Status
    reason: str                      # plain-language, audit-friendly explanation
    score: Optional[float] = None    # similarity 0..1 where relevant

    # Compliance rule applied, so the result is defensible after the fact.
    rule: str = Field(default="", description="Which comparison rule decided this, e.g. 'fuzzy', 'strict-canonical'.")


class VerificationReport(BaseModel):
    is_alcohol_label: bool
    overall: Status
    model_used: str
    latency_ms: int
    fields: list[FieldResult]
    extraction: LabelExtraction
    message: Optional[str] = None    # set when we short-circuit (junk image, etc.)
    multiple_products: bool = False  # true when the image shows several distinct SKUs
    product_count: Optional[int] = None