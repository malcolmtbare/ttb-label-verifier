"""Gemini — the external benchmark.

Gemini is NOT hosted in Azure. From an Azure-hosted app this is an outbound call
to Google's API, which leaves the boundary and would be blocked by the agency's
firewall in production. It is included as a performance benchmark only, and is
labeled "external" in the UI so the comparison stays honest on the compliance axis.

A Google AI Studio API key is enough — no GCP project or Vertex setup required.
"""

from __future__ import annotations

import os

from ..schema import LabelExtraction
from .base import ExtractorAdapter, ModelInfo


class GeminiAdapter(ExtractorAdapter):
    info = ModelInfo(
        key="gemini",
        label="Gemini 3 Flash (external)",
        boundary="external",
        boundary_note="Outbound call to Google — leaves the Azure boundary. "
                      "Benchmark only; would be firewalled in production.",
    )

    @classmethod
    def is_configured(cls) -> bool:
        return bool(os.getenv("GEMINI_API_KEY"))

    def __init__(self) -> None:
        from google import genai  # lazy import
        self._genai = genai
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self._model = os.getenv("GEMINI_MODEL", "gemini-3-flash")

    def extract(self, image_bytes: bytes, media_type: str) -> LabelExtraction:
        from google.genai import types
        resp = self._client.models.generate_content(
            model=self._model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=media_type),
                self.prompt,
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": LabelExtraction,
                "temperature": 0,
            },
        )
        # google-genai can return a parsed Pydantic instance directly.
        if getattr(resp, "parsed", None) is not None:
            return resp.parsed
        return LabelExtraction.model_validate_json(resp.text)
