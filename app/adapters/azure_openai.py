"""Azure OpenAI adapter — the fully in-boundary option.

Uses the Azure OpenAI **v1 API**: the standard `OpenAI` client pointed at the
resource's `/openai/v1` endpoint. The v1 API drops the dated `api-version`
parameter entirely and authenticates with the API key directly.

Env:
  AZURE_OPENAI_ENDPOINT   e.g. https://your-resource.openai.azure.com
                          (with or without a trailing /openai/v1 — both work)
  AZURE_OPENAI_API_KEY    the key from the deployment's page
  AZURE_OPENAI_DEPLOYMENT the deployment NAME you assigned, e.g. "gpt-5.4-mini"
"""

from __future__ import annotations

import base64
import os

from ..schema import LabelExtraction
from .base import ExtractorAdapter, ModelInfo


class AzureOpenAIAdapter(ExtractorAdapter):
    info = ModelInfo(
        key="azure-openai",
        label="Azure OpenAI",
        boundary="in-boundary (Azure)",
        boundary_note="Runs inside your Azure region. No traffic leaves the boundary.",
    )

    @classmethod
    def is_configured(cls) -> bool:
        return all(os.getenv(v) for v in (
            "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT"))

    def __init__(self) -> None:
        from openai import OpenAI  # imported lazily so the app boots without the SDK
        endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
        if not endpoint.endswith("/openai/v1"):
            endpoint += "/openai/v1"          # tolerate either form in .env
        self._client = OpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            base_url=endpoint,
        )
        self._deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

    def extract(self, image_bytes: bytes, media_type: str) -> LabelExtraction:
        b64 = base64.b64encode(image_bytes).decode()
        resp = self._client.chat.completions.create(
            model=self._deployment,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract the label fields from this image."},
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                ]},
            ],
            # Non-strict json_schema: reliable structured output without the
            # strict-mode requirement that every field be required, which trips
            # up schemas with optional fields.
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "label_extraction",
                    "schema": LabelExtraction.model_json_schema(),
                },
            },
        )
        return LabelExtraction.model_validate_json(resp.choices[0].message.content)