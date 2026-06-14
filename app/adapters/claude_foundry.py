"""Claude via Microsoft Foundry — Azure-native access.

Credentials, billing, and governance flow through Azure's control plane (Foundry),
but note: Claude inference currently runs on Anthropic-hosted infrastructure rather
than physically inside an Azure region. That distinction is a footnote only under
strict FedRAMP data-residency rules — see README. For everything else this behaves
like a normal Azure-integrated model.

Structured output is done with tool-use: we define a single tool whose input schema
is our LabelExtraction, force the model to call it, and read the arguments. This is
the most reliable structured-output path for Claude.

NOTE: the exact base_url / auth headers for your Foundry deployment come from the
Foundry resource (Models > Claude > endpoint). They are read from env here so this
file never hardcodes a tenant.
"""

from __future__ import annotations

import base64
import os

from ..schema import LabelExtraction
from .base import ExtractorAdapter, ModelInfo

_TOOL = {
    "name": "record_label_fields",
    "description": "Record the structured fields extracted from the alcohol label.",
    "input_schema": LabelExtraction.model_json_schema(),
}


class ClaudeFoundryAdapter(ExtractorAdapter):
    info = ModelInfo(
        key="claude-foundry",
        label="Claude (Microsoft Foundry)",
        boundary="azure-native access",
        boundary_note="Reached through Azure/Foundry credentials & governance; "
                      "inference currently runs on Anthropic-hosted infra (FedRAMP footnote).",
    )

    @classmethod
    def is_configured(cls) -> bool:
        return all(os.getenv(v) for v in ("FOUNDRY_CLAUDE_ENDPOINT", "FOUNDRY_CLAUDE_API_KEY"))

    def __init__(self) -> None:
        from anthropic import Anthropic  # lazy import
        # The Anthropic SDK can target the Foundry endpoint via base_url + the
        # Azure-issued key. Exact header/auth scheme is per your Foundry resource.
        self._client = Anthropic(
            base_url=os.environ["FOUNDRY_CLAUDE_ENDPOINT"],
            api_key=os.environ["FOUNDRY_CLAUDE_API_KEY"],
        )
        self._model = os.getenv("FOUNDRY_CLAUDE_MODEL", "claude-haiku-4-5")

    def extract(self, image_bytes: bytes, media_type: str) -> LabelExtraction:
        b64 = base64.b64encode(image_bytes).decode()
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            tools=[_TOOL],
            tool_choice={"type": "tool", "name": "record_label_fields"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": self.prompt},
                ],
            }],
        )
        for block in resp.content:
            if block.type == "tool_use":
                return LabelExtraction.model_validate(block.input)
        raise RuntimeError("Claude did not return structured tool output.")
