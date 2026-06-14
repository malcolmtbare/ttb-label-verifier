"""The model abstraction.

Every model is one implementation of `ExtractorAdapter.extract`. The picker and
the eval harness only ever see this interface, so adding or swapping a model is a
local change. `boundary` records whether the model runs through Azure's control
plane or requires an external call — surfaced in the UI so the compliance posture
of each option is visible, not hidden.
"""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass

from ..prompts import EXTRACTION_PROMPT
from ..schema import LabelExtraction


@dataclass
class ModelInfo:
    key: str            # stable id used by the API/UI, e.g. "azure-openai"
    label: str          # display name, e.g. "Azure OpenAI (GPT-4o)"
    boundary: str       # "in-boundary (Azure)" | "azure-native access" | "external"
    boundary_note: str  # one-line explanation for the UI


class ExtractorAdapter(abc.ABC):
    info: ModelInfo

    @abc.abstractmethod
    def extract(self, image_bytes: bytes, media_type: str) -> LabelExtraction:
        """Run the shared prompt against the image and return structured fields."""

    @classmethod
    @abc.abstractmethod
    def is_configured(cls) -> bool:
        """True if the required env vars are present, so the UI only offers
        models that will actually work."""

    prompt = EXTRACTION_PROMPT


def available_adapters() -> list[ExtractorAdapter]:
    """Instantiate every adapter whose credentials are present."""
    from .azure_openai import AzureOpenAIAdapter
    from .claude_foundry import ClaudeFoundryAdapter
    from .gemini import GeminiAdapter

    out: list[ExtractorAdapter] = []
    for cls in (AzureOpenAIAdapter, ClaudeFoundryAdapter, GeminiAdapter):
        if cls.is_configured():
            out.append(cls())
    return out


def get_adapter(key: str) -> ExtractorAdapter:
    for a in available_adapters():
        if a.info.key == key:
            return a
    raise KeyError(f"Model '{key}' is not configured. Set its API credentials in .env.")
