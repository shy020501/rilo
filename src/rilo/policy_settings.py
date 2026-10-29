"""Project-level model choices, independent of robot/environment configuration.

Only models with verified image input, Responses function calling, reasoning
levels, and local cost estimates are selectable in the web UI. Server-side
custom model configuration remains available through Settings.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


_MODELS = {
    "gpt-6-astra": {
        "label": "GPT-6 Astra",
        "reasoning_efforts": ("low", "medium", "high", "xhigh", "max"),
    },
    "gpt-6.1-sol": {
        "label": "GPT-6.1 Sol",
        "reasoning_efforts": ("low", "medium", "high", "xhigh", "max"),
    },
}


def policy_catalog() -> list[dict]:
    """Return public capabilities; no key, account lookup, or paid API call."""
    return [{"model": model, "label": values["label"],
             "reasoning_efforts": [*values["reasoning_efforts"], "auto"],
             "default_reasoning_effort": "medium",
             "pricing_source": f"https://developers.openai.com/api/docs/models/{model}"}
            for model, values in _MODELS.items()]


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str = Field(min_length=1, max_length=100)
    reasoning_effort: str = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def supported_combination(self):
        capabilities = _MODELS.get(self.model)
        if capabilities is None:
            raise ValueError("Choose a supported GPT model from the model catalog")
        if self.reasoning_effort not in (*capabilities["reasoning_efforts"], "auto"):
            raise ValueError(f"{self.model} does not support reasoning effort {self.reasoning_effort}")
        return self
