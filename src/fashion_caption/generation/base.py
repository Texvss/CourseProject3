from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol

from PIL import Image

from fashion_caption.prompts import PromptConfig


@dataclass
class GenerationResult:
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> Dict[str, Any]:
        return {"text": self.text, "meta": self.meta}


class Generator(Protocol):
    model_id: str
    instruction_capable: bool

    def generate(
        self,
        image: Image.Image,
        prompt_config: PromptConfig,
        params: Dict[str, Any],
    ) -> GenerationResult:
        ...
