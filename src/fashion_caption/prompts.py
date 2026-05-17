from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


ECOMMERCE_PROMPT_ID = "ecommerce_v1"

ECOMMERCE_PROTOCOL = """
You are generating an e-commerce product description from a single product image.

Rules:
- Output 1-2 short sentences (max 25 words).
- Mention: product type + main color. Mention pattern only if clearly visible.
- Do NOT mention: brand, price, material, size, gender, background, people, emotions, marketing phrases.
- Focus on the product only.

Return ONLY the description text.
""".strip()

PROMPT_TEMPLATES: Dict[str, str] = {
    ECOMMERCE_PROMPT_ID: ECOMMERCE_PROTOCOL,
}


@dataclass(frozen=True)
class PromptConfig:
    prompt_id: str = ECOMMERCE_PROMPT_ID
    template: str = ECOMMERCE_PROTOCOL
    article_type: str = ""
    base_colour: str = ""
    user_prompt: str = ""

    @property
    def prompt(self) -> str:
        parts = [self.template.strip()]
        if self.article_type:
            parts.append(f"Known product type: {self.article_type}.")
        if self.base_colour:
            parts.append(f"Known main color: {self.base_colour}.")
        if self.user_prompt:
            parts.append(f"Additional user instruction: {self.user_prompt.strip()}")
        return "\n\n".join(part for part in parts if part)


def get_prompt_config(
    prompt_id: str = ECOMMERCE_PROMPT_ID,
    article_type: Optional[str] = None,
    base_colour: Optional[str] = None,
    user_prompt: Optional[str] = None,
) -> PromptConfig:
    template = PROMPT_TEMPLATES.get(prompt_id)
    if template is None:
        raise KeyError(f"Unknown prompt template: {prompt_id}")
    return PromptConfig(
        prompt_id=prompt_id,
        template=template,
        article_type=(article_type or "").strip(),
        base_colour=(base_colour or "").strip(),
        user_prompt=(user_prompt or "").strip(),
    )
