SEED = 42
SANITY_N = 20
EVAL_N = 300

TOPWEAR_TYPES = [
    "Tshirts",
    "Shirts",
    "Kurtas",
    "Tops",
    "Sweatshirts",
    "Sweaters",
    "Jackets",
    "Kurtis",
    "Tunics",
    "Dupatta",
]

PROTOCOL_TEXT = """
Generation Protocol (v1.0)
Output format: 1-2 sentences, neutral descriptive tone.
Allowed: clothing type, main color, visible details (sleeves, collar/neckline, closures, patterns).
Forbidden: brand, materials, price, quality claims, intended use, season/year.
Errors: wrong type/color = critical; forbidden info = hallucination; missing type/color = partial.
""".strip()

PROMPT = (
    "Describe ONLY the garment for an e-commerce catalog (1-2 short sentences). "
    "Mention garment type, main color, and visible details (sleeves, neckline, closures, pattern/print). "
    "If gender is obvious, include it (e.g., men's / women's / kids'). "
    "Do NOT mention people, background, brand, material, price, quality, season, or use cases."
)
