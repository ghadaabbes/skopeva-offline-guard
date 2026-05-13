import re
from difflib import SequenceMatcher
from typing import List


KNOWN_INGREDIENTS = [
    "eau",
    "avocat",
    "tomates",
    "oignons",
    "farine de blé",
    "préparation fromagère",
    "crème",
    "amidon modifié de maïs",
    "huile de colza",
    "flocons de pomme de terre déshydratés",
    "sulfites",
    "purée d’ail",
    "extrait de lactosérum",
    "lait",
    "sel",
    "épices",
    "purée de citron vert concentré",
    "lactosérum en poudre",
    "sucre",
    "sirop de glucose déshydraté",
    "correcteurs d’acidité",
    "glucono-delta-lactone",
    "acide citrique",
    "piment jalapeño déshydraté",
    "herbes",
    "dextrose",
    "antioxydant",
    "acide ascorbique",
    "colorants",
    "chlorophyllines",
    "lutéine",
    "extrait de levure",
    "caséinate",
]


OCR_FIXES = {
    "farine deble": "farine de blé",
    "hulle de cola": "huile de colza",
    "lactoserum": "lactosérum",
    "sucte": "sucre",
    "slucose": "glucose",
    "ctvique": "citrique",
    "hefdes": "herbes",
    "castinate": "caséinate",
    "allergenes": "allergènes",
    "qvocat": "avocat",
    "morceau": "morceaux",
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub(r"[^a-zàâäéèêëîïôöùûüçœ0-9\s'\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    for wrong, correct in OCR_FIXES.items():
        text = text.replace(wrong, correct)

    return text


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def clean_ocr_ingredients(raw_text: str) -> List[str]:
    """
    Converts noisy OCR text into a clean ingredient list.
    This is a hackathon-safe cleaner, not Skopeva proprietary scoring.
    """
    text = normalize_text(raw_text)
    detected = []

    for ingredient in KNOWN_INGREDIENTS:
        ingredient_norm = normalize_text(ingredient)

        if ingredient_norm in text:
            detected.append(ingredient)
            continue

        # fuzzy check on small chunks
        words = text.split()
        n = len(ingredient_norm.split())

        for i in range(0, max(len(words) - n + 1, 1)):
            chunk = " ".join(words[i : i + n])
            if similarity(chunk, ingredient_norm) >= 0.78:
                detected.append(ingredient)
                break

    # remove duplicates while preserving order
    clean = []
    for item in detected:
        if item not in clean:
            clean.append(item)

    return clean