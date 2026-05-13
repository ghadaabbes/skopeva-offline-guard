import json
import os
from typing import Any, Dict

import requests


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

# Local fallback for low-memory laptops.
# For final hackathon/cloud demo, change to Gemma 4 model if available.
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma3:1b")


def ask_gemma(prompt: str) -> str:
    """
    Calls local Gemma through Ollama.
    Keeps the prototype offline-first after the model is downloaded.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 4096,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        response.raise_for_status()
        return response.json().get("response", "").strip()

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Ollama is not running. Start it with: ollama serve "
            "or run the model with: ollama run gemma3:1b"
        ) from exc

    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            "Gemma response timed out. Try a shorter prompt or a smaller model."
        ) from exc

    except Exception as exc:
        raise RuntimeError(f"Gemma/Ollama error: {exc}") from exc


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Gemma can sometimes return JSON surrounded by text.
    This function extracts the JSON object safely.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"No valid JSON object found in Gemma response: {text}")

        json_block = text[start : end + 1]
        return json.loads(json_block)


def extract_ingredients_with_gemma(label_text: str) -> Dict[str, Any]:
    prompt = f"""
You are a product label extraction assistant.

Extract the product name, product category, and ingredients from the following product label text.

Return only valid JSON.
Do not add markdown.
Do not add explanations.

Use this exact schema:
{{
  "product_name": string or null,
  "category": "food" | "cosmetic" | "baby" | "supplement" | "unknown",
  "ingredients": [
    {{
      "name": string,
      "normalized_name": string,
      "confidence": "high" | "medium" | "low"
    }}
  ],
  "warnings": [string]
}}

Rules:
- Normalize ingredient names in lowercase.
- Do not invent ingredients.
- If the text is unclear, use confidence "low".
- If category is unclear, use "unknown".

Product label text:
{label_text}
"""

    raw_response = ask_gemma(prompt)
    return extract_json_from_text(raw_response)


def explain_risk_with_gemma(
    product_name,
    category,
    user_context,
    overall_risk,
    ingredients,
    findings,
) -> str:
    prompt = f"""
You are Skopeva Offline Guard, a cautious product safety assistant.

Your role:
- Explain product label concerns in simple language.
- Do not diagnose.
- Do not give treatment.
- Do not claim that the product is dangerous.
- Mention uncertainty when relevant.
- Be transparent that this is a hackathon prototype.

Product name: {product_name or "Unknown"}
Product category: {category}
User context: {user_context}
Overall risk level: {overall_risk}

Detected ingredients:
{ingredients}

Simplified rule-based findings:
{findings}

Return a concise answer with this structure:

1. Simple summary
2. Ingredients to watch
3. Why it matters
4. Safer next step
5. Disclaimer
"""

    return ask_gemma(prompt)