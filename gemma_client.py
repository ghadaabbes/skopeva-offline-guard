import json
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"

# Change this if your local model name is different.
# MODEL_NAME = "gemma4:e2b"
MODEL_NAME = "gemma3:1b"

def ask_gemma(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json().get("response", "").strip()


def extract_json_from_text(text: str):
    """
    Gemma may sometimes wrap JSON in text.
    This helper extracts the JSON block safely.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            raise ValueError("No JSON object found in Gemma response.")

        json_block = text[start:end + 1]
        return json.loads(json_block)


def extract_ingredients_with_gemma(label_text: str):
    prompt = f"""
You are a product label extraction assistant.

Extract product category and ingredients from the following label text.
Return only valid JSON. Do not add markdown. Do not add explanations.

Schema:
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

Label text:
{label_text}
"""

    raw_response = ask_gemma(prompt)
    return extract_json_from_text(raw_response)


def explain_risk_with_gemma(product_name, category, user_context, overall_risk, ingredients, findings):
    prompt = f"""
You are Skopeva Offline Guard, a cautious product safety assistant.

Explain the product analysis in simple language.
Do not diagnose.
Do not give medical treatment.
Be transparent that this is a hackathon prototype.
Keep the answer concise and useful.

Product name: {product_name or "Unknown"}
Category: {category}
User context: {user_context}
Overall risk level: {overall_risk}

Detected ingredients:
{ingredients}

Ingredients to watch:
{findings}

Return this structure:
1. Simple summary
2. Ingredients to watch
3. Why it matters
4. Safer next step
5. Disclaimer
"""

    return ask_gemma(prompt)