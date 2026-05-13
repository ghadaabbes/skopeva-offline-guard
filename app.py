import json
import re
from pathlib import Path
from typing import Any, Dict, List
from ocr_utils import extract_text_from_image
from ingredient_cleaner import clean_ocr_ingredients
import streamlit as st


# Optional Gemma client import
# The app must still work even if Ollama/Gemma is not available.
try:
    from gemma_client import (
        extract_ingredients_with_gemma,
        explain_risk_with_gemma,
        clean_ocr_label_text_with_gemma,
    )

    GEMMA_AVAILABLE = True
except Exception:
    GEMMA_AVAILABLE = False


APP_TITLE = "Skopeva Offline Guard"
RULES_PATH = Path("demo_data/risk_rules.json")


# -----------------------------
# Helpers
# -----------------------------
def load_rules() -> Dict[str, Dict[str, str]]:
    if not RULES_PATH.exists():
        st.error("Missing file: demo_data/risk_rules.json")
        return {}

    with open(RULES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(text: str) -> str:
    return text.lower().strip()


def extract_ingredients_fallback(raw_text: str) -> List[str]:
    """
    Simple fallback parser when Gemma is not used or fails.
    This is intentionally basic for the hackathon prototype.
    """
    text = normalize_text(raw_text)

    prefixes = [
        "ingredients:",
        "ingredient:",
        "ingrédients:",
        "composition:",
        "contains:",
        "contient:",
    ]

    for prefix in prefixes:
        text = text.replace(prefix, "")

    parts = re.split(r"[,;\n]", text)
    ingredients = []

    for part in parts:
        cleaned = part.strip(" .:-()[]{}")
        if cleaned and len(cleaned) > 1:
            ingredients.append(cleaned)

    return ingredients


def analyze_ingredients(
    ingredients: List[str],
    rules: Dict[str, Dict[str, str]],
    user_profile: str,
) -> List[Dict[str, Any]]:
    findings = []

    for ingredient in ingredients:
        ingredient_normalized = normalize_text(ingredient)

        for rule_key, rule in rules.items():
            if rule_key in ingredient_normalized:
                risk_level = rule.get("risk_level", "low")

                # Simple demo-only user context adjustment.
                # This is not proprietary scoring logic.
                if user_profile == "Pregnancy-sensitive profile" and rule_key in [
                    "caffeine",
                    "alcohol denat",
                ]:
                    risk_level = "high_attention"

                if user_profile == "Child / parent profile" and rule_key in [
                    "caffeine",
                    "sugar",
                    "glucose syrup",
                ]:
                    risk_level = "high_attention"

                if user_profile == "Sensitive skin" and rule_key in [
                    "fragrance",
                    "alcohol denat",
                    "paraben",
                ]:
                    risk_level = "high_attention"

                if user_profile == "Sugar-conscious profile" and rule_key in [
                    "sugar",
                    "glucose syrup",
                ]:
                    risk_level = "high_attention"

                findings.append(
                    {
                        "ingredient": ingredient,
                        "matched_rule": rule_key,
                        "risk_level": risk_level,
                        "category": rule.get("category", "unknown"),
                        "concern": rule.get("concern", ""),
                    }
                )

    return findings


def compute_overall_risk(findings: List[Dict[str, Any]]) -> str:
    if not findings:
        return "low"

    priority = {
        "low": 1,
        "medium": 2,
        "high_attention": 3,
    }

    max_score = max(priority.get(item.get("risk_level", "low"), 1) for item in findings)

    if max_score == 3:
        return "high_attention"
    if max_score == 2:
        return "medium"

    return "low"


def risk_label(risk_level: str) -> str:
    labels = {
        "low": "Low",
        "medium": "Medium attention",
        "high_attention": "High attention",
    }
    return labels.get(risk_level, "Unknown")


def risk_emoji(risk_level: str) -> str:
    emojis = {
        "low": "🟢",
        "medium": "🟠",
        "high_attention": "🔴",
    }
    return emojis.get(risk_level, "⚪")


def build_local_explanation(
    product_name: str,
    product_category: str,
    user_profile: str,
    overall_risk: str,
    ingredients: List[str],
    findings: List[Dict[str, Any]],
) -> str:
    """
    Local fallback explanation when Gemma is not available.
    """
    if not findings:
        return (
            f"Based on the simplified demo rules, no specific ingredient of concern "
            f"was detected for this {product_category} product. "
            f"This does not mean the product is risk-free; it only means no match was found "
            f"in the limited hackathon rule set."
        )

    watched = ", ".join(sorted({item["ingredient"] for item in findings}))

    return (
        f"{product_name or 'This product'} has an overall risk level of "
        f"{risk_label(overall_risk)} for the selected context: {user_profile}. "
        f"The ingredients to watch are: {watched}. "
        f"This analysis is based on a simplified local rule engine for demo purposes only."
    )


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🛡️",
        layout="wide",
    )

    st.title("🛡️ Skopeva Offline Guard")
    st.subheader("Privacy-first product safety scanner")


    rules = load_rules()

    with st.sidebar:
        st.markdown("## Settings")

        st.markdown("### AI options")

        use_gemma_extraction = st.checkbox(
            "Use local Gemma for ingredient extraction",
            value=False,
            disabled=not GEMMA_AVAILABLE,
        )

        use_gemma_explanation = st.checkbox(
            "Use local Gemma for explanation",
            value=False,
            disabled=not GEMMA_AVAILABLE,
        )

        if not GEMMA_AVAILABLE:
            st.warning(
                "Gemma client not detected. The app will use local fallback logic only."
            )

        st.markdown("### Privacy")
        st.success(
            "This demo is designed as offline-first. The local rule engine runs without "
            "sending personal profile data to a server."
        )

    left, right = st.columns([1, 1])

    with left:
        st.markdown("## 1. Product input")

        product_name = st.text_input(
            "Product name",
            placeholder="Example: Energy drink",
        )

        product_category = st.selectbox(
            "Product category",
            ["food", "cosmetic", "baby", "supplement", "unknown"],
        )

        user_profile = st.selectbox(
            "User context",
            [
                "General user",
                "Pregnancy-sensitive profile",
                "Child / parent profile",
                "Sensitive skin",
                "Sugar-conscious profile",
            ],
        )
        
        if "ocr_raw_text" in st.session_state:
            with st.expander("Raw OCR text"):
                st.text(st.session_state["ocr_raw_text"])
        default_ocr_text = st.session_state.get("ocr_text", "")
        raw_ingredients = st.text_area(
            "Paste product ingredients or label text",
            value=default_ocr_text,
            height=200,
            placeholder=(
                "INGREDIENTS: water, sugar, caffeine, taurine, "
                "citric acid, natural flavors."
            ),
        )

        uploaded_image = st.file_uploader(
            "Optional: upload product label image",
            type=["png", "jpg", "jpeg"],
        )

        if uploaded_image:
            st.image(
                uploaded_image,
                caption="Uploaded product label",
                use_container_width=True,
            )

            if st.button("Extract text from image with local OCR"):
                try:
                    with st.spinner("Extracting text from image..."):
                        extracted_text = extract_text_from_image(uploaded_image)

                    if extracted_text:
                        st.session_state["ocr_raw_text"] = extracted_text

                        if GEMMA_AVAILABLE:
                            try:
                                with st.spinner("Cleaning OCR text with local Gemma..."):
                                    cleaned_text = clean_ocr_label_text_with_gemma(extracted_text)

                                st.session_state["ocr_text"] = cleaned_text
                                st.success("Text extracted and cleaned with local Gemma.")

                            except Exception as exc:
                                st.session_state["ocr_text"] = extracted_text
                                st.warning(
                                    f"OCR worked, but Gemma cleanup failed. Raw OCR will be used. Error: {exc}"
                                )
                        else:
                            st.session_state["ocr_text"] = extracted_text
                            st.success("Text extracted from image.")
                    else:
                        st.warning("No readable text detected. Try a clearer image.")

                except Exception as exc:
                    st.error(f"OCR failed: {exc}")

        analyze_button = st.button("Analyze product", type="primary")

    with right:
        st.markdown("## 2. Analysis result")

        if not analyze_button:
            st.caption("Paste ingredients and click **Analyze product** to start.")
            return

        if not raw_ingredients.strip():
            st.error("Please paste ingredients or label text to analyze.")
            return

        # -----------------------------
        # Ingredient extraction
        # -----------------------------
        extraction_source = "Fallback parser"
        gemma_extraction = None

        if use_gemma_extraction and GEMMA_AVAILABLE:
            try:
                with st.spinner("Extracting ingredients with local Gemma..."):
                    gemma_extraction = extract_ingredients_with_gemma(raw_ingredients)

                extracted_product_name = gemma_extraction.get("product_name")
                extracted_category = gemma_extraction.get("category")

                if extracted_product_name and not product_name:
                    product_name = extracted_product_name

                if extracted_category:
                    product_category = extracted_category

                ingredients = [
                    item.get("normalized_name") or item.get("name")
                    for item in gemma_extraction.get("ingredients", [])
                    if item.get("normalized_name") or item.get("name")
                ]

                extraction_source = "Local Gemma"
                st.success("Ingredients extracted with local Gemma.")

            except Exception as exc:
                st.warning(
                    f"Gemma extraction failed. Using fallback parser instead. Error: {exc}"
                )
                ingredients = clean_ocr_ingredients(raw_ingredients)

                if not ingredients:
                    ingredients = extract_ingredients_fallback(raw_ingredients)
        else:
            ingredients = clean_ocr_ingredients(raw_ingredients)

            if not ingredients:
                ingredients = extract_ingredients_fallback(raw_ingredients)

        # -----------------------------
        # Rule-based analysis
        # -----------------------------
        findings = analyze_ingredients(
            ingredients=ingredients,
            rules=rules,
            user_profile=user_profile,
        )

        overall_risk = compute_overall_risk(findings)

        # -----------------------------
        # Display summary
        # -----------------------------
        st.markdown("### Product summary")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric("Product", product_name or "Unknown")

        with col_b:
            st.metric("Category", product_category)

        with col_c:
            st.metric(
                "Overall risk",
                f"{risk_emoji(overall_risk)} {risk_label(overall_risk)}",
            )

        st.write(f"**User context:** {user_profile}")
        st.write(f"**Extraction source:** {extraction_source}")

        # -----------------------------
        # Display detected ingredients
        # -----------------------------
        st.markdown("### Detected ingredients")

        if ingredients:
            for ingredient in ingredients:
                st.write(f"- {ingredient}")
        else:
            st.write("No ingredients detected.")

        # -----------------------------
        # Display findings
        # -----------------------------
        st.markdown("### Ingredients to watch")

        if findings:
            for item in findings:
                title = (
                    f"{risk_emoji(item['risk_level'])} "
                    f"{item['ingredient']} — {risk_label(item['risk_level'])}"
                )

                with st.expander(title, expanded=True):
                    st.write(f"**Matched rule:** {item['matched_rule']}")
                    st.write(f"**Category:** {item['category']}")
                    st.write(f"**Concern:** {item['concern']}")
        else:
            st.success(
                "No specific concern detected with the simplified hackathon rules."
            )

        # -----------------------------
        # Explanation
        # -----------------------------
        st.markdown("### Explanation")

        explanation = None

        if use_gemma_explanation and GEMMA_AVAILABLE:
            try:
                with st.spinner("Generating explanation with local Gemma..."):
                    explanation = explain_risk_with_gemma(
                        product_name=product_name,
                        category=product_category,
                        user_context=user_profile,
                        overall_risk=overall_risk,
                        ingredients=ingredients,
                        findings=findings,
                    )

                st.write(explanation)

            except Exception as exc:
                st.warning(
                    f"Gemma explanation failed. Using local fallback explanation. Error: {exc}"
                )
                explanation = build_local_explanation(
                    product_name=product_name,
                    product_category=product_category,
                    user_profile=user_profile,
                    overall_risk=overall_risk,
                    ingredients=ingredients,
                    findings=findings,
                )
                st.write(explanation)
        else:
            explanation = build_local_explanation(
                product_name=product_name,
                product_category=product_category,
                user_profile=user_profile,
                overall_risk=overall_risk,
                ingredients=ingredients,
                findings=findings,
            )
            st.write(explanation)

        # -----------------------------
        # Optional debug
        # -----------------------------
        with st.expander("Debug JSON"):
            st.json(
                {
                    "product_name": product_name,
                    "category": product_category,
                    "user_profile": user_profile,
                    "ingredients": ingredients,
                    "findings": findings,
                    "overall_risk": overall_risk,
                    "gemma_extraction": gemma_extraction,
                }
            )

        # -----------------------------
        # Privacy and disclaimer
        # -----------------------------
        st.markdown("### Privacy note")
        st.write(
            "This prototype is designed to run locally. The rule-based analysis runs "
            "offline, and the selected user context is not sent to any external server "
            "in this demo."
        )

        st.markdown("### Disclaimer")
        st.caption(
            "This tool is for educational and informational purposes only. "
            "It is not medical advice, does not diagnose conditions, and does not "
            "replace professional guidance."
        )


if __name__ == "__main__":
    main()