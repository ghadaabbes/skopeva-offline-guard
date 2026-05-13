import json
import re
from pathlib import Path

import streamlit as st


APP_TITLE = "Skopeva Offline Guard"
RULES_PATH = Path("demo_data/risk_rules.json")


def load_rules():
    if not RULES_PATH.exists():
        st.error("risk_rules.json not found in demo_data/")
        return {}
    with open(RULES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(text: str) -> str:
    return text.lower().strip()


def extract_ingredients(raw_text: str):
    text = normalize_text(raw_text)

    prefixes = [
        "ingredients:",
        "ingredient:",
        "ingrédients:",
        "composition:",
        "contains:",
        "contient:"
    ]

    for prefix in prefixes:
        text = text.replace(prefix, "")

    parts = re.split(r"[,;\n]", text)
    ingredients = []

    for part in parts:
        cleaned = part.strip(" .:-()[]")
        if cleaned and len(cleaned) > 1:
            ingredients.append(cleaned)

    return ingredients


def analyze_ingredients(ingredients, rules):
    findings = []

    for ingredient in ingredients:
        ingredient_normalized = normalize_text(ingredient)

        for rule_key, rule in rules.items():
            if rule_key in ingredient_normalized:
                findings.append({
                    "ingredient": ingredient,
                    "matched_rule": rule_key,
                    "risk_level": rule["risk_level"],
                    "category": rule["category"],
                    "concern": rule["concern"]
                })

    return findings


def compute_overall_risk(findings):
    if not findings:
        return "low"

    priority = {
        "low": 1,
        "medium": 2,
        "high_attention": 3
    }

    max_score = max(priority.get(item["risk_level"], 1) for item in findings)

    if max_score == 3:
        return "high_attention"
    if max_score == 2:
        return "medium"
    return "low"


def risk_label(risk_level):
    labels = {
        "low": "Low",
        "medium": "Medium attention",
        "high_attention": "High attention"
    }
    return labels.get(risk_level, "Unknown")


def main():
    st.set_page_config(
        page_title="Skopeva Offline Guard",
        page_icon="🛡️",
        layout="wide"
    )

    st.title("🛡️ Skopeva Offline Guard")
    st.subheader("Privacy-first product safety scanner — Hackathon Edition")

    st.info(
        "This is a hackathon prototype. It does not include Skopeva’s proprietary "
        "scoring engine, private datasets, advanced prompts, or production safety logic."
    )

    rules = load_rules()

    left, right = st.columns([1, 1])

    with left:
        st.markdown("### 1. Product input")

        product_name = st.text_input("Product name", placeholder="Example: Energy drink")
        product_category = st.selectbox(
            "Product category",
            ["food", "cosmetic", "baby", "supplement", "unknown"]
        )

        user_profile = st.selectbox(
            "User context",
            [
                "General user",
                "Pregnancy-sensitive profile",
                "Child / parent profile",
                "Sensitive skin",
                "Sugar-conscious profile"
            ]
        )

        raw_ingredients = st.text_area(
            "Paste ingredients",
            height=180,
            placeholder="INGREDIENTS: water, sugar, caffeine, taurine, citric acid, natural flavors."
        )

        uploaded_image = st.file_uploader(
            "Optional: upload product label image",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded label image", use_container_width=True)
            st.warning(
                "Image OCR/Gemma vision will be added in the next step. "
                "For now, paste the ingredients text manually."
            )

        analyze_button = st.button("Analyze product")

    with right:
        st.markdown("### 2. Analysis result")

        if analyze_button:
            if not raw_ingredients.strip():
                st.error("Please paste ingredients to analyze.")
                return

            ingredients = extract_ingredients(raw_ingredients)
            findings = analyze_ingredients(ingredients, rules)
            overall_risk = compute_overall_risk(findings)

            st.markdown("#### Product summary")
            st.write(f"**Product:** {product_name or 'Unknown'}")
            st.write(f"**Category:** {product_category}")
            st.write(f"**User context:** {user_profile}")

            st.metric("Overall risk level", risk_label(overall_risk))

            st.markdown("#### Detected ingredients")
            if ingredients:
                for ingredient in ingredients:
                    st.write(f"- {ingredient}")
            else:
                st.write("No ingredients detected.")

            st.markdown("#### Ingredients to watch")

            if findings:
                for item in findings:
                    with st.expander(f"{item['ingredient']} — {risk_label(item['risk_level'])}"):
                        st.write(f"**Category:** {item['category']}")
                        st.write(f"**Concern:** {item['concern']}")
            else:
                st.success("No specific concern detected with the simplified demo rules.")

            st.markdown("#### Privacy note")
            st.write(
                "This prototype performs analysis locally with a simplified rule engine. "
                "No personal health profile is sent to a server in this demo."
            )

            st.markdown("#### Disclaimer")
            st.caption(
                "This tool is for educational and informational purposes only. "
                "It is not medical advice and does not replace professional guidance."
            )


if __name__ == "__main__":
    main()