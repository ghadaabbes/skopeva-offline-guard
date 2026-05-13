from pathlib import Path
import re

import cv2
import numpy as np
import pytesseract


TESSERACT_PATH = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

if TESSERACT_PATH.exists():
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_PATH)


def clean_ocr_text(text: str) -> str:
    text = text.replace("|", "I")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def _read_uploaded_image(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not read image file.")

    return image


def _generate_preprocessed_versions(image):
    versions = []

    # Original grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize for small text
    scale = 3
    resized = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )

    # Denoise
    denoised = cv2.fastNlMeansDenoising(resized, None, 30, 7, 21)

    # Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    # Otsu threshold
    _, otsu = cv2.threshold(
        sharpened,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    versions.append(resized)
    versions.append(sharpened)
    versions.append(adaptive)
    versions.append(otsu)

    return versions


def _run_tesseract(image, lang: str) -> str:
    config = "--oem 3 --psm 6"

    try:
        return pytesseract.image_to_string(image, lang=lang, config=config)
    except Exception:
        return ""


def extract_text_from_image(image_file) -> str:
    """
    Extracts text from a product label image using local OCR.
    Several preprocessing variants are tested, then the longest result is selected.
    """
    image = _read_uploaded_image(image_file)
    preprocessed_versions = _generate_preprocessed_versions(image)

    candidates = []

    for processed_image in preprocessed_versions:
        # French + English if available
        text_fra_eng = _run_tesseract(processed_image, "fra+eng")
        candidates.append(text_fra_eng)

        # Fallback English only
        text_eng = _run_tesseract(processed_image, "eng")
        candidates.append(text_eng)

    cleaned_candidates = [clean_ocr_text(text) for text in candidates if text.strip()]

    if not cleaned_candidates:
        return ""

    # Choose the richest extraction
    best_text = max(cleaned_candidates, key=len)

    return best_text