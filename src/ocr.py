"""
OCR module for the Restaurant Menu Translator.
Author: Tarik Vuk

Pipeline for one menu PDF:
    PDF page -> image (PyMuPDF, 300 DPI)
             -> optional image preprocessing (grayscale, denoising, Otsu binarization)
             -> Tesseract OCR (external tool, language model 'hrv' or 'eng')
             -> raw text per page, saved to data/interim/ocr_pages.csv

Usage (from the repository root):
    python src/ocr.py                      # OCR all menus with the default config
    python src/ocr.py --lang eng --no-preprocess
"""

import argparse
import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pymupdf
import pytesseract
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # some menu pages are very large at 300 DPI; the files are our own data

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
OCR_CSV = REPO_ROOT / "data" / "interim" / "ocr_pages.csv"

DEFAULT_DPI = 300
DEFAULT_LANG = "hrv"  # Croatian model: covers č, ć, š, ž, đ (there is also 'bos', results were almost identical)
# --psm 4 = "assume a single column of text of variable sizes"; works best for menus
# where the dish name is on the left and the price on the right.
DEFAULT_PSM = 4


def render_pdf_pages(pdf_path, dpi=DEFAULT_DPI):
    """Render every page of a PDF to a PIL image."""
    images = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            images.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
    return images


def extract_text_layer(pdf_path):
    """Return the embedded (digital) text of every page.

    Most menu PDFs were exported from design tools, so they contain the real text.
    We use this text as the reference ("ground truth") when we measure OCR quality.
    Scanned PDFs return empty strings.
    """
    with pymupdf.open(pdf_path) as doc:
        return [page.get_text() for page in doc]


def preprocess_image(img):
    """Classic image cleanup before OCR.

    1. grayscale        - colour carries no information for OCR
    2. median blur (3x3) - removes salt-and-pepper noise from scans/photos
    3. Otsu threshold   - automatic black/white binarization, good for dark text on light paper
    4. if the page is mostly dark (white text on dark background, common on menus),
       invert it, because Tesseract expects dark text on a light background
    """
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 3)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary.mean() < 127:
        binary = cv2.bitwise_not(binary)
    return Image.fromarray(binary)


def ocr_image(img, lang=DEFAULT_LANG, psm=DEFAULT_PSM):
    """Run Tesseract on one image and return the recognized text."""
    return pytesseract.image_to_string(img, lang=lang, config=f"--psm {psm}")


def ocr_pdf(pdf_path, lang=DEFAULT_LANG, use_preprocessing=True, dpi=DEFAULT_DPI):
    """OCR every page of a PDF. Returns a list of page texts."""
    texts = []
    for img in render_pdf_pages(pdf_path, dpi=dpi):
        if use_preprocessing:
            img = preprocess_image(img)
        texts.append(ocr_image(img, lang=lang))
    return texts


def main():
    parser = argparse.ArgumentParser(description="OCR all restaurant menus.")
    parser.add_argument("--lang", default=DEFAULT_LANG)
    parser.add_argument("--no-preprocess", action="store_true")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    args = parser.parse_args()

    sources = pd.read_csv(RAW_DIR / "menu_sources.csv")
    rows = []

    for _, row in sources.iterrows():
        pdf_path = RAW_DIR / row["file_name"]
        menu_id = pdf_path.stem.replace("_menu", "")  # e.g. restaurant_05
        pages = ocr_pdf(pdf_path, lang=args.lang,
                        use_preprocessing=not args.no_preprocess, dpi=args.dpi)
        for page_no, text in enumerate(pages, start=1):
            rows.append({"menu_id": menu_id, "page": page_no, "ocr_text": text})
        print(f"{menu_id}: {len(pages)} pages OCR-ed")

    OCR_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OCR_CSV, index=False, encoding="utf-8")
    print(f"Saved OCR text of {len(rows)} pages to {OCR_CSV}")


if __name__ == "__main__":
    main()
