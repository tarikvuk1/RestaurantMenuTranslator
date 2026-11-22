# RestaurantMenuTranslator

Team:

1. Lana Hrabač
2. Davud Mahmutović
3. Rijad Pleho
4. Tarik Vuk

Topic: Restaurant Menu Translator

## Pipeline (draft)

```
 menu PDF / photo
       │
       ▼
 [EXTERNAL] PyMuPDF – render page to image (300 DPI)
       │
       ▼
 [OWN] image preprocessing – grayscale, median denoise, Otsu binarization, auto-invert   (src/ocr.py)
       │
       ▼
 [EXTERNAL] Tesseract OCR, Croatian model 'hrv'                                         (src/ocr.py)
       │
       ▼
 [OWN] text preprocessing – remove prices/weights/noise, header/dish/description         (src/preprocess.py)
       classification, rule-based language ID (bs/en/de), allergen extraction,
       item segmentation → data/processed/menu_items.csv
       │
       ▼
 [OWN] glossary dish matcher (fuzzy, normalized)   ──►  [EXTERNAL] NLLB-200 translation   (notebook)
       │
       ▼
 English translation + description, ingredients, meat, allergens
```

## How to run (OCR + preprocessing)

Requirements: Python 3.10+, Tesseract 5 with the Croatian language model (`hrv.traineddata`,
e.g. `sudo apt install tesseract-ocr tesseract-ocr-hrv`, or copy the file from
https://github.com/tesseract-ocr/tessdata_fast into your `tessdata` folder).

```bash
pip install -r requirements.txt
python src/ocr.py            # PDFs in data/raw -> data/interim/ocr_pages.csv   (~3 min)
python src/preprocess.py     # OCR text -> data/processed/menu_items.csv
python src/data_analysis.py  # tables in results/, charts in figures/
python src/evaluate_ocr.py   # OCR evaluation (CER) -> results/ocr_eval_*.csv, figures/  (~5 min, seed 42)
```

## First results – OCR

Reference = embedded text layer of the 8 menus that have one; 21 pages sampled (seed 42).
`cer_bow` = CER after sorting words (ignores reading order); `diacritic_recall` = share of
reference words with č/ć/š/ž/đ that OCR returned exactly.

| lang | preprocessing | cer_ordered | cer_bow | word_recall | diacritic_recall |
|:-----|:--------------|------------:|--------:|------------:|-----------------:|
| eng  | raw           | 0.526 | 0.256 | 0.726 | 0.000 |
| eng  | binarized     | 0.526 | 0.250 | 0.732 | 0.000 |
| hrv  | raw           | 0.502 | 0.172 | 0.825 | 0.908 |
| hrv  | binarized     | 0.499 | 0.159 | 0.834 | 0.904 |

The English model loses every diacritic (Ćevapi → Cevapi), the Croatian model keeps ~90 %.
`hrv` + binarization is the default configuration of the pipeline.

## M3 evaluation

Run `python evaluation/m3_review.py --repo .` from the repository root. Only pandas is required for this review. It reruns text preprocessing on saved OCR output and verifies the existing data tables and OCR summary; Tesseract and NLLB are not rerun.

`evaluation/M3_Final_Check.ipynb` is the Colab entry point. Upload `RestaurantMenuTranslator_M3.zip` and run both cells. Results and input hashes are saved in `evaluation/m3_review_results/`. The original tables, figures and baseline results remain in `results/` and `figures/`.

`evaluation/pilot_test_v1.csv` contains 27 PDF-linked initial reference items, with AI-assisted provenance recorded. Labels are a first pilot version, not a final independently validated benchmark. The review reports missing extractions separately from incorrect matcher labels.

`evaluation/split_manifest.csv` assigns six menus to train/development, Toplik to validation, and preserves the original three test menus. Original source tags remain unchanged for compatibility; this manifest subdivides the development pool without claiming model training.

The translation notebook loads the packaged 32-entry glossary and includes a final cell connecting three saved OCR items to the existing translation function. That added inference cell has not been executed in this review. AI assistance is documented in `docs/AI_LOG.md`.
