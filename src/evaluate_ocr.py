import random
import re
import unicodedata
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import jiwer
import matplotlib
import pandas as pd
import pymupdf

matplotlib.use("Agg")
import matplotlib.pyplot as plt 

from ocr import ocr_image, preprocess_image, render_pdf_pages  

SEED = 42
PAGES_PER_MENU = 3            
MIN_REFERENCE_CHARS = 100     

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
RESULTS_DIR = REPO_ROOT / "results"
FIG_DIR = REPO_ROOT / "figures"

CONFIGS = [("eng", False), ("eng", True), ("hrv", False), ("hrv", True)]
BOSNIAN_CHARS = set("čćšžđČĆŠŽĐ")


def normalize_for_eval(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[.·…_]{3,}", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def words(text):
    return re.findall(r"[^\W_]+", text.lower())


def multiset_recall(ref_words, hyp_words):
    if not ref_words:
        return None
    overlap = sum((Counter(ref_words) & Counter(hyp_words)).values())
    return overlap / len(ref_words)


def score_page(reference, hypothesis):
    ref, hyp = normalize_for_eval(reference), normalize_for_eval(hypothesis)
    ref_w, hyp_w = words(ref), words(hyp)
    ref_dia = [w for w in ref_w if set(w) & BOSNIAN_CHARS]
    return {
        "ref_chars": len(ref),
        "cer_ordered": jiwer.cer(ref, hyp) if hyp else 1.0,
        "cer_bow": jiwer.cer(" ".join(sorted(ref_w)), " ".join(sorted(hyp_w))) if hyp_w else 1.0,
        "word_recall": multiset_recall(ref_w, hyp_w),
        "diacritic_recall": multiset_recall(ref_dia, hyp_w),
        "n_diacritic_words": len(ref_dia),
    }


def select_pages(rng):
    sources = pd.read_csv(RAW_DIR / "menu_sources.csv")
    selected = []
    for _, src in sources.iterrows():
        with pymupdf.open(RAW_DIR / src["file_name"]) as doc:
            texts = [p.get_text() for p in doc]
        candidates = [i for i, t in enumerate(texts) if len(t.strip()) >= MIN_REFERENCE_CHARS]
        if not candidates:
            continue 
        for i in sorted(rng.sample(candidates, min(PAGES_PER_MENU, len(candidates)))):
            selected.append((src["file_name"], src["dataset_split"], i, texts[i]))
    return selected


def evaluate_one(task):
    file_name, split, page_idx, reference = task
    image = render_pdf_pages(RAW_DIR / file_name)[page_idx]
    rows = []
    for lang, use_prep in CONFIGS:
        img = preprocess_image(image) if use_prep else image
        hyp = ocr_image(img, lang=lang)
        rows.append({"menu_id": file_name.replace("_menu.pdf", ""), "split": split,
                     "page": page_idx + 1, "lang": lang,
                     "preprocessing": "binarized" if use_prep else "raw",
                     **score_page(reference, hyp)})
    return rows


def make_figures(pages, summary):
    FIG_DIR.mkdir(exist_ok=True)

    labels = summary["lang"] + " / " + summary["preprocessing"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(summary))
    ax.bar([i - 0.2 for i in x], summary["cer_bow"], width=0.4, label="CER (bag-of-words)")
    ax.bar([i + 0.2 for i in x], 1 - summary["diacritic_recall"], width=0.4,
           label="Missed diacritic words (1 - recall)")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Error rate (lower is better)")
    ax.set_title("OCR quality by Tesseract configuration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ocr_config_comparison.png", dpi=150)
    plt.close(fig)

    best = pages[(pages["lang"] == "hrv") & (pages["preprocessing"] == "binarized")]
    per_menu = best.groupby("menu_id")["cer_bow"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    per_menu.plot.barh(ax=ax, color="tab:green")
    ax.set_xlabel("CER (bag-of-words), hrv / binarized")
    ax.set_title("OCR error rate per menu")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ocr_cer_per_menu.png", dpi=150)
    plt.close(fig)


def main():
    rng = random.Random(SEED)
    tasks = select_pages(rng)
    print(f"Evaluating {len(tasks)} pages x {len(CONFIGS)} configurations ...")
    with Pool(2) as pool:
        rows = [r for page_rows in pool.map(evaluate_one, tasks) for r in page_rows]

    pages = pd.DataFrame(rows)
    summary = (pages.groupby(["lang", "preprocessing"], sort=False)
               [["cer_ordered", "cer_bow", "word_recall", "diacritic_recall"]]
               .mean().round(3).reset_index())

    RESULTS_DIR.mkdir(exist_ok=True)
    pages.to_csv(RESULTS_DIR / "ocr_eval_pages.csv", index=False)
    summary.to_csv(RESULTS_DIR / "ocr_eval_summary.csv", index=False)
    make_figures(pages, summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
