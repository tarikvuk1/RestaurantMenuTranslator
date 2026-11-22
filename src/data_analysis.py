"""
Data analysis of the extracted menu items (tables and charts for the paper / M3).
Author: Tarik Vuk

Input : data/processed/menu_items.csv   (from src/preprocess.py)
        data/processed/traditional_dish_glossary_10.csv
Output: results/items_per_menu.csv, results/language_distribution.csv,
        figures/items_per_menu.png, figures/item_length_hist.png

Usage (from the repository root):
    python src/data_analysis.py
"""

import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ITEMS_CSV = REPO_ROOT / "data" / "processed" / "menu_items.csv"
GLOSSARY_CSV = REPO_ROOT / "data" / "processed" / "traditional_dish_glossary_10.csv"
RESULTS_DIR = REPO_ROOT / "results"
FIG_DIR = REPO_ROOT / "figures"


def fold(text):
    """Lowercase and remove diacritics, used only for glossary look-up."""
    text = str(text).lower().replace("đ", "dj")
    text = unicodedata.normalize("NFKD", text)
    return re.sub(r"\s+", " ", "".join(c for c in text if not unicodedata.combining(c))).strip()


def glossary_hit(item, glossary_names, threshold=0.85):
    """True if the item contains (fuzzily) one of the glossary dish names."""
    item_f = fold(item)
    for name in glossary_names:
        n_words = len(name.split())
        tokens = item_f.split()
        for i in range(max(1, len(tokens) - n_words + 1)):
            if SequenceMatcher(None, " ".join(tokens[i:i + n_words]), name).ratio() >= threshold:
                return True
    return False


def main():
    df = pd.read_csv(ITEMS_CSV, keep_default_na=False)
    glossary = [fold(n) for n in pd.read_csv(GLOSSARY_CSV)["bosnian_name"]]
    RESULTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    # Table 1: language distribution of extracted lines
    lang = df["lang"].value_counts().rename_axis("lang").reset_index(name="lines")
    lang["share"] = (lang["lines"] / lang["lines"].sum()).round(3)
    lang.to_csv(RESULTS_DIR / "language_distribution.csv", index=False)

    # Table 2: items per menu, split and glossary coverage
    items = df[df["keep"].astype(str) == "True"].copy()
    items["in_glossary"] = items["item_text"].apply(lambda t: glossary_hit(t, glossary))
    items["n_words"] = items["item_text"].str.split().str.len()
    per_menu = (items.groupby(["menu_id", "restaurant", "split"])
                .agg(pages=("page", "nunique"), items=("item_id", "count"),
                     with_description=("description", lambda s: (s != "").sum()),
                     glossary_matches=("in_glossary", "sum"))
                .reset_index())
    per_menu.to_csv(RESULTS_DIR / "items_per_menu.csv", index=False)

    # Figure 1: items per menu coloured by split
    colors = per_menu["split"].map({"development": "tab:blue", "test": "tab:orange"})
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(per_menu["menu_id"].str.replace("restaurant_", "R"), per_menu["items"], color=colors)
    ax.set_ylabel("Extracted Bosnian menu items")
    ax.set_title("Menu items per restaurant (blue = development, orange = test)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "items_per_menu.png", dpi=150)
    plt.close(fig)

    # Figure 2: item length distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    items["n_words"].clip(upper=12).plot.hist(bins=range(1, 14), ax=ax, rwidth=0.9)
    ax.set_xlabel("Words per item (12 = 12 or more)")
    ax.set_ylabel("Items")
    ax.set_title("Length of extracted menu items")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "item_length_hist.png", dpi=150)
    plt.close(fig)

    print(lang.to_string(index=False))
    print()
    print(per_menu.to_string(index=False))
    print("\nTotal items:", len(items), "| test items:", (items["split"] == "test").sum())


if __name__ == "__main__":
    main()
