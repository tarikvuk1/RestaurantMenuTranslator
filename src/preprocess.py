import re
import unicodedata
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OCR_CSV = REPO_ROOT / "data" / "interim" / "ocr_pages.csv"
SOURCES_CSV = REPO_ROOT / "data" / "raw" / "menu_sources.csv"
OUT_CSV = REPO_ROOT / "data" / "processed" / "menu_items.csv"

PRICE_RE = re.compile(r"\b\d{1,3}(?:[.,]\s?\d{1,2})?\s*(?:KM|km|Km|BAM|€|EUR)\b|\b\d{1,3}[.,]\d{2}\b")
QUANTITY_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:kg|gr|g|ml|cl|l|kom|pcs)\b\.?", re.IGNORECASE)
LEADER_RE = re.compile(r"[.·…_\-–—]{3,}")          
NOISE_RE = re.compile(r"[|©®™*•“”\"«»~^<>#=_]")   
SPACE_RE = re.compile(r"\s+")

BOSNIAN_CHARS = set("čćšžđČĆŠŽĐ")
GERMAN_CHARS = set("äöüßÄÖÜ")

BS_WORDS = {"sa", "i", "u", "od", "na", "s", "bez", "ili", "po", "za", "domaći", "domaća",
            "piletina", "meso", "sir", "luk", "povrće", "krompir", "pomfrit", "salata",
            "čorba", "supa", "juha", "pileći", "teleći", "juneći", "riba", "kajmak", "sos"}
EN_WORDS = {"with", "and", "the", "of", "in", "on", "served", "chicken", "beef", "cheese",
            "soup", "salad", "fries", "sauce", "grilled", "fish", "homemade", "fried",
            "bread", "veal", "potatoes", "vegetables", "mushrooms", "cream", "plate"}
DE_WORDS = {"mit", "und", "der", "die", "das", "vom", "auf", "hähnchen", "hänhchen", "käse",
            "suppe", "salat", "pilzen", "gemüse", "kartoffeln", "fleisch", "soße", "sosse",
            "kase", "gemuse", "fruhstuck", "frühstück", "omlett", "schinken", "eier", "brot",
            "wurst", "wurstchen", "hausgemachte", "panierter", "pommes", "frites", "gemischtes",
            "englisches", "hahnchen", "rind", "kalb"}

NON_DISH_RE = re.compile(r"www|http|@|wi-?fi|password|instagram|facebook|follow us|radno vrijeme|"
                         r"din\w* and lounge|reklamacij|konobar|pdv|cijene su", re.IGNORECASE)
ALLERGEN_RE = re.compile(r"[f(\[]?al+ergeni?\s*[:;]\s*(.*)", re.IGNORECASE)

HEADER_WORDS = {"predjela", "predjelo", "hladna", "topla", "glavna", "jela", "salate", "supe",
                "čorbe", "corbe", "deserti", "desert", "pića", "pica", "doručak", "dorucak",
                "roštilj", "rostilj", "tjestenine", "paste", "pizze", "pice", "prilozi",
                "specijaliteti", "sendviči", "napici", "meni", "jelovnik", "drinks", "dessert",
                "starters", "breakfast", "rižota", "rizota", "burgeri", "ponuda", "riba"}


def normalize_unicode(text):
    """NFC normalization: the same letter 'č' can be stored as one or two code points."""
    return unicodedata.normalize("NFC", text)


def clean_line(line):
    """Remove everything that is not part of the dish name or its description."""
    line = normalize_unicode(line)
    line = re.sub(r"^[f(\[](?=[A-ZČĆŠŽĐ])", "", line.strip())  
    line = LEADER_RE.sub(" ", line)
    line = PRICE_RE.sub(" ", line)
    line = QUANTITY_RE.sub(" ", line)
    line = NOISE_RE.sub(" ", line)
    line = re.sub(r"(?<![A-Za-zčćšžđČĆŠŽĐ])\d+(?![A-Za-zčćšžđČĆŠŽĐ])", " ", line)  
    line = SPACE_RE.sub(" ", line).strip(" .,:;-–/")
    tokens = line.split()
    while len(tokens) > 1 and (len(tokens[-1]) == 1 or re.fullmatch(r"(\w)\1+", tokens[-1])):
        tokens.pop()
    return " ".join(tokens)


def letter_count(text):
    return sum(ch.isalpha() for ch in text)


def detect_language(line):
    """Very small rule-based language identifier: 'bs', 'en', 'de' or 'unk'.

    Score = number of language-specific stop words + strong character cues
    (č/ć/š/ž/đ for Bosnian, ä/ö/ü/ß for German).
    """
    words = re.findall(r"[^\W\d_]+", line.lower())
    scores = {
        "bs": sum(w in BS_WORDS for w in words) + 2 * any(ch in BOSNIAN_CHARS for ch in line),
        "en": sum(w in EN_WORDS for w in words),
        "de": sum(w in DE_WORDS for w in words) + 2 * any(ch in GERMAN_CHARS for ch in line),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unk"


def classify_line(line, bracketed=False):
    """Return 'header', 'description' or 'dish'.

    bracketed=True means the raw OCR line started with "(" / "/" (or "f" + capital letter,
    the usual OCR misreading of "("), which on menus marks an ingredient description.
    """
    if bracketed:
        return "description"
    words = line.split()
    lower_words = {w.lower() for w in words}
    if line.isupper() and len(words) <= 3 and lower_words & HEADER_WORDS:
        return "header"
    if line.startswith(("(", "/")) or line[:1].islower() or line.count(",") >= 2:
        return "description"
    return "dish"


def normalize_case(text):
    """Convert ALL CAPS lines to sentence case, keep normal lines as they are."""
    if text.isupper():
        return text[:1].upper() + text[1:].lower()
    return text


def build_items(page_text):
    """Turn one OCR page into a list of menu items (dish + optional description)."""
    items = []
    for raw in page_text.splitlines():
        bracketed = bool(re.match(r"^\s*([(/\[]|f(?=[A-ZČĆŠŽĐ]))", raw))
        line = clean_line(raw)
        if letter_count(line) < 3:
            continue
        if NON_DISH_RE.search(line):
            continue
        allergen = ALLERGEN_RE.search(line)
        if allergen:  
            if items:
                items[-1]["menu_allergens"] = allergen.group(1).strip(" .)]")
            line = line[:allergen.start()].strip(" ,f")
            if letter_count(line) < 3:
                continue
        kind = classify_line(line, bracketed)
        lang = detect_language(line)
        if kind == "header":
            continue
        if kind == "description":
            if items and items[-1]["lang"] in ("bs", "unk"):
                desc = line.strip("()/ ")
                items[-1]["description"] = (items[-1]["description"] + ", " + desc).strip(", ")
            continue
        items.append({"raw_text": raw.strip(), "item_text": normalize_case(line),
                      "description": "", "menu_allergens": "", "lang": lang})
    return items


def main():
    sources = pd.read_csv(SOURCES_CSV)
    ocr_pages = pd.read_csv(OCR_CSV, keep_default_na=False)
    rows = []
    for _, src in sources.iterrows():
        menu_id = src["file_name"].replace("_menu.pdf", "")
        menu_pages = ocr_pages[ocr_pages["menu_id"] == menu_id].sort_values("page")
        for page_no, page_text in zip(menu_pages["page"], menu_pages["ocr_text"]):
            for item in build_items(page_text):
                rows.append({"menu_id": menu_id, "restaurant": src["restaurant_name"],
                             "split": src["dataset_split"], "page": page_no, **item})

    df = pd.DataFrame(rows)
    df.insert(0, "item_id", [f"{m}_{i:04d}" for i, m in enumerate(df["menu_id"])])
    df["keep"] = df["lang"].isin(["bs", "unk"])  
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Saved {len(df)} lines ({df['keep'].sum()} kept as Bosnian items) to {OUT_CSV}")


if __name__ == "__main__":
    main()
