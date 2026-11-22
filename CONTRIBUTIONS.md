# Team Contributions

 ## Lana Hrabac - Dataset Collection and Annotation
-Collected 10 restaurant menu PDFs for the dataset
-Created data/raw/menu_sources.csv to document data about restaurants
-Created data/processed/traditional_dish_glossary_30.csv with 30 dishes and information about each one
-Defined a menu-level development/test split: 80% development menus and 20% test menus
-Selected restaurant Dzenita and Luna for test menus

## Rijad Pleho - Translation and Custom Dish Glossary

**Responsibilities:**

- Integrated the pretrained NLLB-200 model for Bosnian-to-English menu translation.
- Created a custom glossary that contains 30 traditional Bosnian dishes, English descriptions, typical ingredients, meat content, and potential allergens.
- Implemented text normalization for Bosnian characters.
- Developed a matching algorithm that uses "SequenceMatcher" to recognize dish names and minor spelling errors.
- Implemented dish recognition with longer menu phrases.
- Combined NLLB translation and glossary matching into the "final_translate_menu()" function.
- Tested the system using sample menu items and simulated OCR output.
- Exported baseline and glossary translation results to CSV files.

**Files contributed:**
- "Restaurant_Menu_Translation.ipynb"
- "bosnian_dish_glossary_30.csv"
- "glossary_translation_results.csv"
- "baseline_results.csv"


## Tarik Vuk - OCR Implementation and Text Preprocessing

**Responsibilities:**

- Implemented the OCR module (`src/ocr.py`): PDF page rendering, image preprocessing (grayscale, median denoising, Otsu binarization, automatic inversion of dark pages) and Tesseract OCR with the Croatian language model.
- Implemented text preprocessing (`src/preprocess.py`): removal of prices, weights, dotted leaders and OCR noise; classification of lines into headers, dishes and descriptions; rule-based language identification (Bosnian / English / German) to drop translated duplicates; extraction of allergen information printed on menus; ALL-CAPS normalization; segmentation into menu items.
- Produced `data/processed/menu_items.csv` (1,941 Bosnian menu items, 464 from the test menus) used as input for translation and for building the test set.
- Designed and ran the OCR evaluation (`src/evaluate_ocr.py`): the embedded text layer of 8 menus is used as reference; compared 4 Tesseract configurations with CER, order-independent CER, word recall and diacritic recall.
- Data analysis (`src/data_analysis.py`): items per menu, language distribution, item length, glossary coverage (tables in `results/`, charts in `figures/`).
- Added `requirements.txt`, pipeline diagram and run instructions to the README.

**Files contributed:**
- `src/ocr.py`, `src/preprocess.py`, `src/evaluate_ocr.py`, `src/data_analysis.py`
- `data/interim/ocr_pages.csv`, `data/processed/menu_items.csv`
- `results/ocr_eval_pages.csv`, `results/ocr_eval_summary.csv`, `results/items_per_menu.csv`, `results/language_distribution.csv`
- `figures/*.png`, `requirements.txt`
