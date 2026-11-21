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
