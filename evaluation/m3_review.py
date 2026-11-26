import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


def load_matcher(path):
    selected = {}
    for cell in json.loads(path.read_text(encoding='utf-8-sig'))['cells']:
        if cell['cell_type'] != 'code':
            continue
        try:
            tree = ast.parse(''.join(cell['source']))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in {
                'normalize_text', 'find_dish_in_phrase'}:
                selected[node.name] = node
    if len(selected) != 2:
        raise ValueError('Expected notebook functions missing')
    env = {'re': re, 'unicodedata': unicodedata, 'SequenceMatcher': SequenceMatcher}
    module = ast.Module(body=[selected['normalize_text'], selected['find_dish_in_phrase']], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), env)
    return env['find_dish_in_phrase']


def load_functions(path, names):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in body} != set(names):
        raise ValueError(f'Missing functions in {path}')
    env = {'re': re, 'unicodedata': unicodedata, 'SequenceMatcher': SequenceMatcher}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), 'exec'), env)
    return env


def read(path):
    return pd.read_csv(path, keep_default_na=False)


def equal_frames(a, b):
    return list(a.columns) == list(b.columns) and a.astype(str).equals(b.astype(str))


def run(repo, out, pilot_path):
    out.mkdir(parents=True, exist_ok=True)
    glossary_path = repo/'data/processed/traditional_dish_glossary_10.csv'
    items_path = repo/'data/processed/menu_items.csv'
    items = read(items_path)
    glossary = read(glossary_path)
    sources = read(repo/'data/raw/menu_sources.csv')
    kept = items[items['keep'].astype(str) == 'True'].copy()
    matcher = load_matcher(repo/'Restaurant_Menu_Translation.ipynb')

    split_path = Path(__file__).with_name('split_manifest.csv')
    splits = read(split_path)
    if not splits['menu_id'].is_unique or set(splits['menu_id']) != set(items['menu_id']):
        raise ValueError('Split manifest must assign every menu exactly once')
    if set(splits['evaluation_split']) != {'train','validation','test'}:
        raise ValueError('Expected train, validation and test roles')
    role_map = splits.set_index('menu_id')['evaluation_split']
    split_summary = splits.groupby('evaluation_split').size().reset_index(name='menus')
    split_summary.to_csv(out/'split_summary.csv',index=False)

    module_path = repo/'src/preprocess.py'
    spec = importlib.util.spec_from_file_location('team_preprocess', module_path)
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    prep.OUT_CSV = out/'.preprocessing_check.csv'
    with contextlib.redirect_stdout(io.StringIO()):
        prep.main()
    preprocessing_matches = equal_frames(items, read(prep.OUT_CSV))
    prep.OUT_CSV.unlink()

    lang = items['lang'].value_counts().rename_axis('lang').reset_index(name='lines')
    lang['share'] = (lang['lines']/lang['lines'].sum()).round(3)
    env = load_functions(repo/'src/data_analysis.py', ['fold','glossary_hit'])
    names = [env['fold'](n) for n in glossary['bosnian_name']]
    kept['in_glossary'] = kept['item_text'].apply(lambda x: env['glossary_hit'](x, names))
    per_menu = (kept.groupby(['menu_id','restaurant','split'])
        .agg(pages=('page','nunique'),items=('item_id','count'),
             with_description=('description',lambda s:(s!='').sum()),
             glossary_matches=('in_glossary','sum')).reset_index())

    pages = read(repo/'results/ocr_eval_pages.csv')
    metrics = ['cer_ordered','cer_bow','word_recall','diacritic_recall']
    pages[metrics] = pages[metrics].replace('', pd.NA).apply(pd.to_numeric, errors='raise')
    grouped = pages.groupby(['lang','preprocessing'],sort=False)[metrics].mean().round(3).reset_index()

    pilot = read(pilot_path)
    predictions = []
    tp = fp = fn = exact = assessed = missing = 0
    for row in pilot.to_dict('records'):
        gold = set(json.loads(row['expected_dishes']))
        selected = items[items['item_id'] == row['item_id']] if row['item_id'] else items.iloc[:0]
        present = len(selected) == 1 and str(selected.iloc[0]['keep']) == 'True'
        text = selected.iloc[0]['item_text'] if present else ''
        result = matcher(text,glossary,threshold=0.80) if present else None
        predicted = {result['dish']} if result else set()
        correct = int(predicted == gold) if present else ''
        if present:
            assessed += 1
            tp += len(predicted & gold)
            fp += len(predicted - gold)
            fn += len(gold - predicted)
            exact += correct
        else:
            missing += 1
        predictions.append({**row,'extraction_present':present,'ocr_item_text':text,
             'predicted_dishes':json.dumps(sorted(predicted),ensure_ascii=False),
             'similarity':result['score'] if result else '', 'recognition_exact':correct})
    pred = pd.DataFrame(predictions)
    pred.to_csv(out/'pilot_predictions.csv',index=False)
    failures = pred[(pred['extraction_present'] == False) | (pred['recognition_exact'] == 0)]
    failures.to_csv(out/'pilot_issues.csv',index=False)
    pilot_metrics = {'pilot_references':len(pilot),'extracted_items_assessed':assessed,
        'missing_extractions':missing,'exact_recognition_matches_on_present_items':exact,
        'tp':tp,'fp':fp,'fn':fn,'threshold':0.80,
        'limitation':'Purposive M3 pilot, assistant visual review pending team confirmation; not final held-out accuracy.'}
    (out/'pilot_metrics.json').write_text(json.dumps(pilot_metrics,indent=2),encoding='utf-8')

    checks = {
        'preprocessing_reproduced':preprocessing_matches,
        'language_table_reproduced':equal_frames(lang,read(repo/'results/language_distribution.csv')),
        'menu_table_reproduced':equal_frames(per_menu,read(repo/'results/items_per_menu.csv')),
        'ocr_summary_reproduced':equal_frames(grouped,read(repo/'results/ocr_eval_summary.csv')),
        'unique_item_ids':items['item_id'].is_unique,
        'unique_ocr_config_page_keys':not pages.duplicated(['menu_id','page','lang','preprocessing']).any()}
    checks = {k:bool(v) for k,v in checks.items()}
    (out/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    manifest = {'python_version':sys.version.split()[0],'pandas_version':pd.__version__,
        'ocr_rerun':False,'nllb_rerun':False,'text_preprocessing_rerun':True,
        'glossary_rows':len(glossary),'glossary_path':glossary_path.relative_to(repo).as_posix(),
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'pilot_sha256':hashlib.sha256(pilot_path.read_bytes()).hexdigest(),
        'split_manifest_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest(),
        'input_sha256':{p.relative_to(repo).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [module_path,repo/'src/data_analysis.py',repo/'src/evaluate_ocr.py',
                      repo/'Restaurant_Menu_Translation.ipynb',glossary_path,items_path,
                      repo/'data/interim/ocr_pages.csv',repo/'data/raw/menu_sources.csv',
                      repo/'results/ocr_eval_pages.csv',repo/'results/ocr_eval_summary.csv']}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    lines=['M3 REVIEW',f'All consistency checks passed: {all(checks.values())}.',
        f'Menu sources: {len(sources)}; processed rows: {len(items)}; retained candidate items: {len(kept)}.',
        f'Retained rows from test-designated menus: {(kept["split"]=="test").sum()}.',
        f'Current glossary entries: {len(glossary)} (filename ends in _10).',
        f'Saved OCR evaluation: {len(pages)} configuration/page rows; {len(pages[["menu_id","page"]].drop_duplicates())} distinct pages.',
        f'Pilot: {len(pilot)} PDF item references; {assessed} extracted; {missing} missing.',
        f'Matcher exact agreement on extracted pilot items: {exact}/{assessed}.',
        'The pilot uses assistant-reviewed labels pending team confirmation and is not final project accuracy.',
        'OCR and NLLB inference were not rerun; saved OCR summary was recomputed from page-level metrics.',
        'Original source tags are preserved; evaluation/split_manifest.csv defines six train, one validation and three test menus.']
    (out/'report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))
    if not all(checks.values()):
        raise SystemExit('A consistency check failed; inspect checks.json before submission.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out',type=Path,default=Path(__file__).with_name('m3_review_results'))
    parser.add_argument('--pilot',type=Path,default=Path(__file__).with_name('pilot_test_v1.csv'))
    args=parser.parse_args()
    run(args.repo.resolve(),args.out.resolve(),args.pilot.resolve())
