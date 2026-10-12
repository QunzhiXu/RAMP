"""Verify frozen inputs, UMAP rendering, ridge subsets and final PDF layout."""
import ast
import hashlib
import json
from pathlib import Path

import fitz
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / 'REVISION_PLAN.json').read_text())


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def functions(path):
    return {f.name: ast.dump(f, include_attributes=False)
            for f in ast.parse(path.read_text()).body if isinstance(f, ast.FunctionDef)}


reports = []
hash_cache = {}
for case in PLAN:
    old, new = Path(case['old']), Path(case['new'])
    before = json.loads((old / 'MANIFEST.json').read_text())
    after = json.loads((new / 'MANIFEST.json').read_text())
    cfg = json.loads((new / 'RIDGE_CONFIG.json').read_text())
    a = pd.read_csv(old / 'ridge_display_audit.csv')
    b = pd.read_csv(new / 'ridge_display_audit.csv')
    keys = ['method', 'marker', 'dataset', 'unit_key']
    assert not a.duplicated(keys).any() and not b.duplicated(keys).any()
    joined = b.merge(a, on=keys, how='left', suffixes=('_new', '_old'), validate='one_to_one', indicator=True)
    assert joined['_merge'].eq('both').all()
    preserved = ['n_cells', 'display_low', 'display_high', 'n_below', 'n_above', 'bins', 'sigma_bins', 'peak_height_normalized']
    for k in preserved:
        assert np.array_equal(joined[k + '_new'].to_numpy(), joined[k + '_old'].to_numpy()), (case['name'], k)
    assert len(b) == cfg['ridge_units'] * 18
    assert b.groupby(['method', 'marker']).n_cells.sum().eq(cfg['ridge_cells']).all()
    selected = pd.read_csv(new / 'ridge_unit_selection.csv')
    groups = [(p['dataset'], u, p['color']) for p in cfg['platforms'] for u in p['selected_unit_ids']]
    for _, frame in b.groupby(['method', 'marker']):
        got = list(frame.sort_values('top_to_bottom_index')[['dataset', 'unit_key', 'ridge_color']].itertuples(index=False, name=None))
        assert got == groups
    for p in cfg['platforms']:
        available = sorted(a.loc[a.dataset.eq(p['dataset']), 'unit_key'].unique())
        n = len(p['selected_unit_ids'])
        expected = sorted(sorted(available, key=lambda u: hashlib.sha256(f'20260922|{p["dataset"]}|{u}'.encode()).hexdigest())[:n])
        assert p['selected_unit_ids'] == expected
    # Rendering functions and settings for the UMAP contents must be unchanged.
    fa, fb = functions(old / 'build_figure.py'), functions(new / 'build_figure.py')
    for name in ['scatter', 'point_order', 'point_orders']:
        if name in fa:
            assert fa[name] == fb[name], (case['name'], name)
    umap_keys = ['cells', 'dataset_counts', 'display_cells', 'umap_protocols', 'dataset_palette', 'celltype_palette', 'draw_order']
    umap_keys += [k for k in before if k.startswith(('scatter_', 'background_', 'tumor_', 'non_joint_', 'coordinate_'))]
    for k in umap_keys:
        if k in before:
            assert before[k] == after[k], (case['name'], k)
    # Hash old outputs/scripts, and all inputs tracked by the new rendering.
    for path, expected in {**case['old_hashes'], **after['input_sha256']}.items():
        resolved = str(Path(path).resolve())
        if resolved not in hash_cache:
            hash_cache[resolved] = sha(path)
        assert hash_cache[resolved] == expected, path
    for path, expected in after['output_sha256'].items():
        assert sha(new / path) == expected
    assert after['script_sha256'] == sha(new / 'build_figure.py')
    pdf = new / 'output/pdf' / Path(case['png']).with_suffix('.pdf')
    doc = fitz.open(pdf)
    assert len(doc) == 1
    page = doc[0]
    assert abs(page.rect.width * 25.4 / 72 - 210) < .01
    assert abs(page.rect.height * 25.4 / 72 - 297) < .01
    words = page.get_text('words')
    assert words
    for x0, y0, x1, y1, *_ in words:
        assert 0 <= x0 <= x1 <= page.rect.width and 0 <= y0 <= y1 <= page.rect.height
    txt = page.get_text()
    for p in cfg['platforms']:
        assert f'{p["name"]} ({len(p["selected_unit_ids"])})' in txt
    rendered = new / 'tmp/pdfs'
    rendered.mkdir(parents=True, exist_ok=True)
    page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False).save(rendered / 'page_1.png')
    report = dict(case=case['name'], status='passed', original_files_unchanged=True,
                  tracked_inputs_unchanged=True, umap_contents_and_rendering_unchanged=True,
                  ridge_records=len(b), displayed_units=cfg['ridge_units'], displayed_cells=cfg['ridge_cells'],
                  same_selected_units_for_every_method_and_marker=True, selection_is_id_only=True,
                  old_display_ranges_cell_counts_and_smoothing_preserved=True,
                  legend_and_stack_order_consistent=True, pdf_pages=1, pdf_size_mm=[210, 297],
                  all_pdf_text_within_page=True, pdf_render=str(rendered / 'page_1.png'))
    (new / 'QA.json').write_text(json.dumps(report, indent=2) + '\n')
    reports.append(report)
    print('Verified:', case['name'], flush=True)
(HERE / 'QA_SUMMARY.json').write_text(json.dumps(reports, indent=2) + '\n')
