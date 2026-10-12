"""Verify numerical, figure and manuscript consistency before installation."""
from pathlib import Path
import json
import hashlib
import re

import fitz
import numpy as np
import pandas as pd
from PIL import Image

B = Path(__file__).resolve().parent
F = B.parent
ROOT = B.parents[2]
C = ROOT / 'story_lung_HLA_ABC/Comparison_senosain140k_20260921'
M = B / 'manuscript_stage'


def sha(p):
    with p.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    assert json.loads((C / 'COMPLETE.json').read_text())['status'] == 'PASS'
    assert json.loads((B / 'selection/VERIFICATION.json').read_text())['status'] == 'PASS'
    assert json.loads((B / 'VERIFY.json').read_text())['inputs_unchanged']
    claims = json.loads((B / 'CHECKED_CLAIMS.json').read_text())
    current = pd.read_csv(B / 'selection/metric_ranks_7_benchmarks_all31.csv')
    supplementary = pd.read_csv(M / 'supplementary_data/all31_metric_scores_and_ranks.csv')
    pd.testing.assert_frame_equal(current, supplementary)
    assert sha(B / 'Combined/figure_2_lung_reference_update.pdf') == sha(M / 'figure/figure2_benchmark.pdf')
    before = json.loads((B / 'LIVE_BEFORE.json').read_text())
    unchanged_figures = {}
    for relative, digest in before.items():
        if relative.startswith('figure/') and relative != 'figure/figure2_benchmark.pdf':
            assert sha(M / relative) == digest, relative
            unchanged_figures[relative] = digest
    # Pixel equality confirms that the unchanged gating panels have not shifted.
    unchanged_panels = {}
    for letter in 'EF':
        old = F / 'louvain_ari_fastmnn_20260914' / f'Panel {letter}/panel_{letter.lower()}.png'
        new = B / f'Panel {letter}/panel_{letter.lower()}.png'
        a, b = np.asarray(Image.open(old)), np.asarray(Image.open(new))
        assert np.array_equal(a, b), f'Panel {letter} pixel difference'
        unchanged_panels[letter] = True
    text = (M / 'template.tex').read_text()
    # Every Table S1 cell must correspond to the newly computed sensitivity table.
    table = pd.read_csv(M / 'supplementary_data/rank_sensitivity.csv')
    combinations = [('Main23', 'category_equal'), ('Main23', 'metric_equal'),
                    ('Full31', 'category_equal'), ('Full31', 'metric_equal')]
    for method in table.method.unique():
        pattern = '^' + re.escape(method) + r' & ([0-9.]+) & ([0-9.]+) & ([0-9.]+) & ([0-9.]+) \\\\$'
        match = re.search(pattern, text, re.M)
        assert match, method
        expected = [f'{table[(table.method == method) & (table.selection == s) & (table.weighting == w)].mean_rank.item():.3f}'
                    for s, w in combinations]
        assert list(match.groups()) == expected
    log = (M / 'template.log').read_text(errors='replace')
    assert not re.search(r'Undefined control sequence|LaTeX Error|Emergency stop|Fatal error', log)
    assert 'There were undefined references' not in log
    doc = fitz.open(M / 'template.pdf')
    pages = []
    qa = B / 'visual_qa'
    qa.mkdir(exist_ok=True)
    for i, page in enumerate(doc):
        content = page.get_text()
        if re.search(r'Figure\s+2:', content) or re.search(r'Table\s+S1:', content):
            pages.append(i + 1)
    assert pages, 'Figure 2 / sensitivity table not found in rendered PDF'
    render_pages = sorted({n for p in pages for n in (p - 1, p, p + 1) if 1 <= n <= len(doc)})
    for p in render_pages:
        doc[p - 1].get_pixmap(matrix=fitz.Matrix(1.6, 1.6)).save(qa / f'manuscript_page_{p:02d}.png')
    result = {
        'status': 'PASS', 'numerical_and_caption_claims': claims,
        'gating_panels_pixel_identical': unchanged_panels,
        'other_figure_files_unchanged': unchanged_figures,
        'manuscript_pages': len(doc), 'figure_and_table_pages': pages,
        'rendered_pages_for_visual_review': render_pages,
        'output_sha256': {str(p.relative_to(B)): sha(p) for p in [
            B / 'Combined/figure_2_lung_reference_update.pdf', M / 'template.tex', M / 'template.pdf']},
        'visual_review_required_before_installation': True,
    }
    (B / 'RELEASE_VERIFICATION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
