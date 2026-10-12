"""Verify plotted values, summaries, PDF geometry and frozen source files."""
from pathlib import Path
import hashlib
import json

import fitz
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
QA = HERE / 'visual_qa'
QA.mkdir(exist_ok=True)

def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

manifest = json.loads((HERE / 'MANIFEST.json').read_text())
source = pd.read_csv(BASE / 'metric_scores_main23.csv', float_precision='round_trip')
keys = ['benchmark', 'method', 'metric']
plotted = pd.concat([pd.read_csv(HERE / f'{fig}_plotted_values.csv', float_precision='round_trip')
                     for fig in ['S8', 'S9']], ignore_index=True)
assert len(plotted) == 966 and not plotted.duplicated(keys).any()
assert plotted.metric.nunique() == 23
assert plotted.groupby(['metric', 'method']).size().eq(7).all()
merged = plotted.merge(source, on=keys, validate='one_to_one')
assert len(merged) == 966
np.testing.assert_array_equal(merged.score, merged['mean'])

pdf_checks = []
for fig in manifest['figures']:
    values = plotted[plotted.figure.eq(fig['figure'])]
    summary = pd.read_csv(HERE / f"{fig['figure']}_boxplot_summaries.csv", float_precision='round_trip')
    for row in summary.itertuples():
        a = values[values.metric.eq(row.metric) & values.method.eq(row.method)].score.to_numpy()
        np.testing.assert_allclose([row.mean, row.median, row.q1, row.q3],
                                   [a.mean(), np.median(a), np.quantile(a, .25), np.quantile(a, .75)],
                                   atol=1e-14, rtol=0)
    for panel in fig['panels']:
        scores = values[values.metric.eq(panel['metric'])].score
        lo, hi = panel['score_limits']
        assert lo < scores.min() <= scores.max() < hi
    path = HERE / 'output' / 'pdf' / f"{fig['stem']}.pdf"
    doc = fitz.open(path)
    assert len(doc) == 1
    page = doc[0]
    np.testing.assert_allclose([page.rect.width, page.rect.height],
                               np.array([210, 297]) * 72 / 25.4, atol=.01, rtol=0)
    words = page.get_text('words')
    assert words and all(page.rect.contains(fitz.Rect(*word[:4])) for word in words)
    content = page.get_text()
    assert fig['figure'] in content and 'Higher is better' in content
    for method in manifest['methods_order']:
        assert method in content
    page.get_pixmap(dpi=150).save(QA / f"{fig['stem']}_pdf.png")
    pdf_checks.append(dict(figure=fig['figure'], pages=1, a4=True,
                           all_text_inside_page=True, all_points_inside_axes=True))

source_hashes = dict(manifest['source_hashes'])
source_hashes.update(json.loads((BASE / 'DESIGN_MANIFEST.json').read_text())['source_sha256'])
assert all(sha(path) == expected for path, expected in source_hashes.items())
installed = json.loads((BASE.parent / 'submission_package_20260922' / 'INSTALL_VERIFY.json').read_text())
assert all(sha(Path(installed['live']) / path) == expected
           for path, expected in installed['installed_sha256'].items())

report = dict(scores_verified=966, metrics=23, benchmarks=7, methods=6,
              values_exactly_match_frozen_source=True, summaries_verified=True,
              source_hashes_unchanged=True, installed_manuscript_and_supplement_unchanged=True,
              pdf_checks=pdf_checks)
(HERE / 'BUILD_CHECKS.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
