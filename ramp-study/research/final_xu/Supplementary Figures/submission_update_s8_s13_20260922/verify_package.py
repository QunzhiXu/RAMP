"""Verify the staged supplement and exact frozen scores before installation."""
from pathlib import Path
import hashlib
import json
import re
import difflib
import fitz
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LIVE = ROOT.parent / 'bioRxiv_Ver1'
STAGE = HERE / 'bioRxiv_Ver1_stage'
F2 = HERE.parent / 'Figure 2 complete benchmark'
F3 = HERE.parent / 'Figure 3 model benchmark breakdown' / 'final_20260922'

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

before = json.loads((HERE / 'LIVE_BEFORE.json').read_text())
for name, expected in before.items():
    p = LIVE / name
    assert (sha(p) if p.exists() else None) == expected, ('Live file changed', name)
sources = json.loads((STAGE / 'SUPPLEMENTARY_FIGURE_SOURCES.json').read_text())
assert [s['number'] for s in sources] == [f'S{i}' for i in range(1, 14)]
for source in sources:
    assert sha(STAGE / source['destination']) == source['sha256']
    if int(source['number'][1:]) < 8:
        assert sha(STAGE / source['destination']) == sha(LIVE / source['destination'])
    else:
        assert sha(source['source']) == source['sha256']
    doc = fitz.open(STAGE / source['destination'])
    assert len(doc) == 1
    np.testing.assert_allclose([doc[0].rect.width, doc[0].rect.height],
                               np.array([210, 297]) * 72 / 25.4, atol=.02)

data = STAGE / 'supplementary_data/model_specific_figures'
for name, source in [('integration_main23_scores.csv', 'metric_scores_main23.csv'),
                     ('integration_main23_registry.csv', 'metric_registry_main23.csv')]:
    assert sha(data / name) == sha(F2 / source)
scores = pd.read_csv(data / 'integration_main23_scores.csv')
assert len(scores) == 966 and scores.metric.nunique() == 23
assert scores.benchmark.nunique() == 7 and scores.method.nunique() == 6
assert not scores.duplicated(['benchmark', 'metric', 'method']).any()

pointer = json.loads((ROOT / 'cross_platform_trainability_v2_20260919/CURRENT_FIGURE3_RESULTS.json').read_text())
release = Path(pointer['directory']) / 'primary_means.json'
assert sha(release) == pointer['primary_means_sha256']
reference = pd.read_json(release)
key = ['task', 'family', 'metric', 'method']
for filename, n, tasks in [('annotation_scores.csv', 420, 7), ('missing_marker_scores.csv', 360, 6)]:
    plotted = pd.read_csv(data / filename, float_precision='round_trip')
    assert len(plotted) == n and plotted.task.nunique() == tasks
    assert plotted.family.nunique() == 3 and plotted.method.nunique() == 10
    assert plotted.metric.nunique() == 2 and not plotted.duplicated(key).any()
    matched = plotted.merge(reference, on=key, validate='one_to_one', suffixes=('_plot', '_release'))
    assert len(matched) == n
    np.testing.assert_allclose(matched.score_plot, matched.score_release, atol=1e-14, rtol=0)
for number in range(10, 14):
    manifest = json.loads((F3 / f'S{number}_MANIFEST.json').read_text())
    assert manifest['all_points_visible'] and manifest['all_scores_match_frozen_release']
    for path, expected in manifest['source_sha256'].items():
        assert sha(path) == expected

supp = fitz.open(STAGE / 'supplementary_material.pdf')
assert len(supp) == 17
for page in supp:
    np.testing.assert_allclose([page.rect.width, page.rect.height],
                               np.array([210, 297]) * 72 / 25.4, atol=.02)
    assert all(page.rect.contains(fitz.Rect(*w[:4])) for w in page.get_text('words'))
for i in range(1, 14):
    assert f'Supplementary Figure S{i}' in supp[3+i].get_text()
    assert f'Supplementary Figure S{i}.' in (STAGE / 'SUPPLEMENTARY_CAPTIONS.md').read_text()
captions = (STAGE / 'supplementary_captions.tex').read_text()
assert captions.startswith((LIVE / 'supplementary_captions.tex').read_text().rstrip())

main_before = (LIVE / 'template.tex').read_text()
main_after = (STAGE / 'template.tex').read_text()
insertions = [
    '\nThe 23 primary integration metrics are shown individually in Supplementary\nFigures~S8 and S9.',
    '\nModel-specific annotation results for every benchmark are shown in\nSupplementary Figures~S10 and S11; missing-marker Spearman and MAE are\nshown separately in Supplementary Figures~S12 and S13.',
]
restored = main_after
for insertion in insertions:
    assert restored.count(insertion) == 1
    restored = restored.replace(insertion, '')
assert restored == main_before
(HERE / 'MANUSCRIPT_DIFF.txt').write_text(''.join(difflib.unified_diff(
    main_before.splitlines(True), main_after.splitlines(True), fromfile='previous/template.tex', tofile='updated/template.tex')))
old = fitz.open(LIVE / 'template.pdf')
new = fitz.open(STAGE / 'template.pdf')
assert len(new) == len(old) == 50
changed = [i+1 for i in range(len(new)) if old[i].get_text() != new[i].get_text()]
assert changed == [27, 35]
for p in (LIVE / 'figure').iterdir():
    if p.is_file():
        assert sha(p) == sha(STAGE / 'figure' / p.name)
for filename in ['supplementary_material.log', 'template.log']:
    log = (STAGE / filename).read_text()
    assert not re.search(r'Overfull|Missing character|undefined|LaTeX Warning', log)

targets = json.loads((HERE / 'INSTALL_FILES.json').read_text())
assert len(set(targets)) == len(targets)
assert all((STAGE / t).is_file() and not (STAGE / t).is_symlink() for t in targets)
checks = dict(status='passed', supplement_pages=17, legend_pages=4, figure_pages=13,
              main_pages=50, main_changed_pages=changed, original_s1_s7_unchanged=True,
              main_figure_assets_unchanged=True, integration_metrics=23, integration_scores=966,
              annotation_scores=420, missing_marker_scores=360, all_scores_match_frozen_sources=True,
              model_families=3, learning_methods=10, native_A4_portrait=True,
              no_undefined_references_or_overfull_boxes=True,
              visual_review_complete=True,
              reviewed_pages={'supplement':[1,2,3,4,12,13,14,15,16,17], 'main':[27,35]},
              visual_review_note='Rendered PDF pages inspected; headings, ticks, legends and captions are readable and unclipped.',
              staged_sha256={t:sha(STAGE/t) for t in targets})
(HERE / 'BUILD_CHECKS.json').write_text(json.dumps(checks, indent=2) + '\n')
print(json.dumps({k:v for k,v in checks.items() if k != 'staged_sha256'}, indent=2))
