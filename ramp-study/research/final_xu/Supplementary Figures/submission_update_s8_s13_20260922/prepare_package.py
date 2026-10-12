"""Build a reviewable S1-S13 manuscript supplement before installation."""
from pathlib import Path
import hashlib
import json
import shutil
import pandas as pd

HERE = Path(__file__).resolve().parent
SUPP = HERE.parent
LIVE = SUPP.parent.parent / 'bioRxiv_Ver1'
STAGE = HERE / 'bioRxiv_Ver1_stage'
STAGE.mkdir(parents=True, exist_ok=True)
F2 = SUPP / 'Figure 2 complete benchmark' / 'preview_v1'
F3 = SUPP / 'Figure 3 model benchmark breakdown' / 'final_20260922'

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def tex(text):
    for a, b in [('&', r'\&'), ('%', r'\%'), ('_', r'\_'), ('#', r'\#')]:
        text = text.replace(a, b)
    return text

sources = json.loads((LIVE / 'SUPPLEMENTARY_FIGURE_SOURCES.json').read_text())
assert [s['number'] for s in sources] == [f'S{i}' for i in range(1, 8)]
new = [
    ('S8', 'Cell-type structure, label transfer and UMAP metrics', F2 / 'output/pdf/supplement_s8_structure_transfer_umap.pdf', 'figure/supplementary/figure_s8_structure_transfer_umap.pdf'),
    ('S9', 'Platform and donor mixing metrics', F2 / 'output/pdf/supplement_s9_platform_donor_mixing.pdf', 'figure/supplementary/figure_s9_platform_donor_mixing.pdf'),
    ('S10', 'Annotation: four original benchmarks', F3 / 'output/pdf/s10_annotation_by_model.pdf', 'figure/supplementary/figure_s10_annotation_original_benchmarks.pdf'),
    ('S11', 'Annotation: three application benchmarks', F3 / 'output/pdf/s11_annotation_by_model.pdf', 'figure/supplementary/figure_s11_annotation_application_benchmarks.pdf'),
    ('S12', 'Missing-marker Spearman across models and benchmarks', F3 / 'output/pdf/s12_missing_marker_spearman.pdf', 'figure/supplementary/figure_s12_missing_marker_spearman.pdf'),
    ('S13', 'Missing-marker MAE across models and benchmarks', F3 / 'output/pdf/s13_missing_marker_mae.pdf', 'figure/supplementary/figure_s13_missing_marker_mae.pdf'),
]
for s in sources:
    source = LIVE / s['destination']
    assert sha(source) == s['sha256']
    target = STAGE / s['destination']
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
for number, case, source, destination in new:
    target = STAGE / destination
    shutil.copy2(source, target)
    sources.append(dict(number=number, case=case, source=str(source), destination=destination, sha256=sha(source)))

captions = [
 ('S8', 'Cell-type structure, cross-platform label transfer and UMAP metrics',
  'Nine metrics from the 23-metric reporting set used in Figure 2 are shown in their original units. '
  '(A-E) Cell-type structure in the aligned representation. (F,G) Cross-platform label-transfer macro-F1 and balanced accuracy. '
  '(H,I) Cell-type silhouette and absolute dataset silhouette in UMAP. Each colored point represents one of seven benchmark scores, '
  'averaged over the 20 matched evaluation resamples. Boxes show the median and interquartile range across tasks; whiskers extend to '
  'the most extreme values within 1.5 interquartile ranges. All seven points are shown, including values beyond the whiskers. '
  'Open diamonds indicate benchmark-equal means, and arrows indicate the preferred score direction. Methods use matched evaluation '
  'cells within each task; tasks can share source cohorts.'),
 ('S9', 'Platform and donor mixing metrics',
  'All 14 mixing metrics from the Figure 2 reporting set are shown. (A1-A8) Leiden and Louvain cluster association with dataset '
  'and sample-unit labels, measured by ARI and NMI. (B1-B6) Normalized dataset and sample-unit iLISI, dataset entropy, '
  'cell-type-conditional kBET-like acceptance, and overall and same-cell-type cross-dataset neighbor fractions. Points, boxes, '
  'diamonds and arrows follow Figure S8. Sample units follow the benchmark metadata and can represent donors or acquisition units. '
  'The normalized iLISI and kBET-like implementations are defined in the main Methods. Figures S8 and S9 together contain exactly '
  'the 23 primary reporting metrics; no additional metrics enter these displays.'),
 ('S10', 'Model-specific annotation in the four original benchmarks',
  '(A) Bone marrow, (B) TNBC, (C) original three-platform melanoma (Melanoma 1), and (D) original lung cancer benchmark. '
  'Columns show linear, k-nearest-neighbor and neural-network classifiers. Filled circles indicate balanced accuracy and open '
  'triangles indicate macro-F1; vertical offsets separate the two markers without changing their scores. Each row contains all '
  'ten processing methods. RAMP is highlighted in green; axes and method order are shared. These are the model-specific '
  'components of Figure 3B, with no averaging across models, metrics or benchmarks.'),
 ('S11', 'Model-specific annotation in the three application benchmarks',
  '(A) Lung HLA-ABC, (B) melanoma IMC/CITE-seq (Melanoma 2), and (C) HNSCC. Models, symbols, axes and method order follow '
  'Figure S10. The Lung benchmark uses the updated Senosain reference. Figures S10 and S11 together report all seven '
  'annotation benchmarks, using the task-specific classification scopes described for Figure 3B in Methods.'),
 ('S12', 'Model-specific missing-marker prediction: Spearman correlation',
  '(A) Bone marrow, (B) TNBC, (C) original lung cancer benchmark, (D) Lung HLA-ABC, (E) melanoma IMC/CITE-seq and (F) HNSCC. '
  'Columns show linear, k-nearest-neighbor and neural-network regression. Each point is the frozen benchmark-level effective '
  'Spearman correlation for one processing method and model; higher values are better. Correlations are computed within eligible '
  'target sample units, with constant predictions assigned zero under the registered scoring rule. All six eligible benchmarks '
  'and ten methods are retained. The original three-platform melanoma benchmark was excluded from marker holdout because '
  'upstream marker dependencies in its distributed MIBI input could not be resolved.'),
 ('S13', 'Model-specific missing-marker prediction: relative-expression MAE',
  'Benchmark rows and model columns follow Figure S12. Mean absolute error compares predictions clipped to [0,1] with the '
  'within-unit relative-expression target defined in Methods; lower values are better. This endpoint measures error on the '
  'common relative-expression scale, not absolute protein concentration. Axes and method order are shared across panels. '
  'Figures S12 and S13 show the model-specific components of Figure 3C without averaging across models or benchmarks.'),
]
common = ('Figures S10-S13 use the frozen primary experiments with 500 source training cells and 512 source validation cells. '
          'Models are selected using source validation only. Classification scores retain the patient/sample-unit weighting '
          'specified in Methods. Within each model, registered seeds are averaged, then held-out markers within direction '
          'for regression, and then directed transfers within each benchmark. Annotation uses the revised classification scopes '
          'of Figure 3B; marker prediction retains all frozen cells without that class filter. For marker prediction, the held-out '
          'marker is removed from all platform inputs before representations are refitted. Different benchmarks can share source '
          'cohorts. These plots are descriptive breakdowns of existing results, not new training runs or independent-patient replicates.')
md = (LIVE / 'SUPPLEMENTARY_CAPTIONS.md').read_text().rstrip() + '\n\n'
body = (LIVE / 'supplementary_captions.tex').read_text().rstrip() + '\n\n'
for i, (number, title, content) in enumerate(captions):
    if i == 0:
        body += '\\newpage\n\\section*{Complete integration metrics}\n'
    if i == 2:
        body += '\\newpage\n\\section*{Model-specific cross-platform prediction}\n' + tex(common) + '\n\n'
        md += '## Common conventions for Figures S10-S13\n\n' + common + '\n\n'
    heading = f'Supplementary Figure {number}. {title}'
    md += f'## {heading}\n\n{content}\n\n'
    body += '\\subsection*{' + tex(heading) + '}\n' + tex(content) + '\n\n'
(STAGE / 'SUPPLEMENTARY_CAPTIONS.md').write_text(md)
(STAGE / 'supplementary_captions.tex').write_text(body)
document = (LIVE / 'supplementary_material.tex').read_text()
append = ''
for number, _, _, destination in new:
    append += '\\includepdf[pages=1,fitpaper=true,noautoscale=true,scale=1,pagecommand={\\suppHeader{' + number + '}}]{' + destination + '}\n'
document = document.replace('\\end{document}', append + '\\end{document}')
(STAGE / 'supplementary_material.tex').write_text(document)
(STAGE / 'SUPPLEMENTARY_FIGURE_SOURCES.json').write_text(json.dumps(sources, indent=2) + '\n')

# Exact plotted source data, without changing existing tables.
data_dir = STAGE / 'supplementary_data' / 'model_specific_figures'
data_dir.mkdir(parents=True, exist_ok=True)
annotation = pd.concat([pd.read_csv(F3 / f'S{i}_plotted_values.csv', float_precision='round_trip') for i in [10, 11]], ignore_index=True)
missing = pd.concat([pd.read_csv(F3 / f'S{i}_plotted_values.csv', float_precision='round_trip') for i in [12, 13]], ignore_index=True)
keys = ['task', 'family', 'metric', 'method']
assert len(annotation) == 420 and len(missing) == 360
assert not annotation.duplicated(keys).any() and not missing.duplicated(keys).any()
annotation[keys + ['score']].to_csv(data_dir / 'annotation_scores.csv', index=False)
missing[keys + ['score']].to_csv(data_dir / 'missing_marker_scores.csv', index=False)
shutil.copy2(F2.parent / 'metric_scores_main23.csv', data_dir / 'integration_main23_scores.csv')
shutil.copy2(F2.parent / 'metric_registry_main23.csv', data_dir / 'integration_main23_registry.csv')
(data_dir / 'README.md').write_text('# Supplementary Figures S8-S13 source data\n\n'
    'S8-S9: 966 frozen integration scores, exactly the 23 primary reporting metrics, seven tasks and six methods.\n\n'
    'S10-S11: 420 annotation scores, seven benchmarks, three models, ten processing methods and two metrics.\n\n'
    'S12-S13: 360 missing-marker scores, six benchmarks, three models, ten processing methods and two metrics.\n\n'
    'No model or benchmark averaging is applied in the prediction files. Within-benchmark aggregation follows Figure 3 Methods. '
    'Each CSV stores full-precision plotted scores; figures are descriptive and do not introduce new independent replicates.\n')
provenance = {f'S{i}': json.loads((F3 / f'S{i}_MANIFEST.json').read_text()) for i in range(10, 14)}
provenance.update({'S8_S9': json.loads((F2 / 'MANIFEST.json').read_text())})
(data_dir / 'SOURCE_MANIFEST.json').write_text(json.dumps(provenance, indent=2) + '\n')

# Two small textual pointers in Methods; main-figure assets are untouched.
template = (LIVE / 'template.tex').read_text()
needle = 'and shared plotting conventions are given in the separate supplementary file.'
assert template.count(needle) == 1
template = template.replace(needle, needle + '\nThe 23 primary integration metrics are shown individually in Supplementary\nFigures~S8 and S9.')
needle = '\\texttt{supplementary\\_data/trainability/}.'
assert template.count(needle) == 1
template = template.replace(needle, needle + '\nModel-specific annotation results for every benchmark are shown in\nSupplementary Figures~S10 and S11; missing-marker Spearman and MAE are\nshown separately in Supplementary Figures~S12 and S13.')
(STAGE / 'template.tex').write_text(template)
for filename in ['bibliography.bib', 'ramp_harvard.bst', 'agsm.bst']:
    shutil.copy2(LIVE / filename, STAGE / filename)
for p in (LIVE / 'figure').iterdir():
    if p.is_file():
        q = STAGE / 'figure' / p.name
        if not q.exists():
            q.symlink_to(p.resolve())
readme = (LIVE / 'README.md').read_text()
readme = readme.replace('Figures S1-S7 and concise legends', 'Figures S1-S13 and concise legends')
readme = readme.replace('The seven source figures remain', 'The source figures remain')
readme += ('\n\n2026-09-22 supplementary update: S8-S9 expand the 23 integration metrics; S10-S11 '
           'show annotation by model and benchmark; S12-S13 show missing-marker Spearman and MAE separately. '
           'All figure pages are A4 portrait and are included at their native size. Exact plotted scores and provenance are in '
           '`supplementary_data/model_specific_figures/`. Main figure assets are unchanged.\n')
(STAGE / 'README.md').write_text(readme)
core = ['template.tex', 'template.pdf', 'README.md', 'supplementary_material.tex', 'supplementary_material.pdf',
        'supplementary_captions.tex', 'SUPPLEMENTARY_CAPTIONS.md', 'SUPPLEMENTARY_FIGURE_SOURCES.json']
targets = core + [s['destination'] for s in sources[7:]] + [str(p.relative_to(STAGE)) for p in data_dir.iterdir() if p.is_file()]
before = {name: sha(LIVE / name) if (LIVE / name).exists() else None for name in targets}
(HERE / 'LIVE_BEFORE.json').write_text(json.dumps(before, indent=2) + '\n')
(HERE / 'INSTALL_FILES.json').write_text(json.dumps(targets, indent=2) + '\n')
(HERE / 'CAPTION_RECORDS.json').write_text(json.dumps(dict(common_learning=common, captions=captions), indent=2) + '\n')
print(STAGE)
