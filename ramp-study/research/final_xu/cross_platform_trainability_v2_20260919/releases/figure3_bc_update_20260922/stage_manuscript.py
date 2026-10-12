"""Prepare and check manuscript changes in the workspace before publication."""
from pathlib import Path
import hashlib
import json
import re
import shutil

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
SOURCE = ROOT.parent.parent / 'bioRxiv_Ver1'
STAGE = OUT / 'manuscript_stage'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(text, old, new):
    assert text.count(old) == 1, (old[:100],text.count(old))
    return text.replace(old,new)


def update_sections(text, latex=False):
    ref = r'Figure~\ref{fig:trainability}' if latex else 'Figure 3'
    start = text.index('For cell-type annotation, we evaluated 34 directed transfers')
    end = text.index('\nFor missing-marker prediction,',start)
    text = text[:start] + f'''For cell-type annotation, we evaluated 34 directed transfers across seven
benchmark configurations, using 500 source training cells and 512 source
validation cells. In the updated, task-specific classification scopes
(Methods), RAMP achieved mean balanced accuracy of 0.773 and macro-F1 of
0.751, compared with 0.584 and 0.520 without alignment ({ref}B).
The next highest means were 0.744 and 0.709 for clipped $z$-score preprocessing.
RAMP ranked first for both metrics in six of seven benchmarks and second in
HNSCC, and had the highest benchmark-mean scores in all three model families.
These comparisons use identical eligible labels and cells across methods;
they should not be interpreted as gains caused by restricting the class scope.
''' + text[end:]
    text=replace_once(text,'Spearman correlation from 0.279 to 0.350','Spearman correlation from 0.277 to 0.349')
    text=replace_once(text,'with correlation 0.360 and error 0.219','with correlation 0.359 and error 0.219')
    text=replace_once(text,
        'We then varied the amount of source supervision and neural-network capacity.',
        'Separately, the original-scope experiments varied source supervision and\nneural-network capacity. These curves retain the initial class sets and Lung\nreference pool, rather than the revised annotation scopes in panel B.')
    start=text.index('For '+ref+', we reused the frozen cells and shared-marker')
    end=text.index('\nWe compared ten pipelines:',start)
    pct=r'20\%' if latex else '20%'
    intro=f'''For {ref}, cells, markers and input transformations were recorded
in versioned task manifests. The updated Lung reference in B,C contains 84,000
cells: 28,000 each from Senosain CyTOF, Cords IMC and Zheng multiplexed
immunofluorescence, with 1,000 cells per registered patient. The Senosain cells
were resampled from the reviewed 140,000-cell annotation pool; Cords and Zheng
were unchanged. Patient identities and sample units were registered separately;
a tissue region was not treated as an independent patient. All methods used
the same unlabeled input pool within each experiment. Classification scope
restrictions affected supervised training, validation and target scoring,
but did not remove cells from the alignment pool. Target answers did not enter
alignment or prediction-model fitting and selection. Predictions were saved
with cell identities and checksums before target scoring.

The updated annotation comparison uses the following task-specific scopes:
bone marrow, CD4 T, CD8 T, B and monocyte; TNBC, CD4 T, CD8 T, myeloid, stroma
and tumor/epithelial; original three-platform melanoma, myeloid, stroma and
tumor; original lung benchmark, CD8 T, endothelial, myeloid and tumor/epithelial;
Lung story, CD4 T, CD8 T, myeloid/APC and tumor/epithelial; Melanoma story,
B lineage, CD4 T including Treg, and CD8 T; and HNSCC, B, CD4 T and CD8 T.
Cells outside these sets were excluded from supervised classification rather
than pooled into an ``other'' class. The scopes were revised after exploratory
review of the initial results, annotation compatibility, marker support and
training coverage; they were not prespecified before the initial benchmark.
All methods were evaluated with the same revised scopes and matched source
draws. The bone-marrow experiment was unchanged. Original full-scope scores
are retained with the source data. Shared classes omitted for insufficient
training coverage, such as CD8 T in the original three-platform melanoma
benchmark, are not claimed to be biologically absent or intrinsically
incomparable. D--F retain the initial class sets and original Lung reference
pool and are separate resource experiments, not continuations of the revised B
classifier. Missing-marker prediction uses all frozen cells without the
annotation-class restrictions.
'''
    text=text[:start]+intro+text[end:]
    text=replace_once(text,
        'identically across methods, without inspecting target performance.',
        'identically across methods, without inspecting target performance. These\nbudget and split rules are distinct from the exploratory class-scope revisions\ndescribed above.')
    text=replace_once(text,
        'Calibration is unit-specific and is not fitted from target measurements.',
        'Calibration is unit-specific and is not fitted from target measurements.\nCalibration cells can come from source validation patients, but their cell IDs\nare disjoint from the model-training and validation sets. Marker holdout begins\nat the frozen provider-processed matrix: upstream denoising, segmentation, QC\nand sampling were not rerun with the marker absent. The Senosain expression\nlayer is provider-normalized/denoised, rather than raw instrument output.')
    return text


def main():
    STAGE.mkdir(exist_ok=True)
    initial={}
    # Stage a compilable copy. Historical backups and build artifacts stay out.
    for p in SOURCE.iterdir():
        if p.is_file() and p.suffix in ['.tex','.bib','.bst','.md','.json']:
            shutil.copy2(p,STAGE/p.name)
            initial[p.name]=sha(p)
    for name in ['figure','supplementary_data']:
        shutil.copytree(SOURCE/name,STAGE/name,dirs_exist_ok=True)
        for p in (SOURCE/name).rglob('*'):
            if p.is_file(): initial[str(p.relative_to(SOURCE))]=sha(p)
    (OUT/'MANUSCRIPT_BEFORE_HASHES.json').write_text(json.dumps(initial,indent=2)+'\n')

    caption=(SOURCE/'FIGURE3_CAPTION.md').read_text()
    caption=replace_once(caption,
        'transfers, using 500 source training and 512 validation cells.',
        'transfers, using 500 source training and 512 validation cells. Training and\nscoring use the updated task-specific class scopes listed in Methods; these\nscopes were revised after exploratory review of the original benchmark.')
    caption=replace_once(caption,
        'calibration uses an additional 64 cells per unit (Methods).',
        'calibration uses an additional 64 cells per unit (Methods). The Lung input\nuses the updated Senosain reference; the other five marker-prediction benchmarks are unchanged.\nNo annotation-class filter is applied to this regression task.')
    caption=replace_once(caption,
        'conditions. Three random seeds were used, with support exclusions applied',
        'conditions. D--F retain the original classification scopes and Lung\nreference pool and therefore are separate from the revised panel B comparison.\nThree random seeds were used, with budget/split support exclusions applied')
    (STAGE/'FIGURE3_CAPTION.md').write_text(caption)
    (ROOT/'Figure/CAPTION.md').write_text(caption)
    tex=update_sections((SOURCE/'template.tex').read_text(),latex=True)
    tex=replace_once(tex,
        'improved mean cross-platform annotation macro-F1 from 0.376 to 0.526 across\nthree prediction-model families.',
        'improved mean cross-platform annotation macro-F1 from 0.520 to 0.751 across\nthree prediction-model families within the evaluated class sets.')
    body=caption.split('\n\n',1)[1].strip()
    body=re.sub(r'\*\*(.*?)\*\*',lambda m:r'\textbf{'+m.group(1)+'}',body)
    start=tex.index(r'\paperfig{figure3_cross_platform_learning}{%')
    end=tex.index('}{fig:trainability}',start)+len('}{fig:trainability}')
    tex=tex[:start]+r'\paperfig{figure3_cross_platform_learning}{%'+'\n'+body+'\n}{fig:trainability}'+tex[end:]
    (STAGE/'template.tex').write_text(tex)
    section=update_sections((SOURCE/'CROSS_PLATFORM_LEARNING_SECTION.md').read_text())
    (STAGE/'CROSS_PLATFORM_LEARNING_SECTION.md').write_text(section)
    shutil.copy2(ROOT/'Figure/Combined ABCDEF/figure_cross_platform_learning.pdf',
                 STAGE/'figure/figure3_cross_platform_learning.pdf')

    data=STAGE/'supplementary_data/trainability'
    orig=data/'original_v0_5_primary_model_endpoints.json'
    shutil.copy2(SOURCE/'supplementary_data/trainability/all_primary_model_endpoints.json',orig)
    provenance=json.loads((data/'SOURCE_MANIFEST.json').read_text())
    source_by_copy={r['copy']:r for r in provenance}
    def copy(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
        key=str(dst.relative_to(STAGE))
        source_by_copy[key]=dict(source=str(src),copy=key,sha256=sha(dst))
    for letter in ['B','C']:
        for name in ['per_benchmark_scores.csv','per_model_scores.csv','overall_means.csv','PROVENANCE.json']:
            copy(ROOT/f'Figure/Panel {letter}'/name,data/f'Panel_{letter}'/name)
    copy(OUT/'primary_means.json',data/'all_primary_model_endpoints.json')
    original_summary=Path((ROOT/'results/formal_v0_5/LATEST_SUMMARY.txt').read_text().strip())
    copy(original_summary/'primary_means.json',orig)
    for name in ['REPORT.json','J1_PER_BENCHMARK.csv','J1_PER_MODEL.csv','J1_OVERALL.csv',
                 'J2_PER_BENCHMARK.csv','J2_OVERALL.csv','LUNG_BEFORE_AFTER.csv','OVERALL_BEFORE_AFTER.csv']:
        copy(OUT/name,data/'update_20260922'/name)
    j1=ROOT/'releases/j1_scope_update_20260921_r3'
    j2=ROOT/'exploration_lung_new140k_j2_20260922'
    copy(j1/'REPORT.json',data/'protocol/UPDATED_J1_REPORT.json')
    copy(j2/'summary/CHECKS.json',data/'protocol/UPDATED_LUNG_J2_CHECKS.json')
    copy(j2/'INPUT_CONTRACT.json',data/'protocol/UPDATED_LUNG_J2_INPUT_CONTRACT.json')
    copy(j2/'summary/SCORES_PER_JOB.csv',data/'update_20260922/LUNG_J2_SCORES_PER_JOB.csv')
    (data/'SOURCE_MANIFEST.json').write_text(json.dumps(list(source_by_copy.values()),indent=2)+'\n')
    (data/'README.md').write_text('''# Figure 3 source data

Updated 2026-09-22. Panel B uses the accepted seven-benchmark J1 revision
`j1_scope_update_20260921_r3`; panel C incorporates the completed new Lung J2
run (4,860 prediction jobs and 90 fitted representations), with the other five
J2 tasks unchanged. The 79 benchmark-marker combinations are unchanged.

- `Panel_B/`, `Panel_C/`: current plot values, per-model endpoints and provenance.
- `Panel_D/`, `Panel_E/`, `Panel_F/`: unchanged original-v0.5 resource experiments,
  using the initial class sets and original Lung pool. They are not curves of
  the updated panel B classifiers. D/E use neural networks; F shows all three
  families over five three-dataset tasks and 13 target settings.
- `all_primary_model_endpoints.json`: current J1/J2 endpoints.
- `original_v0_5_primary_model_endpoints.json`: original full-scope endpoints.
- `update_20260922/`: current summaries, new Lung job scores and before/after tables.
- `protocol/`: original frozen protocol plus explicitly named updated J1 and
  Lung J2 records. `COMPLETION_REPORT.json` describes the original run only.
- `SOURCE_MANIFEST.json`: source paths and checksums for copied source records.

Annotation class scopes were revised after exploratory review and applied to
all methods equally; Methods lists the actual sets. J2 retains all frozen
cells without this classification filter. All ten pipelines and three model
families remain in B,C. Models and benchmarks receive equal aggregation weight.
Benchmark configurations share cohorts and are not independent patient repeats.
No models were refitted during figure/manuscript synchronization.
''')
    readme=(STAGE/'README.md').read_text()
    readme += '\nFigure 3 B/C and its Results, Methods, abstract and source tables were updated on 2026-09-22. D–F retain the original resource-experiment scope; see [update record](FIGURE3_UPDATE_20260922.md).\n'
    (STAGE/'README.md').write_text(readme)
    note='''# Figure 3 update — 2026-09-22

The Lung missing-marker rerun is complete: 4,860/4,860 jobs and 90/90
representations, including MaxFuse. Saved predictions were independently
rescored by the experiment's summary job. Summary checksums and all completion
records were verified before publication.

Panel B now uses the accepted seven-task annotation scopes: RAMP balanced
accuracy 0.772900, macro-F1 0.751133, accuracy 0.793252 (first overall for all
three). Scope revisions are exploratory and explicitly described in Methods;
the original scores remain available.

Panel C uses six tasks, 79 benchmark-marker combinations and 29,310 primary
prediction jobs in total, including the replacement Lung run. RAMP Spearman
0.349041 and relative-expression MAE 0.223443 rank second; percentile scores
0.359330 and 0.219107 rank first. All ten methods are displayed.

Updated: Figure 3 PDF, caption, corresponding Results and Methods, abstract
annotation numbers, companion section MD, source data and provenance.
D–F remain unchanged original-scope resource analyses; caption and Methods
state this explicitly. Other figures and their results are unchanged.

Previous manuscript and Figure 3 files are preserved under
`_backups/figure3_update_20260922/` when this staged version is published.
'''
    (STAGE/'FIGURE3_UPDATE_20260922.md').write_text(note)
    (OUT/'README.md').write_text(note+'\nCurrent plot data: `primary_means.json`; reproducible checks: `build_release.py`.\n')
    (STAGE/'REVISION_NOTES.md').write_text((STAGE/'REVISION_NOTES.md').read_text()+'\n\n'+note)
    print('Manuscript staged:',STAGE)


if __name__ == '__main__':
    main()
