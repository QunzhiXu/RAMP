"""Stage Figure 2 and minimally related text before writing the manuscript."""
from pathlib import Path
import sys, json, hashlib, shutil

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
MANUSCRIPT = Path('/Users/qxu6/Documents/New project/manuscript')
sys.path.insert(0, str(BASE))
import common
import build_all


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(text, old, new):
    assert text.count(old) == 1, old[:120]
    return text.replace(old, new)


def main():
    snapshot = HERE / 'original_manuscript_hashes.json'
    if not snapshot.exists():
        originals = [p for p in MANUSCRIPT.iterdir() if p.is_file()]
        originals += [p for p in (MANUSCRIPT/'figure').iterdir() if p.is_file()]
        snapshot.write_text(json.dumps({str(p): sha(p) for p in originals}, indent=2)+'\n')
    original = json.loads(snapshot.read_text())
    assert all(sha(p) == h for p, h in original.items()), 'Manuscript changed since staging started'

    common.setup()
    common.plt.rcParams['pdf.fonttype'] = 42
    common.save_data = lambda *args, **kwargs: None
    manifest = json.loads((BASE/'Combined/manifest.json').read_text())
    fig = common.plt.figure(figsize=(manifest['width_mm']/25.4, manifest['height_mm']/25.4), facecolor='white')
    for letter in 'ABCDEF':
        build_all.load(letter).draw(fig, manifest['positions_mm'][letter])
    common.case_legend(fig, 121)
    fig.savefig(HERE/'figure2_benchmark.pdf', dpi=360)
    common.plt.close(fig)
    assert all(sha(p) == h for p, h in common.READS.items())

    stage = HERE/'staged'
    stage.mkdir(exist_ok=True)
    figures = stage/'figure'
    figures.mkdir(exist_ok=True)
    for source in (MANUSCRIPT/'figure').iterdir():
        target = figures/source.name
        if source.is_file() and not target.exists():
            shutil.copy2(source, target)
    shutil.copy2(HERE/'figure2_benchmark.pdf', figures/'figure2_benchmark.pdf')
    shutil.copy2(BASE/'Combined/figure_2_seven_benchmarks_main23.png', figures/'figure2_benchmark.png')
    shutil.copy2(BASE/'Combined/figure_2_seven_benchmarks_main23.svg', figures/'figure2_benchmark.svg')

    caption = (HERE/'figure2_caption.tex').read_text().strip()
    for stem in ['manuscript', 'manuscript_figure1_polished']:
        text = (MANUSCRIPT/f'{stem}.tex').read_text()
        start = text.index('% ---------------------------------------------------------------- Fig 2')
        end = text.index('% ---------------------------------------------------------------- Fig 3', start)
        block = (r'''% ---------------------------------------------------------------- Fig 2
\begin{center}
\includegraphics[width=\textwidth]{figure2_benchmark.pdf}
\end{center}
'''+caption+'\n\\label{fig:benchmark}\n\\clearpage\n\n\n')
        text = text[:start]+block+text[end:]
        text = replace_once(text,
            'The complete metric battery\nacross all four benchmarks is given in Figure~\\ref{fig:benchmark}.',
            'An expanded seven-task performance comparison and marker-gate\ntransfer evaluation are given in Figure~\\ref{fig:benchmark}.')
        marker = r'\subsection*{Comparison methods}'
        extra = r'''Figure~\ref{fig:benchmark} additionally reuses the frozen lung, melanoma
and HNSCC application-cohort alignments as three integration tasks. The lung
story combines Senosain CyTOF, Cords IMC and Zheng mIF; the melanoma story
combines Moldoveanu IMC and Engelbrecht CITE-seq protein measurements; HNSCC
combines Rahim CITE-seq protein measurements and Haist CODEX. Application
sampling is described below. Each comparison method uses the corresponding
frozen RAMP cell cohort. These seven tasks can share source datasets and
are not seven independent patient cohorts.

'''
        text = replace_once(text, marker, extra+marker)
        text = replace_once(text,
            'MaxFuse contributes three hub\norientations, combined by their median within each resample before method-level\naggregation.',
            'MaxFuse uses shared proteins matching the respective RAMP inputs and\nits saved full active protein panel. Three-platform tasks use three hub\norientations, combined by their median within each resample and metric;\ntwo-platform tasks use the saved pairwise configuration.')
        old_start = text.index('Within each benchmark, methods were ranked per metric according to the', text.index(r'\subsection*{Evaluation}'))
        old_end = text.index(r'\subsection*{Within-sample marker-order audit}', old_start)
        evaluation = r'''For the primary summary in Figure~\ref{fig:benchmark}, 23 retained metrics
are grouped into cell-type structure (5), platform/donor mixing (14),
cross-platform label transfer (2) and UMAP (2). This reporting subset and
weighting were selected after inspection of the full results and applied
identically to every method and task; all 31 original metrics and an
equal-individual-metric sensitivity summary are retained. For each task,
six methods are ranked per metric according to its direction, with average
ranks for ties. Panel D averages those ranks within category, then gives each
of four categories 25\% weight, and finally averages the seven task scores
equally. Category means are not ranked again before aggregation. Panel C
instead scales direction-adjusted raw scores to the observed six-method
minimum and maximum within each task and metric, then averages metrics within
category and tasks equally; an all-equal comparison receives 1 for every
method. Panels A and B display the original metric means. All seven tasks use
20 matched resamples from frozen alignments, not independently refitted models.

\paragraph{Fixed-source marker thresholds.}
Figure~\ref{fig:benchmark}E,F retains the original four-benchmark
value-evaluation scope. Eligible units contain at least ten reference-positive
and ten reference-negative cells for a marker task, and source calibration
requires at least 50 cells per class. Each method calibrates one source-platform
threshold to maximize balanced accuracy, weighting classes equally and units
equally within each class; ties favor higher specificity. The threshold is
frozen and applied to each other platform, with no target-label threshold
selection. CD3 contrasts T with definite non-T cells; CD4 contrasts CD4 T with
CD8 T; CD8 contrasts CD8 T with annotated CD4/CD8-negative T cells.
Ambiguous and non-joint labels are excluded. Target-unit BA is averaged within
direction, then across six directions per benchmark--marker task. For the
overall score, tasks are averaged within benchmark and four benchmarks are
weighted equally. All 11 eligible tasks are shown. Raw is an independent
baseline; marker-gate evaluation is inapplicable to the frozen fastMNN and
MaxFuse embeddings. This is threshold transfer on jointly aligned frozen
data, not a held-out alignment-model evaluation. Illustrative densities use
asinh values (cofactor 1) and Gaussian KDE with bandwidth factor 0.22,
balancing classes and donors; source and target share plotting ranges within
method and marker. No task-level significance tests are reported.

'''
        text = text[:old_start]+evaluation+text[old_end:]
        text = replace_once(text,
            'Display refinements in\nFigure~\\ref{fig:benchmark}B and C alter point rendering and layer order only\nand are applied identically to all methods within a panel.',
            'UMAP display refinements alter point rendering and layer order only\nand are applied identically to all methods within a panel.')
        (stage/f'{stem}.tex').write_text(text)
    (HERE/'STAGING_MANIFEST.json').write_text(json.dumps({
        'status': 'staged', 'native_figure_mm': [180,126], 'text_width_mm': 170.18,
        'figure_scale': 170.18/180,
        'changes': ['Figure 2 PNG/SVG/vector PDF', 'Figure 2 caption',
                    'Direct Figure 1-to-Figure 2 reference', 'Figure 2 benchmark scope, method and evaluation text'],
        'outside_scope': 'No abstract rewrite or full manuscript audit; all other figure assets preserved.',
        'original_hashes': str(snapshot), 'input_sha256': common.READS}, indent=2)+'\n')
    print(stage)


if __name__ == '__main__':
    main()
