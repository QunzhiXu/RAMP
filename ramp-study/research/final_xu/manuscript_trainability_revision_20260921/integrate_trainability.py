"""Integrate the frozen trainability figure into a copied bioRxiv manuscript."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import difflib

HERE = Path(__file__).resolve().parent
WORK = HERE / "work/bioRxiv_Ver1"
ORIGINAL = Path("/Users/qxu6/Documents/New project/Biorxiv")
PROJECT = HERE.parent / "cross_platform_trainability_v2_20260919"

RESULTS = r"""\subsection{RAMP supports cross-platform predictive learning}\label{res-trainability}

We next asked whether alignment makes measurements from different platforms
more useful for training a shared prediction model. We evaluated cell-type
annotation and missing-marker prediction using linear models, $k$-nearest
neighbors and a single-hidden-layer neural network (Figure~\ref{fig:trainability}A).
RAMP was compared with five alignment methods, an assay-ready unaligned
baseline, and three simple preprocessing strategies: upper-clipped $z$-score,
min--max scaling and within-sample percentile transformation. The methods used
the same frozen cells and shared-marker information. Prediction models were
trained and selected using source-cohort supervision; target answers were
reserved for evaluation. The unlabeled target expression was included in
alignment, so this experiment evaluates prediction within jointly processed
cohorts rather than alignment of a previously unseen cohort (Methods).

For cell-type annotation, we evaluated 34 directed transfers across seven
benchmark configurations, using 500 source training cells and 512 source
validation cells. Averaged equally across model families and benchmarks, RAMP
achieved balanced accuracy of 0.605 and macro-F1 of 0.526, compared with 0.454
and 0.376 without alignment (Figure~\ref{fig:trainability}B). The next highest
means were 0.586 and 0.517 for fastMNN. RAMP had the highest mean balanced
accuracy in all three model families and the highest mean macro-F1 for the
linear and neural-network models; fastMNN was slightly higher for $k$-nearest
neighbors (0.521 versus 0.517).

For missing-marker prediction, each marker was removed from every platform's
inputs before rebuilding preprocessing and alignment. Models learned the
relationship between the remaining markers and withheld source measurements,
then predicted the same marker in the target cohort.
The evaluation covered all 79 eligible benchmark--marker combinations in six
benchmarks (Figure~\ref{fig:trainability}C). RAMP increased mean within-unit
Spearman correlation from 0.279 to 0.350 and reduced relative-expression mean
absolute error from 0.322 to 0.223. Percentile preprocessing gave the best means
for this task, with correlation 0.360 and error 0.219; RAMP ranked second for
both endpoints. Thus, alignment improved recovery relative to unaligned
inputs, while a simple rank-based transformation remained a strong baseline
for predicting a relative-expression target.

We then varied the amount of source supervision and neural-network capacity.
RAMP had the highest benchmark-mean macro-F1 at each of the four tested training
budgets (Figure~\ref{fig:trainability}D). With 100 training cells, it reached
0.480, compared with 0.381 for unaligned inputs with 2,000 training cells.
Both conditions additionally used 512 validation labels, giving total
supervision budgets of 612 and 2,512 cells. At a fixed training budget of 500
cells, RAMP also led at each tested hidden-layer width
(Figure~\ref{fig:trainability}E). An eight-unit network using RAMP reached
macro-F1 of 0.515, compared with 0.392 for a 128-unit network using unaligned
inputs. This comparison concerns downstream model capacity; the alignment
itself uses the same full unlabeled cell pool at every budget.

Finally, we compared training on one source cohort with training on two sources
to predict a third cohort. Across 13 eligible target settings in five
three-dataset benchmarks, total training and validation labels and their
per-class counts were matched across conditions. The single-source baseline
averages the two separately trained source models' scores. With RAMP, mean
macro-F1 increased from 0.656 to 0.675 for linear models, from 0.655 to 0.663
for $k$-nearest neighbors, and from 0.647 to 0.671 for neural networks
(Figure~\ref{fig:trainability}F). RAMP achieved the highest combined-source
endpoint in all three families, although other pipelines showed larger gains
from lower single-source baselines. Together, these experiments show that
RAMP supports cross-platform predictive models across several supervision
and capacity settings, complementing the marker-threshold evaluation above.
"""

CAPTION = r"""\textbf{RAMP supports cross-platform prediction across model families and supervision budgets.}
\textbf{(A)} Evaluation workflow. Unlabeled source and target expression is
processed by RAMP or nine control pipelines. Linear, $k$-nearest-neighbor and
neural-network models use source supervision for training and model selection,
then predict target cell types or withheld-marker expression. For marker
prediction, the marker is removed from all platforms before preprocessing and
alignment are refitted. Diagrams are schematic.
\textbf{(B)} Cell-type annotation across seven benchmarks and 34 directed
transfers, using 500 source training and 512 validation cells.
\textbf{(C)} Missing-marker prediction across six benchmarks and 79
benchmark--marker combinations. Spearman correlation is computed within
sample units; mean absolute error (MAE) compares predictions with a common
0--1 relative-expression target, not absolute protein abundance. Source
calibration uses an additional 64 cells per unit (Methods).
In B,C, each point is a benchmark score averaged equally across the three model
families. Open diamonds show benchmark-equal means; boxes show quartiles and
the median, and whiskers extend to the most extreme values within 1.5
interquartile ranges. All benchmark points are displayed.
\textbf{(D)} Neural-network annotation macro-F1 at 20, 100, 500 and 2,000
training cells, with hidden-layer width fixed at 32 and 512 additional
validation cells at every budget. The training-cell axis is logarithmic.
\textbf{(E)} Annotation performance at hidden-layer widths 8, 32 and 128, with
500 training and 512 validation cells. Width is shown on a logarithmic axis;
it is not the total number of model parameters.
\textbf{(F)} One- versus two-source training for a third target dataset, shown
separately for each model family across five benchmarks and 13 target settings.
Single source is the mean of the two single-source macro-F1 scores, not an
ensemble of predictions. Combined sources uses both sources with the same
total budget: 50 training and 20 validation labels per class, split equally
between sources. D--F show benchmark-equal mean curves; lines connect tested
conditions. Three random seeds were used, with support exclusions applied
before comparison. Unlabeled alignment pools are fixed across supervision
conditions. Benchmark configurations share some source cohorts; points and
curves are descriptive summaries, not independent-patient replicates.
"""

METHODS = r"""\subsection{Cross-platform learning experiments}\label{methods-trainability}

\paragraph{Data, information boundaries and preprocessing.}
For Figure~\ref{fig:trainability}, we reused the frozen cells and shared-marker
identities of the seven integration tasks and rebuilt inputs with explicit
records of their upstream transformations. Patient identities and sample units
were registered separately; a tissue region was not treated as an independent
patient. The eligible classification ontology comprised harmonized cell types
shared by the datasets in a task. Unlabeled preprocessing used the complete,
fixed cell pool for each task, including target expression. This pool was
unchanged when the number of labeled cells, model width or source-cohort
combination changed. Target cell-type labels and target withheld-marker
measurements did not enter alignment, training or hyperparameter selection. Predictions were saved with
cell identities and checksums before target scoring.

We compared ten pipelines: unaligned assay-ready input, RAMP 2.3.0,
Harmony-direct, cyCombine, CytoNorm, fastMNN, MaxFuse, upper-clipped $z$-score,
min--max and within-unit percentile transformation. The existing numerical
alignment implementations were retained, with dataset-specific input contracts
distinguishing source-scale input for RAMP and Harmony-direct from assay-ready
input for other methods. Assay-ready transformations included the registered
arcsinh, $\log(1+x)$ or shared-panel CLR operations and passthrough of
provider-processed values where appropriate. This comparison evaluates the
configured pipelines, including their preprocessing. Clipped $z$-scores cap
each marker at its within-unit 99th percentile and then standardize it within
each dataset. Min--max scaling maps each dataset--marker range to $[0,10]$.
Percentile transformation uses $(r-0.5)/n$ within each unit, where $r$ is the
average rank for ties and $n$ is the number of cells. Constant columns are set
to zero for the first two strategies and to 0.5 for percentile transformation.

FastMNN and MaxFuse contribute their corrected embeddings to the same
downstream prediction models; they are not required to return marker values.
Unlike the full-active-panel MaxFuse configuration used in
Figure~\ref{fig:benchmark}, this experiment uses the same shared-marker panel
for its shared and active feature spaces. Features are standardized within
dataset, and the first registered dataset is the fixed hub for three-view
integration, with dimensions limited by the retained marker rank. RAMP uses
its fixed parameters and seed 42. Method configurations, input contracts and
the frozen experiment profile accompany the source data.

\paragraph{Source supervision and prediction models.}
The primary single-source annotation and missing-marker comparisons use 500 source training
cells and 512 source validation cells. Approximately 20\% of source patients
are reserved for validation where multiple verified patients are available;
single-patient sources use disjoint cell sets. Where patient crosswalks are
unresolved, splits and scores are explicitly sample-unit descriptive. Cell
budgets are drawn as prefixes of seeded random permutations, independently of
cell-type labels for these two tasks. The same cells and candidate grids are
used across processing methods. Three seeds, 20260919, 20260920 and 20260921,
govern source sampling and prediction-model initialization. Infeasible
budgets or splits are registered before model comparison and excluded
identically across methods, without inspecting target performance.

Linear models are logistic regression for annotation and ridge regression for
marker prediction. Their inverse regularization strength $C$ or ridge penalty
$\alpha$ is selected from $\{0.1,1,10\}$. Nearest-neighbor models use
$k\in\{5,15,50\}$ and uniform or inverse-distance weights; $k$ is capped by
the training-cell count. Neural networks have one ReLU hidden layer, width 32
in the primary comparison, Adam learning rate in $\{0.001,0.01\}$ and
$L_2$ penalty in $\{0.001,1\}$. Training uses batches of at most 128 cells,
at most 100 epochs and early stopping after 15 epochs without improvement on
source validation. The best source-validation checkpoint is retained.
Each model family also compares unchanged features with a global
feature-wise standardization fitted only on source training cells and applied
unchanged to source validation and target cells. Hyperparameters are selected
by source-validation macro-F1 for classification and within-unit Spearman
correlation for regression; ties follow a fixed lexicographic ordering of the
effective configurations. No target answers select models or scaling.

\paragraph{Marker masking and relative-expression targets.}
One marker is held out at a time from every platform, including aliases of the
same physical marker. Input operations that depend on the marker panel are
recomputed after masking, and all representations, including RAMP matching and
transformation, are refitted using the remaining markers. Thus, full-panel
alignment caches do not supply information about the withheld marker. The
source retains its measured marker values only as supervised answers. Before
training/validation splitting, 64 additional cells per source unit are
reserved for calibration and kept separate from both sets. Their empirical
distribution converts source marker measurements to relative-expression
targets using a mid-distribution empirical cumulative distribution function.
Calibration is unit-specific and is not fitted from target measurements.

For scoring only, target truth is defined as $(r_i-0.5)/n_u$ from the measured
marker within target unit $u$. MAE is computed against this common target after
clipping predictions to $[0,1]$. Spearman correlation uses the un-clipped
predictions and measured values within units containing at least 30 cells and
nonconstant truth. Constant predictions receive effective correlation zero;
units with insufficient or constant truth support do not contribute to the
correlation but remain eligible for MAE. Units are averaged within each patient
and patients equally, with sample-unit summaries retained where patient
identity is unresolved. This task evaluates relative expression and ordering,
not recovery of absolute protein concentration. It covers 79 eligible
benchmark--marker combinations and 28 directed transfers across six tasks.
The original three-platform melanoma benchmark is excluded from strict marker
holdout because upstream marker dependencies in its distributed MIBI input
could not be resolved. This eligibility restriction applies to every method. A source split
with no varying marker truth for validation is likewise excluded uniformly.

\paragraph{Annotation budgets, capacity and source-cohort diversity.}
Annotation covers all 34 directed transfers across the seven tasks. Label
efficiency uses 20, 100, 500 or 2,000 training cells with width fixed at 32;
every point additionally requires the same 512 validation cells. Capacity
comparisons use widths 8, 32 and 128 with 500 training and 512 validation cells.
The full unlabeled preprocessing pool is unchanged. Width denotes the hidden
layer, and actual fitted head-parameter counts are recorded separately because
input dimensions and class counts vary. These counts do not include alignment
parameters or preprocessing costs.

Multi-source annotation uses five three-dataset tasks with 13 eligible target
settings. Two datasets serve as sources A and B and the third as target; A/B
names follow registration order, not performance. Each class contributes 50
training and 20 validation cells to every condition. A-only and B-only draw
the full budget from one source; A+B draws 25 training and ten validation cells
per class from each source. Combined-source cells are nested subsets of the
corresponding single-source draws, and no cell switches between training and
validation across conditions. The eligible target cells and unlabeled
alignment pool are fixed. For display, the arithmetic mean of the A-only and
B-only macro-F1 scores is computed within each model, processing method,
target setting and seed, then compared with A+B. This is a mean of scores,
not a prediction ensemble or selection of the better source.

\paragraph{Scoring, aggregation and source records.}
Classification uses a confusion matrix normalized by each target patient's
cell count and then averaged equally across patients, or across registered
sample units for unit-descriptive data. Macro-F1 and balanced accuracy are
computed from this matrix across the fixed task ontology; undefined class
ratios are assigned zero. Accuracy is retained in the accompanying tables.
Within a model family, results are averaged over available registered seeds,
then over held-out markers within direction for regression, then directions
within benchmark, and finally benchmarks equally. B,C additionally average
the three model families equally for each benchmark point; D,E use neural
networks and F shows model families separately. Resource curves retain matched
direction support across the tested levels. Boxplots and curves are
descriptive summaries over overlapping benchmark configurations; random seeds
and configurations are not independent-patient replicates. Per-model scores,
resource curves, source/target mappings, parameter counts, eligibility records
and input provenance are supplied under
\texttt{supplementary\_data/trainability/}.
"""

DISCUSSION = r"""Marker-level alignment and learned representations address different
requirements. The controlled learning experiments provide evidence for using
aligned measurements as inputs to predictive models: with matched source
supervision, RAMP improved cross-platform annotation, supported smaller
networks and fewer training labels, and produced the highest combined-source
classification endpoint across the three model families tested. The benefit
depended on the task. Percentile preprocessing performed slightly better for
missing-marker relative expression, showing that a common rank scale can be
effective without estimating RAMP's cross-platform correspondences.

These experiments evaluate preprocessing for conventional prediction models,
not the training or performance of a proteomic foundation model. They provide
a tractable test of whether measurements from different platforms can support
shared predictive rules. Extending this test to large-scale pretraining remains
an open question. The supervised targets used here are observed cell types
and within-unit relative marker expression; they do not establish a ground
truth for cross-platform absolute abundance. Training a model to return
quantitatively calibrated marker values would require an explicit calibration
target and independent validation of that scale. RAMP may provide a useful
input representation or constrained calibration component for such work, but
its aligned values should not themselves be treated as experimentally
established ground truth. Scaling correspondence estimation beyond the two or
three datasets aligned together here is a further requirement.
"""


def replace_once(text, old, new):
    assert text.count(old) == 1, old[:100]
    return text.replace(old, new, 1)


def markdown(text):
    text = re.sub(r"\\subsection\{([^}]+)\}\\label\{[^}]+\}", r"## \1", text)
    text = re.sub(r"\\paragraph\{([^}]+)\}", r"### \1", text)
    text = text.replace(r"Figure~\ref{fig:trainability}", "Figure 3")
    text = text.replace(r"Figure~\ref{fig:benchmark}", "Figure 2")
    text = re.sub(r"\\textbf\{([^}]+)\}", r"**\1**", text)
    text = re.sub(r"\\texttt\{([^}]+)\}", r"`\1`", text)
    text = text.replace(r"\_", "_").replace(r"\%", "%")
    return text.strip() + "\n"


def main():
    original = (ORIGINAL / "template.tex").read_text()
    text = original
    for old, new in [("figure3_cd19", "figure4_cd19"), ("figure4_melanoma", "figure5_melanoma"),
                     ("figure5_lung_hla", "figure6_lung_hla"), ("figure6_hnscc", "figure7_hnscc")]:
        text = replace_once(text, r"\paperfig{"+old+"}", r"\paperfig{"+new+"}")
    anchor = r"\subsection{Reference-based completion recovers B-cell-associated CD19 signal}\label{res-cd19}"
    figure = r"\paperfig{figure3_cross_platform_learning}{%" + "\n" + CAPTION + "}{fig:trainability}\n\n"
    text = replace_once(text, anchor, RESULTS + "\n" + figure + anchor)
    anchor = r"\subsection{Within-sample marker-order audit}"
    text = replace_once(text, anchor, METHODS + "\n" + anchor)
    begin = text.index("The two approaches are therefore complementary rather than competing")
    end = text.index("The four applications illustrate distinct ways", begin)
    text = text[:begin] + DISCUSSION + "\n" + text[end:]
    text = replace_once(text,
        "monotone transformations avoided. We combined RAMP preprocessing and alignment",
        "monotone transformations avoided. With matched source supervision, RAMP also\n"
        "improved mean cross-platform annotation macro-F1 from 0.376 to 0.526 across\n"
        "three prediction-model families. We combined RAMP preprocessing and alignment")
    text = replace_once(text,
        "balanced accuracy of fixed-threshold transfer from 0.648 to 0.869. We further\nuse the aligned measurements",
        "balanced accuracy of fixed-threshold transfer from 0.648 to 0.869. We then\n"
        "test how preprocessing affects cross-platform annotation, missing-marker\n"
        "prediction, label requirements, model capacity and multi-source training\n"
        "under matched supervision. We further use the aligned measurements")
    text = replace_once(text,
        "the full-metric sensitivity summaries make that choice assessable. Clinical",
        "the full-metric sensitivity summaries make that choice assessable. The\n"
        "learning experiments use fixed, jointly processed cohorts and source-only\n"
        "supervision, so they do not establish performance on a new target cohort\n"
        "without refitting the alignment. Training-label budgets also exclude the\n"
        "unlabeled data and computation used for preprocessing. Clinical")
    text = replace_once(text,
        "include the metric registry, all seven-task metric scores and alternative\nranking summaries.",
        "include the metric registry, all seven-task metric scores, alternative\n"
        "ranking summaries, and the per-benchmark and per-model prediction scores,\n"
        "learning curves and frozen protocol records underlying\nFigure~\\ref{fig:trainability}.")
    (WORK / "template.tex").write_text(text)
    (WORK / "FIGURE3_CAPTION.md").write_text("# Figure 3 caption\n\n" + markdown(CAPTION))
    (WORK / "CROSS_PLATFORM_LEARNING_SECTION.md").write_text(
        "# Cross-platform learning: manuscript additions\n\n"
        + markdown(RESULTS) + "\n" + markdown(METHODS)
        + "\n## Discussion: alignment and predictive learning\n\n" + markdown(DISCUSSION))
    (HERE / "manuscript_changes.diff").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), text.splitlines(keepends=True),
        fromfile="Biorxiv/template.tex", tofile="bioRxiv_Ver1/template.tex")))

    dest = WORK / "supplementary_data/trainability"
    dest.mkdir(exist_ok=True)
    source_records = []
    def copy(source, relative):
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_records.append(dict(source=str(source), copy=str(target.relative_to(WORK)),
            sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    for letter in "BCDEF":
        panel = PROJECT / "Figure" / f"Panel {letter}"
        for name in ["per_benchmark_scores.csv", "per_model_scores.csv", "overall_means.csv", "curve_means.csv",
                     "per_direction_scores.csv", "parameter_counts.csv", "paired_seed_scores.csv",
                     "source_arm_means.csv", "source_arm_per_benchmark_scores.csv",
                     "source_arm_per_direction_scores.csv", "source_target_mapping.csv", "PROVENANCE.json"]:
            if (panel/name).exists(): copy(panel/name, f"Panel_{letter}/{name}")
    for source, relative in [
        (PROJECT/"config/formal_v0_5/PROFILE.json", "protocol/PROFILE.json"),
        (PROJECT/"validation/formal_preflight_v0_5/EXCLUSIONS.json", "protocol/EXCLUSIONS.json"),
        (PROJECT/"validation/formal_preflight_v0_5/REPORT.json", "protocol/PREFLIGHT_REPORT.json"),
        (PROJECT/"data/real_inputs_v0_3_r2/SUMMARY.json", "protocol/INPUT_SUMMARY.json"),
    ]: copy(source, relative)
    summary = Path((PROJECT/"results/formal_v0_5/LATEST_SUMMARY.txt").read_text().strip())
    copy(summary/"REPORT.json", "protocol/COMPLETION_REPORT.json")
    copy(summary/"primary_means.json", "all_primary_model_endpoints.json")
    # Input receipts document allowable information and cohort identities.
    for task in sorted((PROJECT/"data/real_inputs_v0_3_r2").iterdir()):
        if task.is_dir():
            for p in task.glob("*.json"):
                copy(p, f"protocol/input_contracts/{task.name}/{p.name}")
    for task in sorted((PROJECT/"representations/formal_v0_5").iterdir()):
        if task.is_dir() and (task/"complete").exists():
            for method in sorted((task/"complete").iterdir()):
                if method.is_dir():
                    for name in ["COMPLETE.json", "runtime.json", "profile.json"]:
                        if (method/name).exists():
                            copy(method/name, f"protocol/representation_settings/{task.name}/{method.name}/{name}")
    (dest/"SOURCE_MANIFEST.json").write_text(json.dumps(source_records, indent=2)+"\n")
    (dest/"README.md").write_text("""# Figure 3 source data

All files are copied from the completed, frozen cross-platform trainability v0.5 experiment. No fitting, task selection or score recomputation was performed for manuscript integration.

- `Panel_B/`: annotation, seven benchmark configurations; per-model endpoints and three-model benchmark means.
- `Panel_C/`: missing-marker prediction, six configurations; relative-expression targets, all 79 eligible benchmark-marker combinations.
- `Panel_D/`: neural-network training-label curves, with 512 additional validation cells at each point.
- `Panel_E/`: hidden-width curves and actual fitted head-parameter counts.
- `Panel_F/`: three model families, five three-dataset configurations, 13 target settings. Single source is the mean of A-only and B-only scores, compared with A+B at matched total class budgets. Original arms and source/target identities are retained.
- `all_primary_model_endpoints.json`: all primary model/metric endpoints, including classification accuracy.
- `protocol/`: frozen profile, exclusions, completion report, input contracts and full-panel representation settings. Exclusions apply uniformly across preprocessing methods.
- `SOURCE_MANIFEST.json`: original paths and SHA-256 hashes.

Benchmark configurations can share cohorts. Boxes summarize benchmark scores, not independent patients. Figure 3 uses shared-panel MaxFuse and corrected embeddings from fastMNN, which differs from the full-panel/marker-reconstruction settings used for some earlier figures.
""")
    print(json.dumps(dict(manuscript=str(WORK/"template.tex"), source_files=len(source_records)), indent=2))


if __name__ == "__main__":
    main()
