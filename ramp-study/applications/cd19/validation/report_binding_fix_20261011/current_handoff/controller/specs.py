"""Fixed published CD19 numerical chain; final PDF rendering is separate."""
from common import STORY, EXP, MARKERS

CODES = [f'algorithm/ramp/{name}' for name in ('__init__.py', 'ramp.py', 'ramp_ver1.py', 'ramp_ver2.py')]
CODES += [f'{STORY}/{name}' for name in (
    'prediction/masked_marker_model.py', 'prediction/run_prediction_and_validation.py',
    'alignment/run_alignment.py', 'umap/plot_umap.py', 'figure/panel_B/run_panel_B_qc.py')]
CODES += [f'{EXP}/scripts/{name}.py' for name in (
    'runtime', 'cd3_nn_joint_validation', 'cd4_nn_umap_validation', 'all_marker_nn_validation',
    'draw_four_metric_accuracy', 'draw_donor_mean_accuracy', 'review_all_donor_marker_scatter')]
CODES += [f'{EXP}/real_spatial0913/scripts/project_h14.py']
INPUTS = [f'{STORY}/sampled_data/{slug}/source_input_15.csv.gz'
          for slug in ('bm_codex', 'triana_abseq', 'bm_cytof')]
INPUTS += [f'{STORY}/sampled_data/relabel/relabeled_metadata.csv.gz']
INPUTS += [f'{EXP}/real_spatial0913/{name}' for name in (
    'data/H14_full_source15.csv.gz', 'data/H14_full_metadata.csv.gz', 'SELECTION_PLAN.md')]


def marker_dir(marker):
    if marker not in MARKERS:
        raise ValueError('Unknown marker')
    if marker in ('CD3', 'CD4'):
        return f'{EXP}/{marker}_NN_UMAP_validation'
    return f'{EXP}/validation_subpanels/all_marker_validation/{marker}'


REF_DIRS = [f'{STORY}/prediction/output', f'{STORY}/alignment/output',
            f'{STORY}/umap/data', f'{STORY}/figure/panel_B/data']
REF_DIRS += [marker_dir(marker) for marker in MARKERS]
REF_DIRS += [f'{EXP}/validation_subpanels/{name}' for name in (
    'all_marker_validation', 'all_markers_four_metrics', 'donor_mean_four_metrics')]
REF_DIRS += [f'{EXP}/all_donor_marker_review0913/data',
             f'{EXP}/real_spatial0913/data', f'{EXP}/real_spatial0913/results']
STAGES = ['main_prediction'] + [f'holdout_{m}' for m in MARKERS]
STAGES += ['alignment', 'cd4_embedding', 'cd4_evaluation', 'cd3_embedding', 'cd3_evaluation',
           'marker_summary', 'four_metrics', 'donor_metrics', 'scatter_values', 'projection', 'umap', 'qc']


def action(stage):
    if stage == 'main_prediction':
        return f'{STORY}/prediction/run_prediction_and_validation.py', 'main', []
    if stage.startswith('holdout_'):
        marker = stage.removeprefix('holdout_')
        if marker not in MARKERS:
            raise ValueError('Unknown marker')
        if marker == 'CD3':
            return f'{EXP}/scripts/cd3_nn_joint_validation.py', 'train', []
        if marker == 'CD4':
            return f'{EXP}/scripts/cd4_nn_umap_validation.py', 'train', []
        return f'{EXP}/scripts/all_marker_nn_validation.py', 'run_worker', [marker]
    mapping = {
        'alignment': (f'{STORY}/alignment/run_alignment.py', 'main'),
        'cd3_embedding': (f'{EXP}/scripts/cd3_nn_joint_validation.py', 'embed'),
        'cd4_embedding': (f'{EXP}/scripts/cd4_nn_umap_validation.py', 'embed'),
        'cd3_evaluation': (f'{EXP}/scripts/cd3_nn_joint_validation.py', 'evaluate'),
        'cd4_evaluation': (f'{EXP}/scripts/cd4_nn_umap_validation.py', 'evaluate'),
        'marker_summary': (f'{EXP}/scripts/all_marker_nn_validation.py', 'summarize'),
        'four_metrics': (f'{EXP}/scripts/draw_four_metric_accuracy.py', '__main__'),
        'donor_metrics': (f'{EXP}/scripts/draw_donor_mean_accuracy.py', '__main__'),
        'scatter_values': (f'{EXP}/scripts/review_all_donor_marker_scatter.py', 'evaluate'),
        'projection': (f'{EXP}/real_spatial0913/scripts/project_h14.py', 'main'),
        'umap': (f'{STORY}/umap/plot_umap.py', 'main'),
        'qc': (f'{STORY}/figure/panel_B/run_panel_B_qc.py', 'main'),
    }
    script, function = mapping[stage]
    return script, function, []

