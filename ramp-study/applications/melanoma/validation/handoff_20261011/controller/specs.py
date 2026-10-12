"""Fixed v8 numerical scope. No ROI scan, old threshold search or figure rendering."""
from common import STORY

A = STORY + '/analysis/all_cd8_prediction'
R = 'test0827/story_cd8_internal_validation/ramp230_clean15/distance_threshold_sweep/survival_sensitivity'
MOLD = 'raw_data/melanoma/moldoveanu_imc/Moldoveanu_2022_CyTOF_melanoma/Data'
ENG = 'raw_data/melanoma/engelbrecht_citeseq/raw'
ALIGN = STORY + '/analysis/alignment/output'
CODES = [STORY + '/inputs/cohort_package/build_story_sample.py',
         A + '/run_all_cd8_patient_heldout.py', A + '/build_all_cd8_citeseq_projection.py',
         A + '/analyze_all_cd8_biology.py', A + '/spatial_plotting.py',
         A + '/make_all_cd8_spatial_figure.py', STORY + '/01_build_three_row_figure.py',
         STORY + '/figure_layout_v8/prepare_visual_data.py', R + '/run_observed20_sensitivity.R']
INPUTS = [MOLD + '/ICI_meanIntensity.tsv.gz', MOLD + '/ST1_sample_info.txt',
          ENG + '/cell_metadata.csv', ENG + '/ADT_counts.csv',
          ALIGN + '/ramp_aligned.npy', ALIGN + '/cell_metadata.csv.gz', ALIGN + '/manifest.json',
          STORY + '/inputs/harmonized_display_labels.csv.gz',
          STORY + '/inputs/legacy_panel_d_projection.csv.gz']
STAGES = ['observed_inputs', 'observed_survival', 'heldout', 'projection', 'biology',
          'visual_metrics', 'display_umap', 'fixed_roi']
ACTIONS = {'heldout': (A + '/run_all_cd8_patient_heldout.py', '__main__', ()),
           'projection': (A + '/build_all_cd8_citeseq_projection.py', '__main__', ()),
           'biology': (A + '/analyze_all_cd8_biology.py', '__main__', ()),
           'visual_metrics': (STORY + '/figure_layout_v8/prepare_visual_data.py', '__main__', ()),
           'display_umap': (STORY + '/01_build_three_row_figure.py', 'prepare_alignment_display', ())}

def action(stage):
    return ACTIONS[stage]

def references(source):
    exact = [STORY + '/inputs/spatial_cell_metadata.csv.gz',
             STORY + '/inputs/observed_patient_level_features.csv',
             STORY + '/inputs/observed_survival_results.csv',
             STORY + '/inputs/cohort_package/moldoveanu_imc/locked_metadata.csv.gz',
             STORY + '/inputs/cohort_package/engelbrecht_citeseq/locked_metadata.csv.gz',
             STORY + '/cache/ramp230_clean15_display_umap.npy',
             STORY + '/cache/ramp230_clean15_display_metadata.csv.gz',
             STORY + '/cache/ramp230_clean15_display_manifest.json',
             A + '/patient_metrics.csv', A + '/outer_patient_predictions.csv.gz', A + '/summary.json',
             A + '/spatial_figure/manifest.json', R + '/survival_results_20_vs_30.json']
    for folder in (A + '/citeseq_projection', A + '/biological_interpretation', STORY + '/figure_layout_v8/data'):
        exact += [str(p.relative_to(source)) for p in sorted((source / folder).iterdir())
                  if p.is_file() and p.suffix in ('.csv', '.gz', '.npy', '.json', '.md')]
    return sorted(set(exact))
