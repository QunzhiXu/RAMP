"""Current Senosain140k release only; fixed molecular and DFS endpoints."""
from common import STORY, EXP

SCRIPTS = EXP + '/scripts'
ANN = 'raw_data/lung_cancer/Senosain_2025_CyTOF_and_omics/annotation'
CODES = [SCRIPTS + '/' + n for n in ('runtime.py', 'align.py', 'project.py', 'frozen_maps.py',
    'analyze.py', 'survival.R', 'statistics_helpers.py', 'mechanism.py', 'umaps.py')]
CODES += [EXP + '/vendor/ramp/' + n for n in ('__init__.py', 'ramp.py', 'ramp_ver2.py')]
ALIGN_FILES = ['baseline_SHARED9.npy', 'RAMP_SHARED9.npy', 'REFERENCE_PROFILES.npz', 'FROZEN_PLATFORM_MAPS.npz']
INPUTS = [ANN + '/' + n for n in ('senosain_140k_cell_metadata.csv.gz',
    'rough_annotation/senosain_140k_final_cell_annotations.csv.gz', 'senosain_140k_raw35.npy', 'senosain_140k_markers35.csv')]
INPUTS += ['cross_platform_trainability_v2_20260919/reports/senosain_new_annotation_review_20260921/CANDIDATE_REFERENCE_IDS_REVIEW_ONLY.csv.gz']
INPUTS += [STORY + '/' + n for n in (
    'data_umap/data/CELL_METADATA.csv.gz', 'data_umap/data/RAW_SHARED9.npy', 'data_umap/data/DESIGN.json',
    'data_umap/data/SELECTED_DONORS.csv', 'data_umap/reference/baseline_SHARED9.npy',
    'data_umap/reference/RAMP_SHARED9.npy', 'data_umap/reference/RAMP_UMAP.json', 'data_umap/reference/RAMP_UMAP.npy',
    'data_cords/data/CELL_METADATA.csv.gz', 'data_cords/data/RAW_SHARED9.npy', 'data_cords/data/baseline_SHARED9.npy',
    'validation/data/ZHENG_METADATA.csv.gz', 'validation/data/ZHENG_SHARED9.npy',
    'validation/results/PREDICTIONS_AND_TRUTH.csv.gz', 'validation/results/SUMMARY_METRICS.csv',
    'data_cords/reference/THRESHOLD_CAF_CELL_IDS.csv.gz', 'survival/data/CLINICAL_INPUTS.csv',
    'survival/results/PATIENT_INPUTS.csv', 'data_cords/results/ALL_PATIENT_SCORES.csv',
    'mechanism_support/results/VCAF_MEASURED_AND_PREDICTED.csv.gz',
    'mechanism_support/patient_immune/results/PATIENT_IMMUNE_COMPOSITION.csv',
    'mechanism_support/patient_immune/results/PATIENT_LINEAGE_MARKERS.csv',
    'mechanism_support/results/PATIENT_ASSOCIATIONS.csv',
    'mechanism_support/patient_immune/results/COMPOSITION_ADJUSTED.csv',
    'mechanism_support/patient_immune/results/ADJUSTED_COMPARISONS.csv')]
INPUTS += [EXP + '/alignment/' + n for n in ALIGN_FILES]
PROVENANCE = [STORY + '/' + n for n in ('CURRENT_ANALYSIS.json', 'README.md', 'FIGURE_CAPTION.md',
    'update_20260921/scripts/promote.py', 'data_cords/scripts/project.py',
    'data_cords/scripts/prepare_reference.py', 'data_cords/reference/IMPORT_MANIFEST.json',
    'validation/scripts/predict.py', 'survival/scripts/prepare_survival.py',
    'mechanism_support/scripts/prepare.py', 'mechanism_support/scripts/export_measurements.R',
    'mechanism_support/patient_immune/scripts/prepare.py', 'mechanism_support/patient_immune/scripts/analyze.py',
    'mechanism_support/patient_immune/scripts/runtime.py')]
PROVENANCE += [EXP + '/' + n for n in ('README_ZH.md', 'PLAN_ZH.md', 'RESULTS_ZH.md', 'run.sh', 'RELEASE_MANIFEST.json')]
STAGES = ['reference_inputs', 'project_validation', 'project_cords', 'validation_statistics',
          'molecular_scores', 'survival', 'mechanism', 'umaps']
ACTIONS = {'project_validation': (SCRIPTS + '/project.py', 'main', ('validation',)),
    'project_cords': (SCRIPTS + '/project.py', 'main', ('cords',)),
    'validation_statistics': (SCRIPTS + '/analyze.py', 'validation', ()),
    'molecular_scores': (SCRIPTS + '/analyze.py', 'scores', ()),
    'mechanism': (SCRIPTS + '/mechanism.py', 'main', ()), 'umaps': (SCRIPTS + '/umaps.py', 'main', ())}

def action(stage): return ACTIONS[stage]

def references(source):
    rels = []
    for folder in ('inputs', 'validation', 'cords', 'analysis'):
        rels += [str(p.relative_to(source)) for p in sorted((source / EXP / folder).rglob('*'))
                 if p.is_file() and p.suffix in ('.json', '.csv', '.gz', '.npy', '.npz', '.txt')]
    rels += [EXP + '/alignment/' + n for n in ('baseline_UMAP.npy', 'RAMP_UMAP.npy', 'UMAP_COMPLETE.json')]
    return sorted(rels)
