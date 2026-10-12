"""Current PT74 figure's fixed numerical route; no threshold or ROI search."""
from common import EXP

def at(path): return EXP + '/' + path

CODES = list(map(at, [
 'survival_motivation/01_build_spatial_niche_scores.py',
 'survival_motivation/02_run_survival_and_plot.R',
 'biomarker_discovery/01_project_spatial_state.py',
 'biomarker_discovery/02_screen_biomarkers.py',
 'spatial_return_survival/01_return_fixed_program.py',
 'alignment/03_plot_umap.py',
 'big_figure/panel_E/group_fraction_scan_20260910/scan_group_fractions.py',
 'spatial_score_review_20260913/analyze_spatial_scores.py',
 'spatial_score_review_20260913/check_codex_phenotype.py']))
INPUTS = list(map(at, [
 'sampled_data/haist_codex/complete/HNSCCdiscovery_CODEX.csv',
 'sampled_data/haist_codex/complete/HNSCCdiscovery_clinmetadata.csv',
 'sampled_data/rahim_citeseq/complete/GSE212797_adata_ADT.h5ad',
 'sampled_data/rahim_citeseq/complete/GSE212797_adata.h5ad',
 'survival_motivation/analysis_plan.json',
 'alignment/input/raw_shared_markers.float64.npy',
 'alignment/input/cell_metadata.csv.gz',
 'alignment/input/selected_74_patients.csv',
 'alignment/input/markers.txt',
 'alignment/output/aligned_ramp_ver2.float32.npy',
 'alignment/output/alignment_manifest.json']))
ALIGN_FILES=['aligned_ramp_ver2.float32.npy']
STAGES=['spatial_scores','survival','projection','screen','return_state','molecular30','phenotype','fixed_roi','umap']

def action(stage):
    rel={'spatial_scores':CODES[0], 'projection':CODES[2], 'screen':CODES[3], 'umap':CODES[5]}
    return rel[stage], 'main', []

PROVENANCE=list(map(at,[
 'README.md','FIGURE_CAPTION.md','alignment/input/input_manifest.json',
 'biomarker_discovery/analysis_plan.json','spatial_return_survival/analysis_plan.json',
 'big_figure/polish_20260913/CURRENT_FIGURE.md',
 'big_figure/polish_20260913/CEF_D_colorbar_aligned_20260914/build_figure.py',
 'big_figure/polish_20260913/CEF_D_colorbar_aligned_20260914/LAYOUT_AUDIT.json',
 'big_figure/polish_20260913/CEF_D_colorbar_aligned_20260914/hnscc_CEF_D_colorbar_aligned.pdf',
 'big_figure/polish_20260913/larger_LAG3_spatial/panel_F/CAPTION.md',
 'spatial_score_review_20260913/roi_selection_20260913/RANKING_AUDIT.json']))

SPECIAL = {
 at('big_figure/panel_E/group_fraction_scan_20260910/group_counts.csv'):'molecular30/group_counts.csv',
 at('big_figure/panel_E/group_fraction_scan_20260910/donor_contrasts.csv'):'molecular30/donor_contrasts.csv',
 at('big_figure/panel_E/group_fraction_scan_20260910/equal_donor_contrasts.csv'):'molecular30/equal_donor_contrasts.csv',
 at('big_figure/panel_E/group_fraction_scan_20260910/cell_group_assignments.csv.gz'):'molecular30/cell_group_assignments.csv.gz',
 at('spatial_score_review_20260913/roi_selection_20260913/large_roi_cells.csv'):'fixed_roi/large_roi_cells.csv',
 'story_test_hnscc/final_return_exploration_20260910/target_all_aligned.npy':'rebuilt_phenotype/target_all_aligned.npy',
 at('big_figure/polish_20260913/horizontal_E_spatial_left/panel_D/plotted_donor_values.csv'):'display_values/panel_D.csv',
}

def references(source):
    paths=[]
    for folder in ['survival_motivation/output','biomarker_discovery/projection','biomarker_discovery/screen','alignment/umap']:
        for p in sorted((source/EXP/folder).iterdir()):
            if p.name=='validation_report.json' or 'transfer_program_umap' in p.name: continue
            if p.name.endswith(('.csv','.csv.gz','.json','.txt')): paths.append(str(p.relative_to(source)))
    paths += list(map(at,[
        'spatial_return_survival/output/haist74_pt_cd8_tigit_cd39_return.csv.gz',
        'spatial_return_survival/output/rahim_donor_source_audit.csv',
        'spatial_return_survival/output/return_manifest.json']))
    paths += [at('spatial_score_review_20260913/'+name) for name in [
        'codex_protein_patient_contrasts.csv','codex_protein_patient_correlations.csv',
        'codex_protein_contrast_summary.csv','codex_protein_correlation_summary.csv',
        'codex_measured_protein_cells.csv.gz']]
    return sorted(paths + list(SPECIAL))

def result_path(tree, rel):
    return tree/EXP/SPECIAL[rel] if rel in SPECIAL else tree/rel
