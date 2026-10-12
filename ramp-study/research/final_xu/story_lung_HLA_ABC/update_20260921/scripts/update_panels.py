from pathlib import Path
import shutil,json
import pandas as pd
S=Path(__file__).resolve().parents[2];R=S/'releases/senosain140k_20260921';F=S/'Figure'
def edit(p,a,b):
 s=p.read_text();assert a in s,a;p.write_text(s.replace(a,b))
p=F/'Panel B/scripts/draw_panel_B.py'
edit(p,"'Other / uncertain': 'Others',","'Other / uncertain': 'Others',\n    'Other / mixed': 'Others',")
edit(p,"'umap/results/RAMP_UMAP.npy'","'releases/senosain140k_20260921/alignment/RAMP_UMAP.npy'")
edit(p,"'data_umap/data/CELL_METADATA.csv.gz'","'releases/senosain140k_20260921/inputs/CELL_METADATA.csv.gz'")
edit(p,"'umap/results/RAMP_UMAP.json'","'releases/senosain140k_20260921/alignment/UMAP_COMPLETE.json'")
edit(p,"'umap/results/VERIFICATION.json'","'releases/senosain140k_20260921/provenance/FINAL_CHECKS.json'")
s=p.read_text();start=s.index('    expected_counts = {');end=s.index('    pooled =',start)
s=s[:start]+"    expected_counts = md.display_cell_type.value_counts().to_dict()\n    assert sum(expected_counts.values()) == 84000\n"+s[end:]
s=s.replace('pooled.sum() == 2667 and ','').replace("'coordinate_refit': False","'coordinate_refit': True").replace("'version': 'v4_CD4_top_simplified_groups'","'version': 'senosain140k_20260921'")
# Fit a taller or wider new embedding inside the original panel canvas; retain aspect ratio.
s=s.replace("    plot_left = [.22, 3.17]", "    max_plot_h = 1.94\n    if plot_h > max_plot_h:\n        plot_w *= max_plot_h / plot_h\n        plot_h = max_plot_h\n    plot_left = [.22, 3.17]")
p.write_text(s)
for rel in ['data/RAMP_UMAP.npy','data/CELL_METADATA.csv.gz','provenance/RAMP_UMAP_PARAMETERS.json','provenance/SOURCE_UMAP_VERIFICATION.json','provenance/SOURCES.json']:(F/'Panel B'/rel).unlink()
p=F/'Panel D/scripts/draw_panel_D.py'
edit(p,"SELECTED = SOURCE/'roi_threshold_search0050913'", "SELECTED = SOURCE/'roi_threshold_search0050913'\nNEW = STORY/'releases/senosain140k_20260921/analysis/C30_SPATIAL.csv.gz'")
edit(p,"SOURCE/'data/C30_synthetic_cell_labels.tif',SOURCE/'results/synthesis_audit.json']","SOURCE/'data/C30_synthetic_cell_labels.tif',SOURCE/'results/synthesis_audit.json',NEW]")
edit(p,'    assert values.CellID.equals(geom.CellID)', '''    assert values.CellID.equals(geom.CellID)
    new=pd.read_csv(NEW).set_index('CellID')
    assert new.index.is_unique and len(new)==len(values)==10719
    np.testing.assert_allclose(values.observed_HLA_A_patient_rank, new.loc[values.CellID,'measured_relative_score'],rtol=0,atol=1e-14)
    values['RAMP_k30']=new.loc[values.CellID,'new_prediction'].to_numpy()
    full_rho=float(spearmanr(values.observed_HLA_A_patient_rank,values.RAMP_k30).statistic)''')
edit(p,'pd.testing.assert_frame_equal(selected.reset_index(drop=True),frozen,check_exact=False,rtol=1e-14,atol=1e-12)',"pd.testing.assert_frame_equal(selected.drop(columns='RAMP_k30').reset_index(drop=True),frozen.drop(columns='RAMP_k30'),check_exact=False,rtol=1e-14,atol=1e-12)")
edit(p,"    assert abs(rho-prior['raw_spearman'])<1e-12","    assert np.isfinite(rho)")
edit(p,"'full_C30_cell_spearman':search['whole_C30_raw_spearman']","'full_C30_cell_spearman':full_rho")
edit(p,"'same_ROI_values_and_blending_as_approved_preview':True", "'same_ROI_geometry_color_limits_and_blending_as_approved_preview':True")
edit(p,"'new_prediction_or_refit':False", "'new_prediction_or_refit':True")
edit(p,"'numeric_values_changed':False", "'numeric_values_changed':True, 'new_prediction_source':'senosain140k_20260921'")
edit(p,"'version':'v3_selected_C30_synthetic_ROI'","'version':'senosain140k_20260921_same_ROI'")
for name in ['E','F']:
 # Remove only already-backed-up local inputs, so staging checks ingest new summary.
 script=F/f'Panel {name}/scripts/draw_panel_{name}.py';txt=script.read_text()
 begin=txt.index('FILES =');end=txt.index('\ndef sha',begin)
 ns={'SUMMARY':S/'summary_steps3_6','STORY':S};exec(txt[begin:end],ns)
 for rel in ns['FILES']:
  q=F/f'Panel {name}'/rel
  if q.exists():q.unlink()
p=F/'Panel F/scripts/draw_panel_F.py'
edit(p,"'Log-rank p = 6.83 × 10⁻⁵'", "f'Log-rank p = {k.logrank_p:.4f}'")
edit(p,"    assert abs(k.logrank_p - 6.82600858054177e-05) < 1e-15", "    assert 0 < k.logrank_p < 1")
print('Updated B/D/F renderers; E reads refreshed summary.')
