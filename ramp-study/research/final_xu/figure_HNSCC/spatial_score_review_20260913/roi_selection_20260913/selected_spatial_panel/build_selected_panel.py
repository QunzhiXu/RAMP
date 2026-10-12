"""Keep the selected first ROI and replace both footers with one cohort label."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
sys.dont_write_bytecode=True

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
spec=importlib.util.spec_from_file_location('candidate_plot',PARENT/'plot_candidates.py')
plot=importlib.util.module_from_spec(spec);spec.loader.exec_module(plot)
np,pd,plt,fitz=plot.np,plot.pd,plot.plt,plot.fitz

def main():
    record=json.loads((PARENT/'RANKING_AUDIT.json').read_text())['top']['min100']
    assert record['roi']=='522_S1reg067' and record['n_cd8']==115
    cells_path=PARENT/'selected_roi_cells.csv'
    plane_path=PARENT/'images'/record['roi']/'DAPI.raw'
    summary_path=PARENT.parent/'codex_protein_correlation_summary.csv'
    hashes={str(p):plot.sha(p) for p in [cells_path,plane_path,summary_path,PARENT/'plot_candidates.py']}
    d=pd.read_csv(cells_path);assert len(d)==115
    summary=pd.read_csv(summary_path)
    cohort=float(summary[summary.marker.eq('LAG3')&summary.aggregation.eq('unadjusted')]['mean'].iloc[0])
    raw=np.fromfile(plane_path,dtype='>u2').reshape(1440,1920).astype(float)
    bg=np.clip(raw/np.percentile(raw[raw>0],99.8),0,1)**.72*.30
    fig=plt.figure(figsize=(8.8,3.95))
    plot.row(fig,0,d,bg,cohort,'Highest correlation')
    for text in list(fig.texts)[-2:]:text.remove()
    footer=f'74 patients: mean Spearman ρ = {cohort:.2f}'
    fig.text(.5,.24/3.95,footer,ha='center',va='center',fontsize=10.5)
    assert len(fig.texts)==3
    assert not any('Highest' in t.get_text() for t in fig.texts)
    plot.check_text(fig)
    for ext in ['png','pdf','svg']:
        fig.savefig(HERE/f'spatial_state_LAG3.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
    doc=fitz.open(HERE/'spatial_state_LAG3.pdf');assert len(doc)==1
    text=doc[0].get_text()
    assert 'Highest' not in text and '115' not in text and '0.73' not in text
    doc[0].get_pixmap(matrix=fitz.Matrix(1.7,1.7)).save(HERE/'pdf_preview.png');doc.close()
    assert hashes=={p:plot.sha(Path(p)) for p in hashes}
    (HERE/'AUDIT.json').write_text(json.dumps(dict(status='PASS',selected_roi=record,selected_by_user=True,
        display_footer=footer,footer_horizontal_alignment='center',cohort_mean=cohort,
        input_hashes_unchanged=hashes,other_display_parameters_unchanged=True,official_manuscript_modified=False),indent=2)+'\n')
    (HERE/'CAPTION.md').write_text('''# Spatial phenotype correspondence

Predicted CD39/TIGIT state (left) and measured LAG3 (right) are shown for the same 115 CODEX CD8 T cells at their original spatial coordinates over the measured DAPI image. Colors indicate within-patient percentile ranks on a shared 0–1 display scale. The full ROI is shown without cropping. Colored cell-centroid overlays represent cell-level measurements or predictions, rather than raw marker fluorescence.

The illustrative ROI, 522_S1reg067 from patient 23956, was selected for the highest within-ROI Spearman association among the 93 ROIs containing at least 100 CD8 T cells, after screening all 176 ROIs. Its within-ROI Spearman correlation is 0.732. The centered annotation instead reports the equal-weight mean of within-patient Spearman correlations across all 74 patients: 0.522 (95% patient-bootstrap CI, 0.496–0.550). The selected example and the cohort-level statistic are distinct summaries.

LAG3 was excluded from the current shared-protein alignment and CD39/TIGIT state construction. This is an auxiliary phenotype assessment; historical analyses had examined LAG3, so it is not presented as an untouched, independent validation endpoint.
''')
    (HERE/'README_ZH.md').write_text('''# 选定的空间双图

用户选择了第一组、115 个 CD8 的 ROI（522_S1reg067）。两张空间图、颜色和细胞均保持原样。

底部只保留居中的一行：**74 patients: mean Spearman ρ = 0.52**。

ROI 的选择规则、局部相关 0.732，以及图中 0.52 属于全队列的说明保存在 `CAPTION.md`。原两组比较图、正式 HNSCC figure 和 manuscript 均未覆盖。

复现：`python3 build_selected_panel.py`。输出为 `spatial_state_LAG3.png/.pdf/.svg`。
''')
    print('PASS:',footer)

if __name__=='__main__':main()
