from pathlib import Path
import json
root=Path(__file__).resolve().parent.parent
folder=root/'Main Figure/publication_display_20260922'
p=folder/'rendering_helpers.py';s=p.read_text()
# Apply the accepted melanoma display palette and layer order at Figure 1 size.
m=json.loads((root/'Main Figure/Figure 2/Panel C/umap_style_review/manifest.json').read_text())
a=s.index('PALETTE = {');b=s.index('B_METHODS =',a)
s=s[:a]+'PALETTE = '+repr(m['celltype_palette'])+'\nORDER = '+repr(m['draw_order_bottom_to_top'])+'\n'+s[b:]
s=s.replace("benchmark='tnbc',method=method,marker=marker", "benchmark='melanoma',method=method,marker=marker")
s=s.replace("c=colors[order],s=.058,alpha=.72,", "c=colors[order],s=.20*(21.7/(1.975217391304348*25.4))**2,alpha=.90,")
s=s.replace("size,alpha=(.058,.5) if label=='Tumor / epithelial' else (.070,.85)\n            if label=='Non-joint':size,alpha=.058,.65", "size,alpha=(.18,.32) if label in ['Non-joint','Epithelial (non-joint)'] else (.38,.95)\n            size *= (21.7/(1.975217391304348*25.4))**2")
s=s.replace("'Cross-platform alignment in TNBC'","'Cross-platform alignment in melanoma'")
s=s.replace("['IMC1','IMC2','CODEX']","['CODEX','IMC','MIBI']")
s=s.replace("['CD3','CD8','PANCK'", "['CD3','CD8','SOX10'")
s=s.replace("'Harmony-direct_source_scale':'Harmony-direct'", "'Harmony-direct_source_scale':'Harmony'")
a=s.index('    # Two compact legend rows'); b=s.index('\n\ndef statistical_axes',a)
s=s[:a]+'''    # Retain all ten melanoma display identities in two compact rows.
    labels=['Tumor','Stroma','Myeloid','Endothelial','NK',
            'CD4 T','CD8 T','Other T / CD8−','Non-joint','Epithelial (non-joint)']
    positions=[(66,117),(86,117),(108,117),(135,117),(160,117),
               (66,120.5),(83,120.5),(100,120.5),(129,120.5),(149,120.5)]
    for label,(x,y) in zip(labels,positions):
        mini_legend(fig,x,y,[label],[PALETTE[label]],0,size=5.1)
'''+s[b:]
p.write_text(s)
p=folder/'build_figure.py';s=p.read_text()
s=s.replace("h.load_case('tnbc',h.READS);markers,matrices=h.load_matrices('tnbc',meta,h.READS)","h.load_case('melanoma',h.READS);markers,matrices=h.load_matrices('melanoma',meta,h.READS)\n    assert len(meta)==60000 and meta.unit_key.nunique()==46 and len(markers)==12\n    assert set(meta.celltype)==set(h.PALETTE)")
s=s.replace("MAIN/'text/ridge_display_audit.csv'", "MAIN/'Figure 2/Panel C/umap_style_review/ridge_display_audit.csv'")
s=s.replace('pd.testing.assert_frame_equal(pd.DataFrame(h.AUDIT),prior_ridge,check_exact=False,rtol=1e-12,atol=1e-12)', "pd.testing.assert_frame_equal(pd.DataFrame(h.AUDIT)[prior_ridge.columns],prior_ridge,check_exact=False,rtol=1e-12,atol=1e-12)\n    pd.DataFrame(h.AUDIT).to_csv(HERE/'ridge_display_audit.csv',index=False)")
s=s.replace('figure_1_harmony.', 'figure_1_melanoma.')
s=s.replace('panels_a2_b_c_data_unchanged=True', "panels_a2_c_data_unchanged=True,panel_b_case='melanoma',panel_b_cells=60000,panel_b_sample_units=46,panel_b_shared_markers=12,panel_b_ridge_markers=['CD3','CD8','SOX10'],panel_b_display_source=str(MAIN/'Figure 2/Panel C/umap_style_review'),panel_b_all_cells_retained=True")
s=s.replace('Raw, Harmony-direct, CytoNorm, RAMP); lower', 'Raw, Harmony, CytoNorm, RAMP); lower')
p.write_text(s)
