"""Simplify labels and apply one global painter order, retaining every cell."""
from runtime import BASE,save,sha
import numpy as np,pandas as pd,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.lines import Line2D

OUT=BASE/'CD4_NN_UMAP_validation/joint_umap'
SOURCE=OUT/'codex_predictions_on_joint_umap.csv.gz'
source_hash=sha(SOURCE)
d=pd.read_csv(SOURCE)
assert len(d)==30000 and d.dataset.eq('BM CODEX').all() and d.donor.nunique()==12
xy=d[['UMAP1','UMAP2']].to_numpy()
cd4=d.celltype.eq('CD4 T').to_numpy()
assert cd4.sum()==1426
palette={'CD4 T':'#266BAE','Other cells':'#E7C36B'}
SIZE=1.2
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none',
                     'font.size':9,'text.color':'#27323B'})
fig=plt.figure(figsize=(12,3.8))
positions=[.022,.352,.663]
axes=[fig.add_axes([left,.13,.25,.72]) for left in positions]
# Uniform rule for the complete dataset. No outlier masks or position-based edits.
axes[0].scatter(xy[cd4,0],xy[cd4,1],s=SIZE,c=palette['CD4 T'],
                alpha=1,linewidths=0,zorder=2,rasterized=True)
axes[0].scatter(xy[~cd4,0],xy[~cd4,1],s=SIZE,c=palette['Other cells'],
                alpha=1,linewidths=0,zorder=3,rasterized=True)
cmap=LinearSegmentedColormap.from_list('CD4_expression',
       ['#DCE1E6','#F5E7B2','#EDB34E','#CF572D','#861E41'])
prior=json.loads((OUT/'PLOT_AUDIT.json').read_text())
norm=Normalize(prior['vmin'],prior['vmax'],clip=True)
for ax,col in zip(axes[1:],['measured_CD4','predicted_CD4']):
    values=d[col].to_numpy(); order=np.argsort(values,kind='stable')
    sc=ax.scatter(xy[order,0],xy[order,1],s=SIZE,c=values[order],cmap=cmap,
                  norm=norm,alpha=1,linewidths=0,rasterized=True)
lo=xy.min(axis=0); hi=xy.max(axis=0); pad=.035*(hi-lo)
for ax,left,title in zip(axes,positions,['Cell type','Measured CD4','Predicted CD4']):
    ax.set(xlim=(lo[0]-pad[0],hi[0]+pad[0]),ylim=(lo[1]-pad[1],hi[1]+pad[1]))
    ax.set_aspect('equal',adjustable='box'); ax.axis('off')
    fig.text(left+.125,.835,title,ha='center',va='bottom',fontsize=11.5,fontweight='bold')
fig.legend([Line2D([0],[0],marker='o',ls='',color=c,ms=4.3) for c in palette.values()],
           list(palette),loc='center left',bbox_to_anchor=(.269,.47),
           frameon=False,fontsize=7.8,labelspacing=.8,handletextpad=.35,borderaxespad=0)
cax=fig.add_axes([.943,.28,.009,.39]); cb=fig.colorbar(sc,cax=cax)
cb.set_label('CD4 (KMEANSLOW)',fontsize=8.2); cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=8,width=.6,length=2.5)
fig.text(.03,.965,'CD4 prediction validation',ha='left',va='top',fontsize=14.5,fontweight='bold')
fig.text(.03,.044,'Joint three-platform UMAP · 30,000 CODEX cells · Shared coordinates and expression scale',fontsize=8.4)
fig.text(.03,.011,'Cell-type layer order: CD4 T first, Other cells above; all cells retained.',fontsize=7.8,color='#65717B')
for ext in ['png','svg']:
    fig.savefig(OUT/f'CD4_validation_simple_layers.{ext}',dpi=300,facecolor='white')
plt.close(fig)
assert sha(SOURCE)==source_hash
save(OUT/'CD4_simple_layers_audit.json',{
    'source_sha256':source_hash,'n_cells':len(d),'n_CD4_T':int(cd4.sum()),
    'n_other':int((~cd4).sum()),'cell_type_mapping':'author-derived CD4 T versus all remaining CODEX cells',
    'cell_type_painter_order':['CD4 T','Other cells'],'palette':palette,
    'point_size_all_cells_all_panels':SIZE,'alpha_all_cells':1,
    'cell_removal':False,'position_specific_rules':False,'label_changes':False,
    'coordinates_or_prediction_changed':False,'expression_color_limits':[prior['vmin'],prior['vmax']],
    'expression_painter_order':'ascending expression, as in previous comparison',
    'overplotting_note':'Other cells can occlude overlapping CD4 points; this is a display-order variant, not evidence of increased biological purity.'})
