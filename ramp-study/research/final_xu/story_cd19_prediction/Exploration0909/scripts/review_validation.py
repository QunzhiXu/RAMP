"""Audit frozen NN validation and create exploratory layouts; no model refit."""
from runtime import BASE, STORY, sha, save, now
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

OUT = BASE / 'validation_review'
OUT.mkdir(exist_ok=True)
SOURCE = STORY / 'figure/panel_E/data'
FILES = [SOURCE / 'heldout_marker_validation_summary.csv',
         SOURCE / 'heldout_marker_validation_all_methods_by_donor.csv',
         STORY / 'figure/panel_E/README.md',
         STORY / 'prediction/run_prediction_and_validation.py',
         STORY / 'prediction/masked_marker_model.py',
         STORY / 'prediction/output/heldout_marker_validation_summary.csv']
BEFORE = {str(p.relative_to(STORY)): sha(p) for p in FILES}
summary = pd.read_csv(FILES[0]).set_index('marker')
donors = pd.read_csv(FILES[1])
donors = donors.loc[donors.method.eq('Equal dual-source')].copy()
assert len(summary) == 14 and len(donors) == 42
assert donors.CODEX_donor.nunique() == 3
assert not donors.duplicated(['marker', 'CODEX_donor']).any()
assert not donors[['spearman', 'high_state_auroc']].isna().any().any()
recomputed = donors.groupby('marker').agg(
    median_donor_spearman=('spearman', 'median'),
    mean_high_state_auroc=('high_state_auroc', 'mean'))
assert np.allclose(summary.loc[recomputed.index].values, recomputed.values,
                   atol=1e-8, rtol=0)
summary.to_csv(OUT / 'frozen_marker_summary.csv')
donors.to_csv(OUT / 'frozen_donor_metrics.csv', index=False)

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
                     'svg.fonttype':'none', 'pdf.fonttype':42,
                     'axes.labelcolor':'#28313A', 'text.color':'#28313A',
                     'axes.edgecolor':'#7F8891', 'xtick.color':'#52606B',
                     'ytick.color':'#52606B'})
BLUE='#337EB2'
CMAP=LinearSegmentedColormap.from_list('validation_blue', ['#F2F5F8','#BAD7E6','#5299BF','#12507D'])

def export(fig, stem):
    for ext in ('png', 'svg'):
        fig.savefig(OUT / f'{stem}.{ext}', dpi=240, facecolor='white')
    plt.close(fig)

def main_layout():
    fig=plt.figure(figsize=(6.5,5.25))
    fig.text(.035,.965,'Masked-marker validation',ha='left',va='top',
             fontsize=16,fontweight='bold')
    axflow=fig.add_axes([.035,.805,.94,.10]); axflow.set_axis_off()
    for x, label in zip((.02,.366,.712),
                        ('Hide a CODEX marker','Predict with the NN','Check measured values')):
        axflow.add_patch(FancyBboxPatch((x,.18),.268,.64,
            boxstyle='round,pad=0.012,rounding_size=0.045',
            edgecolor='#AAB3BB',facecolor='white',linewidth=.8))
        axflow.text(x+.134,.50,label,ha='center',va='center',fontsize=8.0)
    for x in (.302,.648):
        axflow.annotate('',xy=(x+.05,.5),xytext=(x,.5),
                        arrowprops={'arrowstyle':'->','lw':1.0,'color':'#687780'})
    ax=fig.add_axes([.135,.165,.79,.57])
    ax.set_xlim(0,1); ax.set_ylim(.49,1.005)
    ax.set_xticks(np.arange(0,1.01,.2)); ax.set_yticks(np.arange(.5,1.01,.1))
    ax.set_xlabel('Expression agreement · Spearman ρ',labelpad=8,fontsize=10)
    ax.set_ylabel('High-state discrimination · AUROC',labelpad=8,fontsize=10)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(color='#EBEFF2',linewidth=.65,zorder=0)
    ax.tick_params(length=3,width=.6)
    ax.axhline(.5,color='#9BA6AD',linestyle=(0,(4,3)),lw=.8)
    # Offsets are layout choices only; every frozen marker is displayed.
    offsets={'CD117':(-13,17),'CD11B':(-52,-2),'CD11C':(12,-12),
             'CD123':(-57,10),'CD14':(11,-12),'CD3':(-30,12),
             'CD33':(13,8),'CD34':(-12,14),'CD38':(-24,12),
             'CD4':(-29,-12),'CD45':(15,0),'CD45RA':(-48,11),
             'CD61':(13,-12),'CD8':(12,8)}
    for marker,row in summary.iterrows():
        x,y=row.median_donor_spearman,row.mean_high_state_auroc
        ax.scatter(x,y,s=43,color=BLUE,edgecolor='white',linewidth=.6,zorder=3)
        ax.annotate(marker,(x,y),xytext=offsets[marker],textcoords='offset points',
                    fontsize=8.4,ha='left',va='center',
                    arrowprops={'arrowstyle':'-','lw':.55,'color':'#93A3AE',
                                'shrinkA':2,'shrinkB':5})
    fig.text(.135,.054,'Each point: one marker, summarized over 3 CODEX donors.',fontsize=8.2)
    fig.text(.135,.026,'14 simulated missing markers · Frozen equal dual-source NN results',
             fontsize=8.0,color='#66747F')
    export(fig,'validation_joint_view')

def donor_layout():
    order=summary.sort_values('median_donor_spearman',ascending=False).index.tolist()
    donor_order=sorted(donors.CODEX_donor.unique())
    fig,axes=plt.subplots(1,2,figsize=(7.1,4.8),sharey=True)
    configs=[('spearman','Expression agreement',0,1,[0,.5,1]),
             ('high_state_auroc','High-state discrimination',.5,1,[.5,.75,1])]
    for ax,(metric,title,vmin,vmax,ticks) in zip(axes,configs):
        table=donors.pivot(index='marker',columns='CODEX_donor',values=metric).loc[order,donor_order]
        im=ax.imshow(table,aspect='auto',cmap=CMAP,vmin=vmin,vmax=vmax)
        ax.set_title(title,fontsize=10.5,fontweight='bold',pad=9)
        ax.set_xticks(range(3),['NBM27','NBM33','NBM40'],fontsize=8.5)
        ax.set_yticks(range(14),order,fontsize=9)
        ax.set_xticks(np.arange(-.5,3,1),minor=True)
        ax.set_yticks(np.arange(-.5,14,1),minor=True)
        ax.grid(which='minor',color='white',linewidth=1.8)
        ax.tick_params(which='both',length=0)
        ax.spines[:].set_visible(False)
        cbar=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.045,pad=.09,ticks=ticks)
        cbar.outline.set_visible(False)
        cbar.set_label('Spearman ρ' if metric=='spearman' else 'AUROC',fontsize=8.5)
    fig.suptitle('Validation across individual CODEX donors',fontsize=14,fontweight='bold',
                 ha='left',x=.035,y=.98)
    fig.text(.035,.017,'All 14 markers and all 3 donors in the frozen outer validation; no cell-type labels used.',
             fontsize=8.0,color='#66747F')
    fig.subplots_adjust(left=.13,right=.97,top=.86,bottom=.14,wspace=.17)
    export(fig,'validation_donor_heatmap')

main_layout(); donor_layout()
current=pd.read_csv(FILES[-1])
audit={
    'created_utc':now(), 'action':'review and replot existing neural-network validation only',
    'formal_files_modified':False, 'neural_network_refit':False,
    'frozen_panel_E':{
        'n_markers':14,'n_codex_donors':3,'n_marker_donor_pairs':42,
        'method':'Equal dual-source',
        'median_marker_median_donor_rho':float(summary.median_donor_spearman.median()),
        'mean_marker_mean_donor_auroc':float(summary.mean_high_state_auroc.mean()),
        'summary_matches_recomputed_donor_metrics':True,
        'protocol_source':'figure/panel_E/README.md; permanent holdout is documented, original training implementation is not bundled here',
    },
    'current_prediction_output':{
        'median_marker_median_donor_rho':float(current.median_donor_spearman.median()),
        'mean_marker_mean_donor_auroc':float(current.mean_high_state_auroc.mean()),
        'classification':'inference-masked reconstruction; not permanent CODEX target holdout',
        'reason':'All 14 measured CODEX markers are eligible training targets; the same fitted model is then evaluated on all CODEX rows.'
    },
    'inputs_sha256':BEFORE,
}
assert BEFORE == {str(p.relative_to(STORY)):sha(p) for p in FILES}
save(OUT/'audit.json',audit)
print(audit['frozen_panel_E'])
print(f'Saved layouts and audit to {OUT}')
