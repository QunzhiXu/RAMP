"""Show CD8 specificity in all five frozen Meyer IMC sample units."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from figure_style import ROOT, setup, track, READS, sha, plt
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata, mannwhitneyu
from matplotlib.patches import Patch

setup()
base = ROOT/'benchmark_results/tnbc'
meta = pd.read_csv(track(base/'ramp/alignment/input/cell_metadata.csv.gz'), low_memory=False)
markers = track(base/'ramp/alignment/input/markers.txt').read_text().splitlines()
j = markers.index('CD8')
raw = np.load(track(base/'ramp/alignment/input/raw.npy'), mmap_mode='r')[:,j]
ramp = np.load(track(base/'ramp/alignment/data/ramp_aligned.npy'), mmap_mode='r')[:,j]
table = pd.read_csv(track(base/'cycombine/alignment/data/corrected_expression.csv.gz'), usecols=['cell_id','CD8'])
assert table.cell_id.is_unique and set(table.cell_id) == set(meta.cell_id)
cyco = table.set_index('cell_id').loc[meta.cell_id,'CD8'].to_numpy()
methods = {'Raw': raw, 'cyCombine': cyco, 'RAMP': ramp}
units = sorted(meta.loc[meta.dataset.eq('Meyer IMC'),'unit_key'].unique())
assert len(units) == 5
fig, axes = plt.subplots(1,5,figsize=(14,3.8),sharey=True)
fig.subplots_adjust(left=.045,right=.995,bottom=.14,top=.70,wspace=.13)
fig.text(.045,.92,'CD8 specificity in all five Meyer IMC samples',fontsize=15,weight='bold')
fig.legend(handles=[Patch(facecolor='#297AA4',label='CD8 T'),Patch(facecolor='#D36C58',label='CD4 T')],
           loc='upper right',bbox_to_anchor=(.993,.985),ncol=2,frameon=False,fontsize=10)
rows=[]
for ax,unit in zip(axes,units):
    scope=meta.unit_key.eq(unit).to_numpy()
    types=meta.loc[scope,'celltype']
    target=types.eq('CD8 T').to_numpy(); other=types.eq('CD4 T').to_numpy()
    aucs=[]
    for x,(method,values) in enumerate(methods.items()):
        ranks=rankdata(values[scope],method='average')/(scope.sum()+1)
        arrays=[ranks[target],ranks[other]]
        artist=ax.boxplot(arrays,positions=[x-.16,x+.16],widths=.26,patch_artist=True,
            showfliers=False,manage_ticks=False,medianprops={'color':'#222222','linewidth':1},
            boxprops={'edgecolor':'#49525A','linewidth':.7},whiskerprops={'color':'#49525A','linewidth':.7},
            capprops={'color':'#49525A','linewidth':.7})
        for patch,color in zip(artist['boxes'],['#297AA4','#D36C58']): patch.set_facecolor(color)
        auc=mannwhitneyu(*arrays).statistic/(target.sum()*other.sum())
        aucs.append(auc)
        for label,a in zip(['CD8 T','CD4 T'],arrays):
            rows.append({'unit_key':unit,'method':method,'celltype':label,'n_cells':len(a),
                         'mean_rank':float(a.mean()),'median_rank':float(np.median(a)),
                         'target_vs_other_auc':float(auc)})
    ax.set_title(unit.split('|')[1].strip()+'\n'+f'AUC: {aucs[0]:.3f} → {aucs[1]:.3f}',fontsize=10.7,weight='bold',pad=7)
    ax.set_ylim(0,1); ax.set_xlim(-.55,2.55)
    ax.set_xticks(range(3),methods.keys()); ax.tick_params(axis='x',length=0,labelsize=10,pad=6)
    ax.get_xticklabels()[1].set_weight('bold')
    ax.set_yticks([0,.5,1]); ax.tick_params(axis='y',length=2,labelsize=9.5)
    ax.grid(axis='y',color='#E9E9E9',lw=.7); ax.set_axisbelow(True)
    for spine in ax.spines.values(): spine.set_color('#8C8C8C'); spine.set_linewidth(.65)
axes[0].set_ylabel('CD8 percentile',fontsize=11,weight='bold')
pd.DataFrame(rows).to_csv(HERE/'meyer_cd8_display_statistics.csv',index=False)
for ext in ['png','svg']: fig.savefig(HERE/f'meyer_cd8_all_samples.{ext}',dpi=210)
plt.close(fig)
assert all(sha(p) == h for p,h in READS.items())
(HERE/'plot_manifest.json').write_text(json.dumps({'input_sha256':READS,'inputs_unchanged':True,
    'samples':units,'marker':'CD8','comparison':'CD8 T versus CD4 T',
    'auc_label':'Raw → cyCombine; all five current frozen Meyer IMC sample units are shown.',
    'plot':'Average rank/(n+1) within all cells in each unit/output; median/IQR boxes, 1.5 IQR whiskers, outlier points omitted.'},indent=2)+'\n')
print(HERE/'meyer_cd8_all_samples.png')
