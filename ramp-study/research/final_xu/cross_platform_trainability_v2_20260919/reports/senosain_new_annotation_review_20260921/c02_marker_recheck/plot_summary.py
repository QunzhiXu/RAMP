"""Patient-level native marker summary, independent of diagnostic gates."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/senosain_marker_recheck_mpl')
from pathlib import Path
import hashlib,json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ANN=HERE.parents[3]/'raw_data/lung_cancer/Senosain_2025_CyTOF_and_omics/annotation'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
paths=[ANN/'rough_annotation/senosain_140k_final_cell_annotations.csv.gz',ANN/'senosain_140k_raw35.npy',ANN/'senosain_140k_markers35.csv']
hs={str(p):sha(p) for p in paths}
a=pd.read_csv(paths[0],dtype={'patient_id':str});x=np.load(paths[1],mmap_mode='r');markers=pd.read_csv(paths[2]).marker.tolist()
groups={'C02':['C02'],'C09':['C09'],'T cells':['C00','C03'],'Epithelial':['C01','C05'],'Endothelial':['C11']}
colors=['#CE7D15','#087D73','#4875B4','#BC5673','#80719B']
keys=['CD45','CD3','CD11B','HLA_DR','EPCAM','CK7'];rows=[]
for patient,g in a.groupby('patient_id'):
    for name,cl in groups.items():
        idx=g.index[g.parent_cluster.isin(cl)].to_numpy()
        if len(idx)<20:continue
        for marker in keys:rows.append(dict(patient_id=patient,group=name,marker=marker,cells=len(idx),median=float(np.median(x[idx,markers.index(marker)]))))
d=pd.DataFrame(rows);d.to_csv(HERE/'PATIENT_GROUP_RAW_MEDIANS.csv',index=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,3,figsize=(12.8,7.6),layout='constrained');rng=np.random.default_rng(20260921)
for ax,marker in zip(axes.flat,keys):
    vals=[d[(d.marker==marker)&d.group.eq(name)]['median'].to_numpy() for name in groups]
    p=ax.boxplot([np.arcsinh(v/5) for v in vals],positions=np.arange(5),widths=.52,patch_artist=True,showfliers=False,
                 medianprops=dict(color='black',lw=1.5),whiskerprops=dict(color='#666666'),capprops=dict(color='#666666'))
    for i,(v,c,b) in enumerate(zip(vals,colors,p['boxes'])):
        b.set_facecolor(c);b.set_alpha(.25);b.set_edgecolor(c)
        ax.scatter(i+rng.uniform(-.15,.15,len(v)),np.arcsinh(v/5),color=c,s=12,alpha=.65,edgecolors='none',zorder=3)
    hi=max(map(np.max,vals))*1.2;ticks=np.array([0,5,50,500,5000,50000]);ticks=ticks[ticks<=hi]
    ax.set_yticks(np.arcsinh(ticks/5),[f'{v:g}' for v in ticks]);ax.set_ylim(-.1,np.arcsinh(hi/5))
    ax.set_xticks(np.arange(5),[f'{name}\n(n={len(v)})' for name,v in zip(groups,vals)],fontsize=9,rotation=15)
    ax.set_title(marker.replace('_','-'),fontsize=13);ax.set_ylabel('Provider expression');ax.grid(axis='y',alpha=.15)
fig.suptitle('C02 biomarker expression compared with lineage references',fontweight='bold',fontsize=15)
fig.supxlabel('Each point: one patient median (at least 20 cells). Native values shown on an asinh(value / 5) display.',fontsize=10)
fig.savefig(HERE/'C02_KEY_MARKER_BOX_PLOTS.png',dpi=190,bbox_inches='tight');plt.close(fig)
assert all(sha(p)==h for p,h in hs.items())
(HERE/'SUMMARY_PLOT_CHECKS.json').write_text(json.dumps(dict(inputs_sha256=hs,source_files_unchanged=True,script_sha256=sha(__file__),
    patients_by_group=d[d.marker.eq('CD45')].groupby('group').size().to_dict()),indent=2)+'\n')
