"""Diagnose the shifted bone-marrow CD3 ridge without changing frozen results."""
from pathlib import Path
import os
import sys
import json
import hashlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(HERE.parent/'bandwidth_review/.cache/matplotlib'))
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('LOKY_MAX_CPU_COUNT','1')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter1d

TARGET = 'BM CODEX|SB67_NBM36_H26_CODEX_Mesmer'
T_LABELS = ['CD4 T','CD8 T','Other T / NKT']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = ROOT/'benchmark_results/bone_marrow/ramp/alignment'
    files = [base/p for p in ['input/raw.npy','input/cell_metadata.csv.gz','input/markers.txt',
                             'data/kmeanslow_global.npy','data/ramp_aligned.npy',
                             'data/reference_weight_audit.csv','data/strength_audit.csv']]
    hashes = {str(p):sha(p) for p in files}
    meta = pd.read_csv(base/'input/cell_metadata.csv.gz',low_memory=False)
    markers = (base/'input/markers.txt').read_text().splitlines()
    j = markers.index('CD3')
    raw = np.load(base/'input/raw.npy')[:,j]
    baseline = np.load(base/'data/kmeanslow_global.npy')[:,j]
    aligned = np.load(base/'data/ramp_aligned.npy')[:,j]
    t = meta.celltype.isin(T_LABELS).to_numpy()
    rows, cluster_rows = [], []
    lo,hi = np.quantile(aligned,[.005,.995])
    for unit in sorted(meta.unit_key.unique()):
        s = meta.unit_key.eq(unit).to_numpy()
        counts,edges=np.histogram(aligned[s],bins=160,range=(lo,hi))
        smooth=gaussian_filter1d(counts.astype(float),4,mode='constant')
        grid=(edges[1:]+edges[:-1])/2
        row={'unit_key':unit,'dataset':meta.loc[s,'dataset'].iloc[0],
             'short_name':unit.split('|')[-1].replace('SB67_','').replace('_CODEX_Mesmer',''),
             'n_cells':int(s.sum()),'n_T':int((s&t).sum()),'T_fraction':float(t[s].mean()),
             'ramp_total_density_mode':float(grid[smooth.argmax()]),
             'nonT_raw_median':float(np.median(raw[s&~t])),
             'nonT_kmeanslow_median':float(np.median(baseline[s&~t])),
             'nonT_ramp_median':float(np.median(aligned[s&~t])),
             'T_ramp_median':float(np.median(aligned[s&t]))}
        rows.append(row)
        if row['dataset'] != 'BM CODEX':
            continue
        values=raw[s].astype(float)
        shift=-float(values.min())+.01 if values.min()<0 else 0.
        v=values+shift
        cofactor=max(float(v.mean()) if np.median(v)==0 else float(np.median(v)),1e-6)
        z=np.arcsinh(v/cofactor)
        scale=max(float(z.std()),1e-9)
        payload='|'.join(map(str,(42,'KMEANSLOW','BM CODEX',unit,'CD3'))).encode()
        seed=int.from_bytes(hashlib.blake2b(payload,digest_size=8).digest(),'little')%(2**32-1)
        km=KMeans(n_clusters=2,init='k-means++',n_init=20,max_iter=500,tol=1e-6,
                  algorithm='lloyd',random_state=seed).fit(z[:,None])
        lower=km.labels_==km.cluster_centers_.ravel().argmin()
        rebuilt=(z-z[lower].mean())/scale
        exact=np.array_equal(rebuilt.astype(np.float32),baseline[s])
        assert exact,unit
        tt=t[s]
        cluster_rows.append({'unit_key':unit,'low_n':int(lower.sum()),
                             'high_n':int((~lower).sum()),'low_T_n':int((lower&tt).sum()),
                             'high_T_n':int((~lower&tt).sum()),
                             'nonT_assigned_high_fraction':float((~lower)[~tt].mean()),
                             'high_cluster_T_fraction':float(tt[~lower].mean()),
                             'low_cluster_fraction':float(lower.mean()),
                             'cofactor':cofactor,'low_center':float(z[lower].mean()),
                             'high_center':float(z[~lower].mean()),'scale':scale,
                             'baseline_exact_match_after_float32':exact})
    summary=pd.DataFrame(rows)
    clusters=pd.DataFrame(cluster_rows)
    summary.to_csv(HERE/'all_donor_cd3_summary.csv',index=False)
    clusters.to_csv(HERE/'codex_kmeanslow_cluster_audit.csv',index=False)
    sel=meta.unit_key.eq(TARGET).to_numpy()
    type_rows=[]
    for celltype in meta.loc[sel,'celltype'].value_counts().index:
        s=sel&meta.celltype.eq(celltype).to_numpy()
        type_rows.append({'celltype':celltype,'n_cells':int(s.sum()),
                          'raw_median':float(np.median(raw[s])),
                          'kmeanslow_median':float(np.median(baseline[s])),
                          'ramp_median':float(np.median(aligned[s]))})
    pd.DataFrame(type_rows).to_csv(HERE/'target_celltype_cd3.csv',index=False)
    for name in ['reference_weight_audit.csv','strength_audit.csv']:
        f=pd.read_csv(base/'data'/name)
        f.loc[f.unit_key.eq(TARGET)&f.marker_index.eq(j)].to_csv(HERE/f'target_{name}',index=False)

    codex=summary.loc[summary.dataset.eq('BM CODEX')].copy()
    target=summary.set_index('unit_key').loc[TARGET]
    c=clusters.set_index('unit_key').loc[TARGET]
    fig,axs=plt.subplots(1,3,figsize=(14,4.7),gridspec_kw={'width_ratios':[1.25,1,1.05]})
    fig.subplots_adjust(left=.085,right=.985,bottom=.18,top=.79,wspace=.43)
    fig.suptitle('CD3 donor check | NBM36 / H26',x=.085,ha='left',y=.985,
                 fontsize=17,weight='bold')
    fig.text(.085,.885,'2,500 cells  |  145 T cells (5.8%)  |  2,355 non-T cells',fontsize=11)
    ax=axs[0]
    colors=['#C43B35' if u==TARGET else '#8EAFC4' for u in codex.unit_key]
    ax.barh(np.arange(len(codex)),codex.T_fraction*100,color=colors,height=.7)
    ax.set_yticks(np.arange(len(codex)),codex.short_name,fontsize=9)
    ax.invert_yaxis();ax.set_xlim(0,22)
    for k,v in enumerate(codex.T_fraction*100):
        ax.text(v+.35,k,f'{v:.1f}',va='center',fontsize=8.5,color=colors[k])
    ax.set_xlabel('T cells (%)',fontsize=11)
    ax.set_title('T-cell abundance',fontsize=12,weight='bold',pad=11)

    ax=axs[1]
    for r in codex.itertuples():
        if r.unit_key==TARGET:
            continue
        ax.plot([0,1],[r.nonT_kmeanslow_median,r.nonT_ramp_median],
                marker='o',color='#AEBBC3',alpha=.75,linewidth=1,markersize=4)
    yy=[target.nonT_kmeanslow_median,target.nonT_ramp_median]
    ax.plot([0,1],yy,marker='o',color='#C43B35',linewidth=2,markersize=6,zorder=5)
    for k,v in enumerate(yy):
        ax.text(k,v+.035,f'{v:.3f}',ha='center',fontsize=11,color='#C43B35',weight='bold')
    ax.axhline(0,color='#777777',linestyle='--',linewidth=.7)
    ax.set_xticks([0,1],['KMEANSLOW','RAMP'],fontsize=10)
    ax.set_xlim(-.25,1.25);ax.set_ylim(-.07,.86)
    ax.set_ylabel('Median CD3 in non-T cells',fontsize=10.5)
    ax.set_title('Offset already in preprocessing',fontsize=12,weight='bold',pad=11)
    ax.text(.04,.43,'NBM36 / H26',color='#C43B35',fontsize=10,transform=ax.transAxes)
    ax.text(.04,.35,'Other CODEX donors',color='#6F7E88',fontsize=9,transform=ax.transAxes)

    ax=axs[2]
    n=np.array([c.low_n,c.high_n],int)
    nt=np.array([c.low_T_n,c.high_T_n],int)
    ax.bar([0,1],n-nt,color='#8EAFC4',width=.6,label='Non-T cells')
    ax.bar([0,1],nt,bottom=n-nt,color='#8559B4',width=.6,label='T cells')
    for k in range(2):
        ax.text(k,n[k]+25,f'n = {n[k]:,}',ha='center',fontsize=10)
        ax.text(k,(n[k]-nt[k])/2,f'{n[k]-nt[k]:,}\nnon-T',ha='center',va='center',fontsize=11)
    ax.text(1,n[1]-nt[1]/2,f'{nt[1]} T',ha='center',va='center',fontsize=9,color='white',weight='bold')
    ax.set_xticks([0,1],['Lower cluster','Higher cluster'],fontsize=10)
    ax.set_ylabel('Cells',fontsize=11);ax.set_ylim(0,1530)
    ax.set_title('Two-means splits the non-T pool',fontsize=12,weight='bold',pad=11)
    ax.legend(loc='upper left',frameon=False,fontsize=8.5)
    for ax in axs:
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(length=3)
    fig.savefig(HERE/'cd3_donor_diagnosis.png',dpi=190)
    plt.close(fig)
    assert all(sha(Path(p))==v for p,v in hashes.items())
    (HERE/'manifest.json').write_text(json.dumps({
        'target':TARGET,'input_sha256':hashes,'inputs_unchanged':True,
        'T_definition_original_annotations':T_LABELS,
        'donors_screened':len(summary),'codex_donors_screened':len(codex),
        'kmeanslow_reconstruction':'CD3 only, all 12 CODEX donors, exact float32 agreement',
        'alignment_refit':False,'benchmark_outputs_modified':False,
    },indent=2)+'\n')
    print(target.to_json(),flush=True)
    print(c.to_json(),flush=True)


if __name__ == '__main__':
    main()
