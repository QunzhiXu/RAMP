"""Find readable examples with >=50 CD4 T, CD8 T and B cells per sample."""
from pathlib import Path
import os,sys,json
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'Main Figure/pair_inversion_boxplot_20260914'))
import compute_pair_inversions as frozen
os.environ['NUMBA_CACHE_DIR']=str(HERE/'.cache/numba')
import numpy as np
import pandas as pd
from scipy.stats import rankdata

BLABEL={'bone_marrow':'Mature / CODEX B','tnbc':'B lineage','lung_cancer':'B-like'}

def auc(a,b):
    r=rankdata(np.r_[a,b]);return (r[:len(a)].sum()-len(a)*(len(a)+1)/2)/(len(a)*len(b))

def main():
    rows=[];coverage=[]
    for case in ['bone_marrow','tnbc','lung_cancer']:
        base=ROOT/'benchmark_results'/case/'ramp/alignment/input'
        meta=pd.read_csv(frozen.track(base/'cell_metadata.csv.gz'),dtype={'cell_id':str},low_memory=False)
        markers=frozen.track(base/'markers.txt').read_text().splitlines()
        raw=np.load(frozen.track(base/'raw.npy'))
        methods={'Raw':raw}
        for m,_,z in frozen.load_methods(case,meta.cell_id.to_numpy(),markers):methods[m]=z
        methods['fastMNN reconstructed']=np.load(frozen.track(ROOT/f'Main Figure/fastmnn_reconstructed_audit_20260914/{case}/reconstructed.npy'))
        labels=meta.celltype.to_numpy();cells=['CD4 T','CD8 T',BLABEL[case]]
        for unit,ix in meta.groupby('unit_key').indices.items():
            ids=[np.flatnonzero(labels[ix]==c) for c in cells]
            ns=[len(p) for p in ids]
            eligible=min(ns)>=50
            coverage.append(dict(case=case,unit_key=unit,n_cd4=ns[0],n_cd8=ns[1],n_b=ns[2],eligible=eligible))
            if not eligible:continue
            for marker in [m for m in ['CD4','CD8','CD14'] if m in markers]:
                j=markers.index(marker)
                for m,z in methods.items():
                    v=z[ix,j];r=rankdata(v)/(len(v)+1)
                    out=dict(case=case,unit_key=unit,marker=marker,method=m,n_cd4=ns[0],n_cd8=ns[1],n_b=ns[2])
                    for c,p in zip(['cd4','cd8','b'],ids):
                        for name,q in zip(['q25','median','q75'],[.25,.5,.75]):out[c+'_'+name]=np.quantile(r[p],q)
                    for left,right in [(0,1),(0,2),(1,2)]:out[['cd4','cd8','b'][left]+'_vs_'+['cd4','cd8','b'][right]+'_auc']=auc(v[ids[left]],v[ids[right]])
                    rows.append(out)
        print(case,'done',flush=True)
    d=pd.DataFrame(rows);d.to_csv(HERE/'common_cell_candidate_statistics.csv',index=False)
    pd.DataFrame(coverage).to_csv(HERE/'common_cell_coverage.csv',index=False)
    assert all(frozen.sha(p)==h for p,h in frozen.READS.items())
    (HERE/'candidate_manifest.json').write_text(json.dumps(dict(minimum_each_type=50,three_types=['CD4 T','CD8 T','B cells'],
        scope='All eligible samples in the three original benchmarks with a B label; melanoma has no B label in the frozen evaluation annotations.',
        percentile_scope='All cells of each sample; never re-ranked among the displayed three types.',
        frozen_sources_unchanged=True,input_sha256=frozen.READS),indent=2)+'\n')
    sub=d[d.case.eq('tnbc')&d.marker.eq('CD8')]
    show=1-sub.pivot(index='unit_key',columns='method',values='cd4_vs_cd8_auc')
    show['joint_loss']=show.Raw-show[['Harmony-direct[source_scale]','cyCombine']].max(axis=1)
    print(show.sort_values('joint_loss',ascending=False).to_string(),flush=True)
    print(d[d.unit_key.eq('Shiao CODEX|h05')&d.marker.eq('CD4')][['method','cd4_vs_b_auc','cd4_median','cd8_median','b_median']].to_string(index=False),flush=True)

if __name__=='__main__':main()
