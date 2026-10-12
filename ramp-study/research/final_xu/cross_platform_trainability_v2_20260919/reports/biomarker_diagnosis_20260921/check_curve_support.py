"""Reconstruct RAMP sampling/role counts without refitting coupling or alignment."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT.parent/'algorithm'))
from ramp import ramp_ver2 as alg


def main():
    rows=[];summ=[];input_hashes={}
    for task in ['bone_marrow','tnbc','melanoma','lung_cancer','lung_story','melanoma_story','hnscc_story']:
        d=ROOT/f'representations/formal_v0_5/{task}/complete/ramp'
        req=json.loads((d/'request.json').read_text());md=pd.read_csv(d/'identity.csv',dtype=str)
        base=np.load(d/'kmeanslow.npy');aligned=np.load(d/'values.npy')
        assert np.array_equal(np.load(d/'cell_ids.npy').astype(str),md.cell_id.to_numpy(str)) if 'cell_id' in md else True
        n,audit=alg.adaptive_pair_fit_size(md,req['datasets'])
        threshold=max(40,4*alg.CURVE_MIN_BIN_CELLS)
        for ds in req['datasets']:
            sample=alg.unit_balanced_fit_indices(md,ds,n)
            for ref in req['datasets']:
                if ds==ref:continue
                role=alg.deterministic_roles(req['family'],ds,ref,sample)
                sampled=md.iloc[sample].copy();sampled['role']=role
                for unit,g in md[md.dataset.eq(ds)].groupby('unit_key'):
                    rr=sampled.loc[sampled.unit_key.eq(unit),'role'];curve=int(rr.eq('curve_fit').sum())
                    ix=g.index.to_numpy();delta=np.abs(aligned[ix]-base[ix])
                    rows.append(dict(task=task,dataset=ds,reference=ref,unit=unit,pair_fit=n,sampled=len(rr),
                        curve_fit_cells=curve,required_curve_cells=threshold,curve_support_passes=curve>=threshold,
                        changed_cells=int(np.any(delta>1e-8,axis=1).sum()),total_cells=len(ix),
                        changed_markers=int(np.any(delta>1e-8,axis=0).sum()),maximum_change=float(delta.max()),
                        exact_equal_to_baseline=bool(np.array_equal(aligned[ix],base[ix]))))
        out=pd.DataFrame([r for r in rows if r['task']==task])
        for ds,g in out.groupby('dataset'):
            by=g.groupby('unit').agg(any_curve_support=('curve_support_passes','max'),
                                    changed_markers=('changed_markers','max'),exact_equal=('exact_equal_to_baseline','all'))
            assert by.loc[~by.any_curve_support,'exact_equal'].all(),(task,ds)
            s=dict(task=task,dataset=ds,units=len(by),units_with_curve_support=int(by.any_curve_support.sum()),
                   units_with_changed_values=int(by.changed_markers.gt(0).sum()),
                   units_exactly_unchanged=int(by.exact_equal.sum()),minimum_curve=int(g.curve_fit_cells.min()),maximum_curve=int(g.curve_fit_cells.max()))
            summ.append(s)
        for file in ['request.json','identity.csv','kmeanslow.npy','values.npy']:
            input_hashes[str(d/file)]=hashlib.sha256((d/file).read_bytes()).hexdigest()
    pd.DataFrame(rows).to_csv(HERE/'RAMP_CURVE_SUPPORT_PER_UNIT.csv',index=False)
    pd.DataFrame(summ).to_csv(HERE/'RAMP_CURVE_SUPPORT_SUMMARY.csv',index=False)
    (HERE/'CURVE_SUPPORT_CHECKS.json').write_text(json.dumps(dict(no_refit=True,
        all_units_without_sufficient_profile_support_exactly_equal_baseline=True,
        actual_required_curve_cells=threshold,nominal_curve_role_fraction=12/25,
        input_sha256=input_hashes),indent=2)+'\n')
    print(pd.DataFrame(summ).to_string(index=False))


if __name__=='__main__':main()
