"""Read-only audit of existing masked-marker RAMP representations in all seven cases."""
from run import ROOT, OUT, ALL_TASKS, save, file_sha256
import json
import numpy as np
import pandas as pd

def main():
    original=pd.read_csv(OUT/'ALL_SEVEN_ORIGINAL_COVERAGE.csv')
    records=[]; hashes={}
    for name in ALL_TASKS:
        root=ROOT/f'representations/formal_v0_5/{name}'
        full=root/'complete/ramp'
        req0=json.loads((full/'request.json').read_text())
        id_hash=file_sha256(full/'identity.csv')
        cell_hash=file_sha256(full/'cell_ids.npy')
        md=pd.read_csv(full/'identity.csv',dtype=str)
        coverage=original[original.task.eq(name)].groupby('unit_key').support_passes.agg(['any','all'])
        for path in sorted(root.glob('*/ramp/request.json')):
            if path.parent.parent.name=='complete':continue
            folder=path.parent;request=json.loads(path.read_text())
            same_ids=file_sha256(folder/'identity.csv')==id_hash and file_sha256(folder/'cell_ids.npy')==cell_hash
            same_roles=request['family']==req0['family'] and request['datasets']==req0['datasets']
            assert same_ids and same_roles, path
            # Numerical support depends on identity, row ordering, datasets and family, not expression.
            base=np.load(folder/'kmeanslow.npy',mmap_mode='r')
            value=np.load(folder/'values.npy',mmap_mode='r')
            units=[]
            for (ds,unit),group in md.groupby(['dataset','unit_key']):
                changed=bool(np.any(np.abs(value[group.index]-base[group.index])>1e-8))
                exact=bool(np.array_equal(value[group.index],base[group.index]))
                if not coverage.at[unit,'any']:assert exact,(name,path.parent.parent.name,unit)
                units.append(dict(dataset=ds,unit_key=unit,any_support=bool(coverage.at[unit,'any']),
                    all_support=bool(coverage.at[unit,'all']),changed=changed,exact_baseline=exact))
            frame=pd.DataFrame(units)
            for ds,g in frame.groupby('dataset'):
                records.append(dict(task=name,masked_marker=folder.parent.name,dataset=ds,
                    donors=len(g),any_support=int(g.any_support.sum()),all_support=int(g.all_support.sum()),
                    donors_with_changed_values=int(g.changed.sum()),donors_exact_baseline=int(g.exact_baseline.sum()),
                    same_identity_and_sampling_roles_as_complete=True))
            for p in [path,folder/'COMPLETE.json']:
                hashes[str(p)]=file_sha256(p)
        print(name,'masked audit complete',flush=True)
    frame=pd.DataFrame(records);frame.to_csv(OUT/'ALL_SEVEN_MASKED_COVERAGE.csv',index=False)
    save(OUT/'MASKED_AUDIT_CHECKS.json',dict(representations=int(frame[['task','masked_marker']].drop_duplicates().shape[0]),
        tasks_with_masked_representations=sorted(frame.task.unique().tolist()),
        tasks_without_masked_representations=sorted(set(ALL_TASKS)-set(frame.task)),
        identical_sampling_support_verified=True,all_units_without_support_exactly_baseline=True,
        no_masked_refit_or_new_prediction=True,input_receipt_hashes=hashes))

if __name__=='__main__':main()
