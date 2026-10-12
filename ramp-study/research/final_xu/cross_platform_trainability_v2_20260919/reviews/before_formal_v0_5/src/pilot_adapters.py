"""Small, label-blind real-data adapter execution, not a benchmark result."""
import sys
sys.dont_write_bytecode=True
import argparse,json,time
from trainability.core import ROOT,output_path,write_json
from trainability.real_data import load_task,pilot_indices
from trainability.adapters import fit_representation

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',default='lung_story')
    p.add_argument('--per-dataset',type=int,default=1500)
    p.add_argument('--methods',nargs='+',default=['unaligned','ramp','harmony_direct','cycombine','cytonorm','fastmnn','maxfuse','clipped_zscore','minmax','percentile'])
    p.add_argument('--name',required=True)
    p.add_argument('--mask')
    a=p.parse_args()
    full=load_task(a.task); idx=pilot_indices(full,a.per_dataset)
    task=full.subset(idx,a.name)
    out=output_path(ROOT/'validation'/a.name);out.mkdir(exist_ok=True)
    manifest=out/'PILOT.json'
    if not manifest.exists():write_json(manifest,dict(task=a.task,full_cells=len(full.metadata),pilot_cells=len(idx),
        selected_indices=idx.tolist(),pool_sha256=task.pool.sha256,methods=a.methods,masked=a.mask,
        selected_using_labels=False,formal_benchmark=False))
    rows=[]
    for method in a.methods:
        t=time.monotonic();print('START',method,flush=True)
        try:
            rep=fit_representation(task.input(method,a.mask),method,fitting_pool=task.pool,
                out=out/method,input_contract=task.receipt,family=f'trainability_v0.3:{task.name}')
            rows.append(dict(method=method,status='passed',shape=list(rep.x.shape),seconds=time.monotonic()-t))
        except Exception as e:rows.append(dict(method=method,status='failed',error=repr(e),seconds=time.monotonic()-t))
        print(rows[-1],flush=True)
        write_json(out/'SUMMARY.json',dict(results=rows,all_passed=all(r['status']=='passed' for r in rows),formal_benchmark=False),replace=True)
if __name__=='__main__':main()
