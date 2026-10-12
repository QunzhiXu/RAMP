"""Extract fixed comparison predictions and CAF sample IDs from historical files."""
import package_runtime as rt
from package_runtime import BASE
import numpy as np
import pandas as pd


def main():
    old=BASE.parents[1]/'overnight0909'
    source=old/'cache/spatial_context_features.pkl'
    sample=pd.read_pickle(source)[['source_row_id']]
    assert len(sample)==150024 and sample.source_row_id.is_unique
    sample.to_csv(BASE/'reference/THRESHOLD_CAF_CELL_IDS.csv.gz',index=False,
                  compression={'method':'gzip','compresslevel':3,'mtime':0})
    path=old/'results/vcaf_projected_dfs/ALL_CORDS_PREDICTIONS.csv.gz'
    configs=['RAMP_k30','RAMP_k100','baseline_k30','baseline_k100']
    oldpred=pd.read_csv(path,usecols=['source_row_id',*configs],float_precision='round_trip')
    assert len(oldpred)==2849385 and oldpred.source_row_id.is_unique
    np.savez_compressed(BASE/'reference/PREVIOUS_ALL_PREDICTIONS.npz',
                        source_row_id=oldpred.source_row_id.to_numpy(),prediction=oldpred[configs].to_numpy())
    rt.save_json(BASE/'reference/IMPORT_MANIFEST.json',{'time_utc':rt.now(),
                 'sources':{str(p):rt.sha(p) for p in [source,path]},'configs':configs,
                 'sample_ids_only':True,'T_neighborhood_fields_used':False})
    print('Fixed CAF sample IDs and all previous predictions saved.',flush=True)


if __name__=='__main__':main()
