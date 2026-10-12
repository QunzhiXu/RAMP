"""Quantify the neighborhood composition in the displayed 175_10 ROI."""
import runtime as rt
from runtime import BASE
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
OUT=BASE/'results'
PID='175_10';ROI='175_A3,3'
p=OUT/'MAP_CONTEXT.csv.gz';original_hash=rt.sha(p)
c=pd.read_csv(p,float_precision='round_trip')
c=c[c.patient_id.eq(PID)&c.roi_id.eq(ROI)].copy()
v=c[c.cell_type.eq('vCAF')].copy();other=c[~c.cell_type.eq('vCAF')].copy()
assert len(v)==230
labels=other.cell_category.replace({'Fibroblast':'Other CAF','vessel':'Vessel'}).to_numpy()
xy=other[['x','y']].to_numpy();query=v[['x','y']].to_numpy();tree=cKDTree(xy)
cuts=pd.read_csv(BASE.parent/'data_cords/results/HIGH_STATE_THRESHOLDS.csv',float_precision='round_trip').set_index('config').threshold
cellrows=[]
for radius in [20,40,80]:
    for row,(indices,q) in enumerate(zip(tree.query_ball_point(query,radius),query)):
        indices=np.asarray(indices,dtype=int)
        indices=indices[np.sqrt(((xy[indices]-q)**2).sum(axis=1))<radius]
        counts={lab:int(np.sum(labels[indices]==lab)) for lab in ['Other CAF','Tumour','Vessel','T cell','Immune','Other']}
        for k in [30,100]:
            score=v[f'RAMP_k{k}'].iloc[row]
            for lab,n in counts.items():
                cellrows.append(dict(patient_id=PID,roi_id=ROI,source_row_id=v.source_row_id.iloc[row],config=f'RAMP_k{k}',
                    state='high' if score>=cuts[f'RAMP_k{k}'] else 'low',radius=radius,target=lab,n_neighbors=len(indices),
                    count=n,fraction=n/len(indices) if len(indices) else np.nan))
cells=pd.DataFrame(cellrows)
assert np.allclose(cells[cells.n_neighbors.gt(0)].groupby(['source_row_id','config','radius']).fraction.sum(),1)
cells.to_csv(OUT/'EXAMPLE_175_10_NEIGHBOR_COMPOSITION_CELLS.csv.gz',index=False)
s=cells.groupby(['config','radius','state','target']).agg(mean_fraction=('fraction','mean'),mean_count=('count','mean'),n_vcaf=('source_row_id','size'),n_vcaf_with_neighbors=('fraction','count')).reset_index()
s.to_csv(OUT/'EXAMPLE_175_10_NEIGHBOR_COMPOSITION.csv',index=False)
rt.save_json(OUT/'EXAMPLE_175_10_NEIGHBOR_COMPOSITION_DESIGN.json',dict(patient_id=PID,roi_id=ROI,
    input=str(p),input_sha256=original_hash,description='Mean per-vCAF fraction of other cells within a fixed centroid radius. All author vCAFs excluded from the neighbor denominator. Strict distance <radius. Empty neighborhoods omitted from mean fraction, retained in counts. Immune here is the provider non-T immune category.',
    radii=[20,40,80],cutoffs={k:float(cuts[k]) for k in ['RAMP_k30','RAMP_k100']},no_refitting=True,source_unchanged=rt.sha(p)==original_hash))
print(s[s.config.eq('RAMP_k30')&s.radius.eq(40)].pivot(index='target',columns='state',values='mean_fraction').to_string())
