"""Read-only check of CD19 availability and CytoNorm CD20 B-cell contrasts."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
reads={}
def track(p):
 p=p.resolve();reads[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
availability=[]
for case in ['bone_marrow','tnbc','melanoma','lung_cancer']:
 base=ROOT/'benchmark_results'/case
 markers=track(base/'ramp/alignment/input/markers.txt').read_text().splitlines()
 cpath=track(base/'cytonorm/alignment/data/corrected_expression.csv.gz')
 columns=pd.read_csv(cpath,nrows=0).columns.tolist()
 availability.append(dict(case=case,CD19_in_alignment='CD19' in markers,CD19_in_CytoNorm='CD19' in columns,
                          CD20_in_alignment='CD20' in markers,CD20_in_CytoNorm='CD20' in columns))
pd.DataFrame(availability).to_csv(HERE/'marker_availability.csv',index=False)
base=ROOT/'benchmark_results/tnbc';inp=base/'ramp/alignment/input'
m=pd.read_csv(track(inp/'cell_metadata.csv.gz'),low_memory=False,dtype={'cell_id':str})
markers=track(inp/'markers.txt').read_text().splitlines()
raw=np.load(track(inp/'raw.npy'))[:,markers.index('CD20')]
cp=base/'cytonorm/alignment/data/corrected_expression.csv.gz'
complete=json.loads(track(base/'cytonorm/COMPLETE.json').read_text())
assert reads[str(cp.resolve())]==complete['outputs']['alignment/data/corrected_expression.csv.gz']
c=pd.read_csv(cp,usecols=['cell_id','CD20'],dtype={'cell_id':str},float_precision='round_trip')
assert np.array_equal(c.cell_id,m.cell_id)
rows=[];box=[];units=[]
for unit,ix in m.groupby('unit_key').indices.items():
 mm=m.iloc[ix];lab=np.array(['Other']*len(ix),dtype=object)
 for cell in ['CD4 T','CD8 T']: lab[mm.celltype.eq(cell).to_numpy()]=cell
 lab[mm.provider_label_l1.isin(['B_cell','Bcell']).to_numpy()]='B cells'
 lab[mm.provider_label_l1.eq('Monocyte').to_numpy()]='Monocytes'
 lab[mm.provider_label_l1.eq('Mac').to_numpy()]='Macrophages'
 assert not np.any(mm.provider_label_l1.isin(['Plasma','plasma_cell']).to_numpy()&(lab=='B cells'))
 ranks={method:rankdata(v[ix])/(len(ix)+1) for method,v in [('Raw',raw),('CytoNorm',c.CD20.to_numpy())]}
 units.append(dict(unit_key=unit,n_cells=len(ix),n_B=int((lab=='B cells').sum()),
                   max_abs_CD20_percentile_change=float(np.max(np.abs(ranks['Raw']-ranks['CytoNorm'])))))
 for method,r in ranks.items():
  for cell in ['B cells','CD4 T','CD8 T','Monocytes','Macrophages']:
   vals=r[lab==cell]
   if len(vals)<50:continue
   q1,med,q3=np.quantile(vals,[.25,.5,.75])
   box.append(dict(unit_key=unit,method=method,celltype=cell,n_cells=len(vals),q1=q1,median=med,q3=q3))
 if (lab=='B cells').sum()<50:continue
 for other in ['CD4 T','CD8 T','Monocytes','Macrophages']:
  a=lab=='B cells';b=lab==other
  if b.sum()<50:continue
  d=dict(unit_key=unit,non_target=other,n_B=int(a.sum()),n_other=int(b.sum()))
  for method,r in ranks.items():
   joint=rankdata(np.r_[r[a],r[b]])
   d[method+'_auc']=(joint[:a.sum()].sum()-a.sum()*(a.sum()+1)/2)/(a.sum()*b.sum())
   d[method+'_B_median']=np.median(r[a]);d[method+'_other_median']=np.median(r[b])
  d['auc_loss']=d['Raw_auc']-d['CytoNorm_auc'];rows.append(d)
pd.DataFrame(rows).to_csv(HERE/'CD20_B_cell_contrasts.csv',index=False)
pd.DataFrame(box).to_csv(HERE/'CD20_box_statistics.csv',index=False)
pd.DataFrame(units).to_csv(HERE/'CD20_all_sample_rank_checks.csv',index=False)
assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in reads.items())
summary=dict(CD19_available=False,CD20_benchmark='tnbc',CD20_samples_checked=len(units),
             samples_with_B_ge_50=sum(v['n_B']>=50 for v in units),eligible_B_contrasts=len(rows),
             max_AUROC_loss=max(r['auc_loss'] for r in rows),
             max_abs_percentile_change=max(u['max_abs_CD20_percentile_change'] for u in units),
             minimum_cells_per_compared_population=50,Plasma_excluded_from_B=True,
             current_figure_changed=False,source_sha256=reads)
(HERE/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='source_sha256'},indent=2))
