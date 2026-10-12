"""Freeze endpoints and ID selection before reading new protein measurements."""
import runtime as rt
from runtime import BASE, INPUTS
import pandas as pd
import numpy as np

out=BASE/'results/DESIGN.json'
if out.exists():raise SystemExit('Frozen design already exists; do not overwrite.')
md=pd.read_csv(INPUTS['metadata'],low_memory=False)
assert len(md)==2849385 and md.patient_id.nunique()==527 and md.source_row_id.is_unique
v=md.cell_type.eq('vCAF')
t=md.cell_category.eq('T cell') & md.cell_type.isin(['CD8','CD4'])
target=md[v|t].copy()
assert v.sum()==52328 and (md.cell_type.eq('CD8')&t).sum()==108943
target.to_csv(BASE/'data/MEASUREMENT_REQUEST.csv.gz',index=False,compression={'method':'gzip','mtime':0})
clinical=pd.read_csv(INPUTS['clinical'])
clinical=clinical[clinical.config.eq('RAMP_k30')].copy()
assert len(clinical)==280 and clinical.patient_id.is_unique and clinical.event.sum()==96
clinical.to_csv(BASE/'data/DFS_280.csv',index=False)
design={
 'time_utc':rt.now(),'status':'frozen_before_new_marker_read',
 'population':'All 527 existing Cords patients; author-annotated vCAF, CD8 and CD4. Primary biological summaries use all available eligible patients. Repeat using IDs of the fixed 280-patient DFS cohort.',
 'model':'Existing all-source RAMP_k30; k100 sensitivity. No new alignment, prediction, patient selection, or survival threshold.',
 'primary_endpoints':['vCAF_CD73','vCAF_IDO','nearby_CD8_PD1','nearby_CD8_TCF1'],
 'expected_directions':{'vCAF_CD73':1,'vCAF_IDO':1,'nearby_CD8_PD1':1,'nearby_CD8_TCF1':-1},
 'primary_self':'Within each patient x ROI with >=15 vCAF, Spearman between continuous predicted HLA-ABC and each measured marker; equal-weight mean ROI correlations within each patient.',
 'primary_spatial':'Each CD8 T cell assigned once to its nearest author-vCAF in the same patient x ROI. Primary maximum distance 40 in original provider coordinate units; repeat at 20 and 80. Within-ROI Spearman between nearest-vCAF HLA score and measured CD8 PD1/TCF1; >=15 assigned CD8 and >=5 distinct nearest vCAF; ROI correlations averaged equally within patient.',
 'nearby_state_companion':'Mean of within-ROI CD8 percentile-rank(PD1) minus percentile-rank(TCF1), an explicitly operational continuous index, not an exhaustion diagnosis.',
 'contrast':'For display, compare high/low vCAF HLA thirds using quantiles 2/3 and 1/3 of all vCAF scores in the ROI; ties retained; disjoint cutoffs required. At least 5 cells per contrasted group; CD8 values are within-ROI ranks among all CD8. Average ROI contrasts equally within patient. These are within-ROI thirds, not the patient survival high/low split.',
 'inference':'Patients are independent units; 3000 patient bootstrap resamples; two-sided paired Wilcoxon on patient mean ROI rho. BH across four primary endpoints separately for full available and DFS280 membership. Secondary endpoints/configs/radii reported completely and separately.',
 'secondary':'Measured FOXP3 in nearest-vCAF-assigned CD4; author Treg indicator among CD4. PD1-minus-TCF1 rank index in CD8. Patient mean vCAF CD73/IDO versus patient mean HLA and fixed-cohort DFS (original covariates), explicitly a separate between-patient level.',
 'measurement':'Provider SCE counts assay; original row IDs are 1-based. New measured targets CD73, IDO, PD1, TCF1, FOXP3 are absent from current nine model inputs. Shared marker values exported only for all-cell identity/value checks.',
 'missing_core_mechanism_readouts':['measured Cords HLA-ABC','PD-L1','PD-L2','FASL','IFN-gamma','direct T-cell killing or apoptosis assay'],
 'reporting':'No effect-size or p-value threshold optimized; failed or opposite-direction results retained. Marker association and spatial association cannot demonstrate causal pathway or antigen-specific killing.',
 'input_sha256':{k:rt.sha(p) for k,p in INPUTS.items()},
 'n_request':len(target),'request_by_type':target.cell_type.value_counts().to_dict(),
 'request_sha256':rt.sha(BASE/'data/MEASUREMENT_REQUEST.csv.gz'),
 'seed':20260909,
}
rt.save(out,design)
print('Design frozen; measurement request',len(target),'cells',flush=True)
