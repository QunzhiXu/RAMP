from runtime import *
hashes={k:sha(p) for k,p in INPUTS.items()}
meta=pd.read_csv(INPUTS['metadata'])
assert len(meta)==2849385 and meta.patient_id.nunique()==527 and meta.source_row_id.is_unique
types=meta.groupby(['cell_category','cell_type']).size().rename('n_cells').reset_index()
types.to_csv(OUT/'CELL_TYPES.csv',index=False)
if (OUT/'DESIGN.json').exists():
    assert json.loads((OUT/'DESIGN.json').read_text())['input_sha256']==hashes
else:
    save(OUT/'DESIGN.json',{'time_utc':now(),'input_sha256':hashes,'cell_types':types.to_dict('records'),
    'scope':'All 2,849,385 already-projected Cords cells from 527 patients; all 22 author cell_type groups. Other is retained as heterogeneous/unclassified. Tumour normal denotes the author non-hypoxic tumour label, not normal lung tissue.',
    'model':'Existing RAMP k30 predictions; k100 sensitivity. No new alignment, transfer, threshold search, survival fitting, or patient selection.',
    'within_ROI':'Exactly match previous vCAF design: within patient x ROI x cell_type, >=15 cells; Spearman predicted HLA vs measured CD73 or IDO. Patient mean of ROI correlations, then patient-equal cohort summary. 3000 patient-bootstrap CI; two-sided Wilcoxon; BH across all 22x2 endpoints, separately for configuration and population.',
    'contrast':'Within each eligible ROI and cell type, bottom/top predicted-HLA thirds; disjoint cutoffs, >=5 cells in each. Measured-marker ranks are within the same ROI and cell type, rank/(n+1). Compare rank means and then average ROIs equally within patient. These are not the patient survival cutoffs.',
    'populations':['All eligible patients among 527','Original 280 DFS cohort membership, unchanged'],
    'matched_specificity':'For every other type and marker, compare its patient-level within-ROI rho to vCAF on shared patients. Bootstrap paired patient differences, Wilcoxon; BH across the 21x2 comparisons. Also restrict to the same patient and ROI where both types meet >=15 cells.',
    'patient_immune':'Use original 280 DFS patients, but each candidate type uses its own continuous patient-mean predicted HLA score (>=15 cells/type/patient). Six frozen readouts: total immune/all cells; CD4, B, neutrophil fractions of immune; measured CD8 and CD4 TCF1 patient means. Standardize predictor and outcome, adjust TMA, age, sex and categorical stage, HC3 SE. BH across all tested type x endpoint combinations. Descriptive own-median splits are not used for inference.',
    'patient_sensitivity':['At least 50 cells of the predictor type','Add predictor-type abundance to covariates','Compare candidate and vCAF on exactly the same eligible patients; candidate score additionally adjusted for vCAF score'],
    'source_readouts':'CD73 and IDO from original counts assay, held out of the prediction inputs. CD3, HLA_DR are extracted only to verify shared-marker values. Neither measured HLA-ABC nor causal perturbation is available.',
    'seed':SEED})
meta[['source_row_id','patient_id','roi_id','x','y','cell_category','cell_type','cell_subtype','matrix_row_index']].to_csv(DATA/'MEASUREMENT_REQUEST.csv.gz',index=False,compression={'method':'gzip','mtime':0})
print('Frozen all 22 cell types, both molecular endpoints and six patient immune readouts.',flush=True)
