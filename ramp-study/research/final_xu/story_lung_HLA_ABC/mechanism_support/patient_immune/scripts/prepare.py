from runtime import *
hashes={k:sha(p) for k,p in INPUTS.items()}
clinical=pd.read_csv(INPUTS['clinical'],float_precision='round_trip')
assert len(clinical)==280 and clinical.patient_id.is_unique
assert clinical.group.value_counts().to_dict()=={'Higher':140,'Lower':140} or clinical.group.value_counts().to_dict()=={'Lower':140,'Higher':140}
if (OUT/'DESIGN.json').exists():
    assert json.loads((OUT/'DESIGN.json').read_text())['input_sha256']==hashes
else:
    save(OUT/'DESIGN.json',{'time_utc':now(),'input_sha256':hashes,'population':'Exact original 280 DFS patients and fixed 140/140 labels. All available author-annotated immune cells in sampled cores; no nearest-vCAF assignment.',
    'primary_expression':PRIMARY,'secondary_expression':SECONDARY,
    'expression':'Within each author lineage, mean per-cell asinh(provider counts) over every available cell per patient. Patients equally weighted; >=1 cell primary, >=20 and >=50 sensitivity. Do not impute expression for absent lineages.',
    'primary_adjustment':'High minus low HLA group, OLS with HC3 standard errors, adjusted for TMA, age, sex and categorical Stage. Outcome standardized by between-patient SD. Two-sided; BH over all 10 primary endpoints, including the 2 previous CD8 endpoints. No preferred direction.',
    'secondary_adjustment':'Same model, BH across 10 secondary endpoints separately. Treg = author CD4_Treg, conventional CD4 = remaining author CD4. HLA_DR was a prediction input and is not an independent validation readout.',
    'composition':'Five disjoint lineages CD8, CD4, Bcell, Myeloid, Neutrophil: fractions of all immune cells and fractions of all cells; total immune fraction of all cells; author CD4_Treg fraction of CD4. Zeros retained. Separate BH family across all 12 composition endpoints.',
    'sensitivity':['Minimum 20 or 50 cells in analyzed lineage; same group labels','Equal ROI means instead of cell-weighted patient means','Additional original CAF covariates','Continuous HLA score instead of binary split','Conventional CD4 versus author Treg secondary subgroup analysis'],
    'interpretation':'PD1 and TCF1 interpreted within T cells. CD73/IDO can describe expression states, but no M2 assignment from CD68, no exhaustion call from PD1 alone, no causal local mechanism from whole-patient associations.',
    'seed':SEED})
meta=pd.read_csv(INPUTS['metadata'])
req=meta[meta.patient_id.isin(clinical.patient_id)&meta.cell_type.isin(LINEAGES)].copy()
assert req.source_row_id.is_unique and req.patient_id.nunique()==280
req.to_csv(DATA/'MEASUREMENT_REQUEST.csv.gz',index=False,compression={'method':'gzip','mtime':0})
clinical.to_csv(DATA/'FROZEN_PATIENT_GROUPS.csv',index=False)
req.groupby(['cell_type','cell_subtype']).size().rename('n_cells').to_csv(OUT/'REQUEST_COVERAGE.csv')
print(f'Design frozen; requesting {len(req):,} immune cells from all 280 patients.',flush=True)
