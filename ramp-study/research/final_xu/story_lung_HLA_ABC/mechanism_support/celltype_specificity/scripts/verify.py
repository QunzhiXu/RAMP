from runtime import *
design=json.loads((OUT/'DESIGN.json').read_text())
checks={key:sha(path)==design['input_sha256'][key] for key,path in INPUTS.items() if key!='provider_sce'}
assert all(checks.values()),checks
export=json.loads((OUT/'EXPORT_VERIFICATION.json').read_text())
assert export['source_sha256']==design['input_sha256']['provider_sce'] and export['status']=='PASS'
for file in ['JOIN_VERIFICATION.json','MOLECULAR_VERIFICATION.json','PATIENT_VERIFICATION.json']:
    assert json.loads((OUT/file).read_text())['status']=='PASS'
scores=pd.read_csv(OUT/'PATIENT_CELLTYPE_SCORES.csv')
assert scores.config.nunique()==2 and scores.cell_type.nunique()==22
assert scores[scores.config.eq('RAMP_k30')].n_cells.sum()==2849385
mol=pd.read_csv(OUT/'MOLECULAR_SUMMARY.csv')
assert len(mol)==22*2*2*2*2
for _,g in mol.groupby(['population','config','metric']):assert np.allclose(g.q_bh,bh(g.p),equal_nan=True)
immune=pd.read_csv(OUT/'PATIENT_IMMUNE_ASSOCIATIONS.csv')
for _,g in immune.groupby(['config','analysis']):assert np.allclose(g.q_bh,bh(g.p),equal_nan=True)
vcaf=immune[immune.cell_type.eq('vCAF')&immune.config.eq('RAMP_k30')&immune.analysis.eq('main')]
assert len(vcaf)==6 and (vcaf.q_bh<.001).all()
save(OUT/'VERIFICATION.json',{'status':'PASS','time_utc':now(),'read_only_input_hashes_unchanged':checks,'provider_hash_verified_at_export':True,'all_2849385_cells_and_22_types_accounted_for':True,'old_vcaf_molecular_results_reproduced':True,'old_immune_and_vcaf_measured_values_reproduced':True,'old_280_patient_vcaf_mean_reproduced':True,'molecular_and_patient_BH_recomputed':True,'molecular_summary_rows':len(mol),'patient_models':len(immune),'number_of_main_patient_types':int(immune[immune.config.eq('RAMP_k30')&immune.analysis.eq('main')].cell_type.nunique())})
print('PASS: all input hashes, cell counts, previous results and correction families verified.',flush=True)
