"""Count-only audit before changing any supervised sampling or ontology."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import json
import numpy as np
import pandas as pd
from trainability.core import file_sha256, seed_for
from trainability.sampling import source_split,nested_draws

TASKS=['bone_marrow','tnbc','melanoma','lung_cancer','lung_story','melanoma_story','hnscc_story']
PILOT=ROOT/'exploration_donor_cap20_20260921'

def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def main():
    rows=[];patients=[];pools=[];inputs={};verified=0;split_records=[]
    pilot=json.loads((PILOT/'MANIFEST.json').read_text())
    for name in TASKS:
        folder=ROOT/f'data/real_inputs_v0_3_r2/{name}'
        manifest=json.loads((folder/'manifest.json').read_text())
        classes=manifest['classes']
        config_path=ROOT/f'config/formal_v0_5/{name}__ramp.json'
        cfg=json.loads(config_path.read_text())
        jobs=[j for j in cfg['jobs'] if j['task_type']=='J1' and 'primary' in j['analysis_scopes']]
        variants=[('original_full',folder/'metadata.csv.gz',jobs,Path(cfg['output'])/'jobs')]
        if name in pilot['tasks']:
            variants.append(('cap20',PILOT/f'inputs/{name}/metadata.csv.gz',pilot['tasks'][name]['jobs'],
                PILOT/f'jobs/{name}/refit_cap20'))
        for p in [folder/'manifest.json',config_path]:inputs[str(p)]=file_sha256(p)
        for variant,path,js,resultroot in variants:
            inputs[str(path)]=file_sha256(path)
            md=pd.read_csv(path,dtype=str,keep_default_na=False).set_index('cell_id')
            eligible=md[md.celltype.isin(classes)]
            assert set(classes)<=set.intersection(*(set(g.celltype) for _,g in eligible.groupby('dataset')))
            for ds,g in eligible.groupby('dataset'):
                for cl in classes:
                    group=g[g.celltype.eq(cl)]
                    counts=group.groupby('patient_id').size()
                    pools.append(dict(task=name,pool=variant,dataset=ds,celltype=cl,cells=len(group),
                        patients=len(counts),largest_patient_fraction=float(counts.max()/counts.sum())))
                    for patient,n in counts.items():patients.append(dict(task=name,pool=variant,dataset=ds,
                        celltype=cl,patient=patient,cells=int(n)))
            by_source={}
            for j in js:
                key=(j['source'],j['seed'])
                if key in by_source:
                    assert by_source[key]['supervision_sha256']==j['supervision_sha256'];continue
                by_source[key]=j
            for (ds,seed),j in by_source.items():
                ids=tuple(eligible[eligible.dataset.eq(ds)].index)
                reviewed=md.loc[list(ids),'identity_level'].eq('patient').all()
                identities=md.patient_id.to_dict() if reviewed else md.unit_key.to_dict()
                split=source_split(ids,identities,.2,seed,patient_mapping_verified=True)
                mode=split['mode'] if reviewed else 'unit_disjoint_descriptive_validation'
                assert mode==j['split_mode']
                actual_fit=nested_draws(split['fit_pool'],[j['fit_budget']],seed_for(seed,'fit'))[j['fit_budget']]
                actual_val=nested_draws(split['validation_pool'],[j['validation_budget']],seed_for(seed,'validation'))[j['validation_budget']]
                ledger_path=resultroot/j['id']/'ledger.json';inputs[str(ledger_path)]=file_sha256(ledger_path)
                ledger=json.loads(ledger_path.read_text())
                assert list(actual_fit)==ledger['fit'] and list(actual_val)==ledger['validation']
                assert not set(actual_fit)&set(actual_val)
                fp=md.loc[list(split['fit_pool'])];vp=md.loc[list(split['validation_pool'])]
                af=md.loc[list(actual_fit)];av=md.loc[list(actual_val)]
                if mode=='patient_disjoint_validation':assert not set(fp.patient_id)&set(vp.patient_id)
                split_records.append(dict(task=name,pool=variant,source=ds,seed=seed,mode=mode,
                    source_patients=len(set(md.loc[list(ids),'patient_id'])),fit_patients=fp.patient_id.nunique(),
                    validation_patients=vp.patient_id.nunique(),fit_pool_cells=len(fp),validation_pool_cells=len(vp),
                    fit_budget=j['fit_budget'],validation_budget=j['validation_budget']))
                for cl in classes:
                    f=fp[fp.celltype.eq(cl)];v=vp[vp.celltype.eq(cl)]
                    a=af[af.celltype.eq(cl)];b=av[av.celltype.eq(cl)]
                    rows.append(dict(task=name,pool=variant,source=ds,seed=seed,celltype=cl,
                        fit_pool_cells=len(f),validation_pool_cells=len(v),
                        fit_pool_patients=f.patient_id.nunique(),validation_pool_patients=v.patient_id.nunique(),
                        actual_fit_cells=len(a),actual_validation_cells=len(b),
                        actual_fit_patients=a.patient_id.nunique(),actual_validation_patients=b.patient_id.nunique(),
                        fit_floor20_possible=len(f)>=20,validation_floor5_possible=len(v)>=5,
                        joint_fit20_val5_possible=len(f)>=20 and len(v)>=5,
                        class_count=len(classes),total_fit_floor=20*len(classes),
                        floor_within_budget=20*len(classes)<=j['fit_budget'],mode=mode))
                verified+=1
            print(name,variant,'checked',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'PER_CLASS_SOURCE_SPLIT.csv',index=False)
    pd.DataFrame(patients).to_csv(OUT/'PER_PATIENT_CLASS_COUNTS.csv',index=False)
    pd.DataFrame(pools).to_csv(OUT/'POOL_CLASS_COUNTS.csv',index=False)
    pd.DataFrame(split_records).to_csv(OUT/'SPLIT_OVERVIEW.csv',index=False)
    summary=[]
    for (name,pool),g in frame.groupby(['task','pool']):
        summary.append(dict(task=name,pool=pool,source_seed_splits=len(g[['source','seed']].drop_duplicates()),
            minimum_fit_pool=int(g.fit_pool_cells.min()),minimum_validation_pool=int(g.validation_pool_cells.min()),
            minimum_actual_fit=int(g.actual_fit_cells.min()),minimum_actual_validation=int(g.actual_validation_cells.min()),
            source_class_seed_cases=len(g),actual_fit_zero_cases=int(g.actual_fit_cells.eq(0).sum()),
            actual_validation_zero_cases=int(g.actual_validation_cells.eq(0).sum()),
            fit20_impossible_cases=int((~g.fit_floor20_possible).sum()),
            val5_impossible_cases=int((~g.validation_floor5_possible).sum()),
            fit20_possible_all=bool(g.fit_floor20_possible.all()),
            fit20_val5_possible_all=bool(g.joint_fit20_val5_possible.all())))
    pd.DataFrame(summary).to_csv(OUT/'SUMMARY.csv',index=False)
    keys=['task','pool','source','celltype']
    grouped=frame.groupby(keys).agg(fit_pool_min=('fit_pool_cells','min'),fit_pool_max=('fit_pool_cells','max'),
        validation_pool_min=('validation_pool_cells','min'),validation_pool_max=('validation_pool_cells','max'),
        actual_fit_min=('actual_fit_cells','min'),actual_fit_max=('actual_fit_cells','max'),
        actual_validation_min=('actual_validation_cells','min'),actual_validation_max=('actual_validation_cells','max'),
        minimum_fit_patients=('fit_pool_patients','min'),minimum_validation_patients=('validation_pool_patients','min')).reset_index()
    grouped.to_csv(OUT/'CLASS_SUPPORT_RANGES.csv',index=False)
    blockers=frame[~frame.joint_fit20_val5_possible]
    blockers.to_csv(OUT/'INFEASIBLE_CASES.csv',index=False)
    for p,h in inputs.items():assert file_sha256(p)==h,p
    save('CHECKS.json',dict(verified_saved_supervision_splits=verified,
        protected_files_unchanged=True,input_sha256=inputs,
        no_alignment_or_model_fitting=True,no_new_sample_selected=True,no_scores_recomputed=True,
        no_changes_to_labels_or_patient_splits=True,
        thresholds_are_diagnostic_not_established_standards=dict(fit_per_class=20,validation_per_class=5)))
    print(pd.DataFrame(summary).to_string(index=False))
    print('\nImpossible cases:\n',blockers[['task','pool','source','seed','celltype','fit_pool_cells','validation_pool_cells']].to_string(index=False))

if __name__=='__main__':main()
