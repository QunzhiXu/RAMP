"""J1/J2/J3 execution: source-only selection, freeze predictions, then score.

Explicit plans are reusable across methods. A failed or infeasible job has no
substitute score. This module does not authorize a formal run by itself.
"""
from __future__ import annotations
from dataclasses import asdict
import time
import numpy as np
from .core import require, write_json, output_path, seed_for, digest
from .sampling import Supervision, nested_draws, source_split, reserve_calibration, matched_j3, validate_j3_supervision
from .validation import SourceValidation
from .targets import SourceCalibration, constant_mae_baseline
from .heads import Candidate, fit_head
from .scoring import freeze_predictions,load_frozen_predictions,classification_score,regression_score


def eligible_ids(task,dataset,classification):
    md=task.metadata
    mask=md.dataset.eq(dataset)
    if classification:mask &= md.celltype.isin(task.receipt['classes'])
    return tuple(md.loc[mask,'cell_id'])


def make_plan(task,source,target,*,fit_budget,validation_budget,seed,marker=None,calibration_per_unit=0,
              labelled_patient_count=None):
    require(source!=target and source in task.metadata.dataset.values and target in task.metadata.dataset.values,'Bad direction')
    md=task.metadata.set_index('cell_id')
    ids=eligible_ids(task,source,marker is None)
    patients=md.patient_id.to_dict();units=md.unit_key.to_dict()
    if labelled_patient_count is not None:
        require(md.loc[list(ids),'identity_level'].eq('patient').all(),
                'Labelled-patient curve requires a verified patient crosswalk')
        people=sorted({patients[i] for i in ids})
        selected=set(nested_draws(people,[labelled_patient_count],seed_for(seed,'labelled_patients'))[labelled_patient_count])
        ids=tuple(i for i in ids if patients[i] in selected)
    calibration=()
    if marker is not None:
        require(marker in task.markers,'Unknown target marker')
        reservation=reserve_calibration(ids,units,calibration_per_unit,seed,minimum_remaining=2)
        calibration=reservation['calibration'];ids=reservation['remaining']
    reviewed=md.loc[list(ids),'identity_level'].eq('patient').all()
    if reviewed:
        split=source_split(ids,patients,.2,seed,patient_mapping_verified=True)
    else:
        # Explicit unit split; do not call it patient-disjoint validation.
        split=source_split(ids,units,.2,seed,patient_mapping_verified=True)
        split['mode']='unit_disjoint_descriptive_validation'
    fit=nested_draws(split['fit_pool'],[fit_budget],seed_for(seed,'fit'))[fit_budget]
    val=nested_draws(split['validation_pool'],[validation_budget],seed_for(seed,'validation'))[validation_budget]
    plan=Supervision(fit,val,calibration)
    plan.assert_allowed(eligible_ids(task,source,marker is None),eligible_ids(task,target,marker is None))
    return plan,split['mode']


def make_j3_plans(task,a,b,target,*,fit_quotas,validation_quotas,seed):
    require(len({a,b,target})==3,'J3 needs disjoint source A/B/target datasets')
    md=task.metadata.set_index('cell_id');labels=md.celltype.to_dict()
    splits=[]
    for ds in [a,b]:
        ids=eligible_ids(task,ds,True)
        require(md.loc[list(ids),'identity_level'].eq('patient').all(),'J3 patient mapping unresolved')
        splits.append(source_split(ids,md.patient_id.to_dict(),.2,seed,patient_mapping_verified=True))
    fits=matched_j3(splits[0]['fit_pool'],splits[1]['fit_pool'],labels,fit_quotas,seed_for(seed,'fit'))
    vals=matched_j3(splits[0]['validation_pool'],splits[1]['validation_pool'],labels,validation_quotas,seed_for(seed,'validation'))
    plans={k:Supervision(fits[k],vals[k]) for k in fits}
    validate_j3_supervision(plans,labels,eligible_ids(task,a,True),eligible_ids(task,b,True),eligible_ids(task,target,True))
    return plans


def run_supervised(task,representation,plan,*,sources,target,marker,candidates,seed,out,split_mode):
    out=output_path(out);require(not out.exists(),'Head output already exists')
    require(representation.fitting_pool_sha256==task.pool.sha256,'Head representation is not frozen full U')
    require(not set(sources).intersection([target]),'Source/target overlap')
    expected_mask=[] if marker is None else [task.receipt['canonical_markers'][marker]]
    require(tuple(representation.masked_markers)==tuple(expected_mask),
            'Representation mask differs from requested prediction task')
    md=task.metadata.set_index('cell_id')
    source_ids=tuple(i for s in sources for i in eligible_ids(task,s,marker is None))
    target_ids=eligible_ids(task,target,marker is None)
    require(bool(target_ids),'Empty fixed target support')
    plan.assert_allowed(source_ids,target_ids)
    require(set(plan.fit+plan.validation+plan.calibration)<=set(representation.cell_ids),'Plan outside full U')
    classes=tuple(task.receipt['classes']) if marker is None else ()
    kind='classification' if marker is None else 'regression'
    source_md=md.loc[list(plan.fit+plan.validation)]
    identity_level='patient' if source_md.identity_level.eq('patient').all() else 'unit_descriptive'
    pats=(md.patient_id if identity_level=='patient' else md.unit_key).to_dict()
    if marker is None:
        ys=md.loc[list(plan.fit),'celltype'].to_numpy(dtype=str)
        yv=md.loc[list(plan.validation),'celltype'].to_numpy(dtype=str);measured=None
    else:
        # Only source answers are extracted before prediction freezing.
        lookup={x:i for i,x in enumerate(task.pool.cell_ids)}
        col=task.markers.index(marker)
        values=lambda ids: task.native[[lookup[i] for i in ids],col]
        cal=SourceCalibration.fit(plan.calibration,values(plan.calibration),
            md.loc[list(plan.calibration),'unit_key'],plan.calibration)
        ys=cal.transform(plan.fit,values(plan.fit),md.loc[list(plan.fit),'unit_key'])
        yv=cal.transform(plan.validation,values(plan.validation),md.loc[list(plan.validation),'unit_key'])
        measured=dict(zip(plan.validation,values(plan.validation)))
    val=SourceValidation.from_cell_maps(cell_ids=plan.validation,x=representation.select(plan.validation),
        targets_by_cell=dict(zip(plan.validation,yv)),patients_by_cell={i:pats[i] for i in plan.validation},
        units_by_cell={i:md.at[i,'unit_key'] for i in plan.validation},
        measured_by_cell=measured,task=kind,classes=classes,identity_level=identity_level,
        identity_reviewed=identity_level=='patient')
    val.assert_scoreable()
    out.mkdir(parents=True)
    ledger=dict(sources=sources,target=target,fit=list(plan.fit),validation=list(plan.validation),
        calibration=list(plan.calibration),total_supervision=plan.total,marker=marker,
        pool_sha256=task.pool.sha256,split_mode=split_mode,identity_level=identity_level,
        candidates=[asdict(c) for c in candidates],seed=seed,mask=expected_mask,
        supervised_identity_count=len({pats[i] for i in plan.fit+plan.validation+plan.calibration}),
        target_cell_ids=list(target_ids),formal_benchmark=not task.receipt.get('pilot',False))
    write_json(out/'ledger.json',ledger)
    start=time.monotonic()
    head=fit_head(representation.select(plan.fit),ys,val,fit_ids=plan.fit,
        task=kind,candidates=candidates,classes=classes,seed=seed)
    training_seconds=time.monotonic()-start
    inference_start=time.monotonic()
    prediction=head.predict(representation.select(target_ids))
    inference_seconds=time.monotonic()-inference_start
    if marker is None:prediction=np.asarray(prediction,dtype=str)
    receipt=freeze_predictions(out/'predictions.npz',target_ids,prediction,
        dict(ledger_sha256=digest(ledger),pool_sha256=task.pool.sha256,selection=head.selection))
    frozen=load_frozen_predictions(receipt,target_ids)
    target_md=md.loc[list(target_ids)]
    target_level='patient' if target_md.identity_level.eq('patient').all() else 'unit_descriptive'
    target_pat=target_md.patient_id if target_level=='patient' else target_md.unit_key
    if marker is None:score=classification_score(target_md.celltype.to_numpy(),frozen,target_pat.to_numpy(),classes,include_sufficient_stats=True)
    else:
        truth=values(target_ids)  # first access to target A in this runner
        score=regression_score(truth,frozen,target_md.unit_key.to_numpy(),target_pat.to_numpy())
        baseline=constant_mae_baseline(ys)
        score['source_constant_baseline']=regression_score(truth,np.repeat(baseline,len(truth)),
            target_md.unit_key.to_numpy(),target_pat.to_numpy())
    result=dict(score=score,target_identity_level=target_level,selection=head.selection,
        candidates=head.candidates,elapsed_seconds=time.monotonic()-start,
        fit_cells=len(plan.fit),validation_cells=len(plan.validation),calibration_cells=len(plan.calibration),
        total_supervision=plan.total,prediction_receipt=receipt)
    result.update(training_seconds_including_selection=training_seconds,inference_seconds=inference_seconds)
    write_json(out/'RESULT.json',result)
    return result
