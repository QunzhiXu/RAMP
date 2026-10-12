"""Small masked-protein model with four readouts and matched update budgets.

The model is deliberately small and is not a foundation model benchmark.
Only the representation transform/scaler sees unlabeled target X; the encoder
optimizer sees source corpus cells only. All readout choices use source labels.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, dataclass
import time
import numpy as np
from .core import require,seed_for,write_json,output_path
from .sampling import nested_draws,validate_pretraining_sets
from .pretraining import make_encoder,make_mask,masked_mse
from .heads import Candidate,fit_head
from .validation import SourceValidation
from .scoring import freeze_predictions,load_frozen_predictions,classification_score
from .runner import eligible_ids,make_plan


@dataclass(frozen=True)
class PretrainConfig:
    width: int = 32
    corpus_per_half: int = 1000
    pretrain_validation_per_source: int = 100
    probe_fit: int = 500
    probe_validation: int = 200
    batch_size: int = 128
    pretrain_updates: int = 300
    supervised_updates: int = 300
    validation_every: int = 20
    learning_rate: float = .001
    mask_fraction: float = .3

    def validate(self):
        for k,v in asdict(self).items():
            if k not in {'learning_rate','mask_fraction'}:require(isinstance(v,int) and v>0,f'Invalid {k}')
        require(0<self.mask_fraction<1 and self.learning_rate>0,'Invalid pretraining hyperparameters')


def source_validation_diagnostics(history,selected_updates,executed_updates,update_budget):
    """Describe observed source checks; checkpoint location is not convergence.

    No target score or assumption of monotonic learning enters these flags.
    A last-step optimum can be budget-sensitive; an earlier optimum can reflect
    noise or overfitting. Neither is a convergence certificate.
    """
    require(bool(history) and 0<selected_updates<=executed_updates<=update_budget,
            'Invalid source checkpoint history/budget')
    steps=[r['updates'] for r in history]
    scores=np.asarray([r['source_macro_f1'] for r in history],dtype=float)
    require(steps==sorted(set(steps)) and steps[0]>0 and steps[-1]==executed_updates
            and np.isfinite(scores).all(),'Incomplete or invalid source validation history')
    require(selected_updates==steps[int(np.argmax(scores))],
            'Selected checkpoint differs from first source-validation optimum')
    delta=None if len(scores)<2 else float(scores[-1]-scores[-2])
    return dict(selected_at_final_update=selected_updates==executed_updates,
        selected_at_update_budget=selected_updates==update_budget,
        update_budget_exhausted=executed_updates==update_budget,
        validation_checks=len(history),best_source_validation_score=float(scores.max()),
        last_source_validation_score=float(scores[-1]),last_interval_score_change=delta,
        last_interval_improved=None if delta is None else delta>0,
        best_minus_last_source_validation_score=float(scores.max()-scores[-1]),
        updates_after_selected=executed_updates-selected_updates,
        convergence_status='not_assessed_fixed_update_budget')


def make_plans(task,a,b,target,cfg,seed):
    require(len({a,b,target})==3,'J4 source addition needs three dataset roles')
    probe,mode=make_plan(task,a,target,fit_budget=cfg.probe_fit,
        validation_budget=cfg.probe_validation,seed=seed)
    blocked=set(probe.fit+probe.validation)
    aa=tuple(i for i in eligible_ids(task,a,False) if i not in blocked)
    bb=eligible_ids(task,b,False)
    v=cfg.pretrain_validation_per_source;n=cfg.corpus_per_half
    drawa=nested_draws(aa,[2*n+v],seed_for(seed,'corpusA'))[2*n+v]
    drawb=nested_draws(bb,[n+v],seed_for(seed,'corpusB'))[n+v]
    ca,va=drawa[:2*n],drawa[2*n:]
    cb,vb=drawb[:n],drawb[n:]
    # Same total corpus N and pretraining validation N across conditions.
    # Both conditions evaluate reconstruction on A-only validation, fixed cells.
    # B validation is reserved and unused, never silently put into its corpus.
    plans={'A-only':dict(corpus=ca,pretrain_validation=va),
           'A+B':dict(corpus=ca[:n]+cb,pretrain_validation=va)}
    for arm,p in plans.items():
        allowed=eligible_ids(task,a,False)+(eligible_ids(task,b,False) if arm=='A+B' else ())
        validate_pretraining_sets(p['corpus'],p['pretrain_validation'],probe,allowed,eligible_ids(task,target,False))
    return probe,mode,plans,dict(reserved_unused_B_validation=list(vb))


def run_j4(task,rep,*,source_a,source_b,target,cfg:PretrainConfig,seed,out):
    import torch
    from torch import nn
    cfg.validate()
    require(rep.method not in {'maxfuse','fastmnn'},'Embedding dimensions are not maskable proteins')
    require(rep.fitting_pool_sha256==task.pool.sha256 and not rep.masked_markers,'J4 needs registered complete marker U')
    require(rep.x.shape[1]==len(task.markers),'J4 feature axes differ from proteins')
    out=output_path(out);require(not out.exists(),'J4 run exists');out.mkdir(parents=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    probe,mode,plans,unused=make_plans(task,source_a,source_b,target,cfg,seed)
    md=task.metadata.set_index('cell_id');classes=tuple(task.receipt['classes'])
    require(md.loc[list(probe.fit+probe.validation),'identity_level'].eq('patient').all(),'J4 probe patient mapping unresolved')
    # Fixed label-free scaling over full U, once per method, outside corpus loop.
    center=rep.x.mean(axis=0);sd=rep.x.std(axis=0);sd=np.where(sd>0,sd,1.)
    x=((rep.x-center)/sd).astype('float32')
    require(np.isfinite(x).all(),'Nonfinite J4 scaling')
    index={i:j for j,i in enumerate(rep.cell_ids)}
    def select(ids):return x[[index[i] for i in ids]]
    target_ids=eligible_ids(task,target,True)
    fy=md.loc[list(probe.fit),'celltype'].to_numpy(dtype=str)
    vy=md.loc[list(probe.validation),'celltype'].to_numpy(dtype=str)
    def validation(values):
        return SourceValidation.from_cell_maps(cell_ids=probe.validation,x=values,
            targets_by_cell=dict(zip(probe.validation,vy)),
            patients_by_cell=md.loc[list(probe.validation),'patient_id'].to_dict(),
            task='classification',classes=classes,identity_level='patient',identity_reviewed=True)
    vx=validation(select(probe.validation));vx.assert_scoreable()
    readout_groups={arm:dict(input_linear='input_linear',random_supervised='random_supervised',
        frozen_probe=arm+'_frozen_probe',finetuned=arm+'_finetuned') for arm in plans}
    ledger=dict(config=asdict(cfg),seed=seed,pool_sha256=task.pool.sha256,
        source_a=source_a,source_b=source_b,target=target,probe=asdict(probe),
        split_mode=mode,conditions=plans,reserved=unused,readout_groups=readout_groups,
        scaler_center=center.tolist(),scaler_sd=sd.tolist(),
        scaler_scope='fixed full U, transductive allowed X only',formal_benchmark=not task.receipt.get('pilot',False))
    write_json(out/'ledger.json',ledger)
    receipts={};logs={};start=time.monotonic()
    def freeze(name,pred):
        receipts[name]=freeze_predictions(out/(name+'.npz'),target_ids,np.asarray(pred,dtype=str),
            dict(readout=name,pool_sha256=task.pool.sha256,source_a=source_a,source_b=source_b,config=asdict(cfg)))
    def linear(fitx,valx,tx,name):
        started=time.monotonic()
        h=fit_head(fitx,fy,validation(valx),fit_ids=probe.fit,task='classification',
            candidates=[Candidate('linear',logistic_c=c) for c in [.1,1.,10.]],classes=classes,seed=seed)
        freeze(name,h.predict(tx));logs[name]=dict(selection=h.selection,candidates=h.candidates,
            seconds=time.monotonic()-started,comparison_role='diagnostic_readout',
            optimization_protocol='logistic_solver_stopping_rules',
            optimization_diagnostics=h.selection['optimization_diagnostics'])
    linear(select(probe.fit),select(probe.validation),select(target_ids),'input_linear')
    zero=lambda t:torch.zeros_like(t,dtype=torch.bool)
    def encode(model,ids):
        model.eval();chunks=[]
        with torch.no_grad():
            for k in range(0,len(ids),4096):
                t=torch.from_numpy(select(ids[k:k+4096]));chunks.append(model.encode(t,zero(t)).numpy())
        return np.concatenate(chunks)
    def supervised(name,pretrained=None):
        # Random and fine-tuned models use exactly the same architecture, head
        # initialization, probe cells, batches and supervised-update budget.
        started=time.monotonic()
        torch.manual_seed(seed);m=make_encoder(len(task.markers),cfg.width)
        if pretrained is not None:m.load_state_dict(pretrained)
        torch.manual_seed(seed_for(seed,'head'));head=nn.Linear(cfg.width,len(classes))
        params=list(m.encoder.parameters())+list(head.parameters())
        opt=torch.optim.Adam(params,lr=cfg.learning_rate)
        tx=torch.from_numpy(select(probe.fit));vv=torch.from_numpy(select(probe.validation))
        yy=torch.tensor([classes.index(c) for c in fy],dtype=torch.long)
        vy_tensor=torch.tensor([classes.index(c) for c in vy],dtype=torch.long)
        rng=np.random.default_rng(seed_for(seed,'supervised'));best=None;hist=[]
        for step in range(1,cfg.supervised_updates+1):
            ix=rng.choice(len(tx),min(cfg.batch_size,len(tx)),replace=False)
            m.train();head.train();opt.zero_grad()
            loss=nn.functional.cross_entropy(head(m.encode(tx[ix],zero(tx[ix]))),yy[ix])
            require(bool(torch.isfinite(loss)),'Nonfinite supervised loss');loss.backward();opt.step()
            if step%cfg.validation_every==0 or step==cfg.supervised_updates:
                m.eval();head.eval()
                with torch.no_grad():
                    logits=head(m.encode(vv,zero(vv)))
                    pred=np.asarray(classes)[logits.argmax(1).numpy()]
                    validation_loss=float(nn.functional.cross_entropy(logits,vy_tensor))
                score=vx.score(pred,cell_ids=probe.validation)
                hist.append(dict(updates=step,source_macro_f1=score,
                    batch_training_loss_before_update=float(loss.detach()),
                    source_validation_cross_entropy=validation_loss))
                if best is None or score>best[0]:best=(score,step,deepcopy(m.state_dict()),deepcopy(head.state_dict()))
        m.load_state_dict(best[2]);head.load_state_dict(best[3]);m.eval();head.eval()
        with torch.no_grad():pred=np.asarray(classes)[head(torch.from_numpy(encode(m,target_ids))).argmax(1).numpy()]
        freeze(name,pred)
        torch.save(dict(encoder=best[2],head=best[3]),out/(name+'.pt'))
        logs[name]=dict(executed_updates=cfg.supervised_updates,selected_updates=best[1],
            source_validation=best[0],trainable_parameters=sum(p.numel() for p in params),source_history=hist,
            seconds=time.monotonic()-started,comparison_role='matched_supervised_primary_control',
            optimization_protocol='adam_fixed_supervised_updates',
            pretrained_initialization=pretrained is not None,
            diagnostics=source_validation_diagnostics(hist,best[1],cfg.supervised_updates,cfg.supervised_updates))
    # Corpus-independent reference: one fit, one checkpoint, one receipt and one
    # score. Both corpus conditions reference it explicitly, not as replicates.
    supervised('random_supervised')
    for arm,plan in plans.items():
        armstart=time.monotonic();torch.manual_seed(seed)
        model=make_encoder(len(task.markers),cfg.width)
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg.learning_rate)
        corpus=select(plan['corpus']);pv=torch.from_numpy(select(plan['pretrain_validation']))
        pm=torch.from_numpy(make_mask(tuple(pv.shape),cfg.mask_fraction,seed_for(seed,'pretrain_validation_mask')))
        rng=np.random.default_rng(seed);history=[]
        for step in range(1,cfg.pretrain_updates+1):
            ix=rng.choice(len(corpus),min(cfg.batch_size,len(corpus)),replace=False)
            t=torch.from_numpy(corpus[ix]);mask=torch.from_numpy(make_mask(tuple(t.shape),cfg.mask_fraction,seed_for(seed,str(step))))
            model.train();optimizer.zero_grad();loss=masked_mse(model(t,mask),t,mask)
            require(bool(torch.isfinite(loss)),'Nonfinite pretraining loss');loss.backward();optimizer.step()
            if step%cfg.validation_every==0 or step==cfg.pretrain_updates:
                model.eval()
                with torch.no_grad():lv=float(masked_mse(model(pv,pm),pv,pm))
                history.append(dict(updates=step,source_reconstruction_loss=lv))
        pretrained=deepcopy(model.state_dict())
        torch.save(pretrained,out/(arm+'_pretrained.pt'))
        logs[arm+'_pretrain']=dict(executed_updates=cfg.pretrain_updates,
            parameter_count=sum(p.numel() for p in model.parameters()),source_history=history,
            seconds=time.monotonic()-armstart,checkpoint_policy='last fixed update; no target selection')
        linear(encode(model,probe.fit),encode(model,probe.validation),encode(model,target_ids),arm+'_frozen_probe')
        supervised(arm+'_finetuned',pretrained)
    # All predictions have been sealed before ANY target labels enter scoring.
    target_md=md.loc[list(target_ids)];scores={}
    level='patient' if target_md.identity_level.eq('patient').all() else 'unit_descriptive'
    patient=target_md.patient_id if level=='patient' else target_md.unit_key
    for name,receipt in receipts.items():
        scores[name]=classification_score(target_md.celltype.to_numpy(),load_frozen_predictions(receipt,target_ids),patient.to_numpy(),classes)
    result=dict(schema_version='trainability.j4.result.v0.4',scores=scores,
        readout_groups=readout_groups,
        shared_baseline=dict(readout='random_supervised',shared_across=list(plans),
            independent_fits=1,independent_replicates=False,cost_record='training.random_supervised'),
        comparison_scope=dict(primary='finetuned vs shared random_supervised within each representation',
            diagnostic='input_linear and frozen_probe use different optimization protocols; not matched-update controls',
            corpus_contrast='FT(A+B)-FT(A-only); subtracting the shared random baseline cancels identically',
            representation_interaction='Compare (FT-RS) across input-processing methods; the RS baseline remains method-specific'),
        target_identity_level=level,training=logs,elapsed_seconds=time.monotonic()-start,
        total_probe_supervision=probe.total,receipts=receipts)
    write_json(out/'RESULT.json',result)
    return result
