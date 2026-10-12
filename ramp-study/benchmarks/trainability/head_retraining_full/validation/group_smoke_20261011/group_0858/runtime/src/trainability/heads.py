"""Common heads with ID-bound source validation and canonical candidate selection.

MLP partial_fit traverses one epoch of minibatches. We retain this behavior and
record optimizer updates separately from epochs, including work after the best
checkpoint. Target answers never enter this interface.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, asdict
import math
import time
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from .core import canonical_json, require, unique_ids
from .validation import SourceValidation

TIE_BREAK = 'maximum_source_score_then_effective_configuration_lexicographic'


@dataclass(frozen=True)
class Candidate:
    family: str
    scale: str = 'identity'
    logistic_c: float = 1.0
    ridge_alpha: float = 1.0
    mlp_alpha: float = 1.0
    neighbors: int = 15
    weights: str = 'uniform'
    width: int = 32
    learning_rate: float = 0.001
    max_epochs: int = 100
    patience: int = 15
    batch_size: int = 128

    def effective(self, task: str, n_fit: int) -> dict:
        require(task in {'classification','regression'} and n_fit > 0, 'Invalid head task/support')
        require(self.family in {'linear','knn','mlp'}, 'Unknown head family')
        require(self.scale in {'identity','standard'}, 'Unknown head scaler')
        e = dict(family=self.family, scale=self.scale)
        if self.family == 'linear':
            if task == 'classification':
                require(np.isfinite(self.logistic_c) and self.logistic_c > 0, 'Invalid logistic C')
                e.update(C=float(self.logistic_c), max_iter=2000)
            else:
                require(np.isfinite(self.ridge_alpha) and self.ridge_alpha >= 0, 'Invalid Ridge alpha')
                e.update(alpha=float(self.ridge_alpha))
        elif self.family == 'knn':
            require(isinstance(self.neighbors,int) and self.neighbors > 0, 'Invalid kNN k')
            require(self.weights in {'uniform','distance'}, 'Invalid kNN weights')
            e.update(neighbors=min(self.neighbors,n_fit), weights=self.weights)
        else:
            require(all(isinstance(v,int) and v > 0 for v in
                        (self.width,self.max_epochs,self.patience,self.batch_size)), 'Invalid MLP capacity/budget')
            require(np.isfinite(self.mlp_alpha) and self.mlp_alpha >= 0, 'Invalid MLP alpha')
            require(np.isfinite(self.learning_rate) and self.learning_rate > 0, 'Invalid learning rate')
            e.update(width=self.width, alpha=float(self.mlp_alpha), learning_rate=float(self.learning_rate),
                     max_epochs=self.max_epochs, patience=self.patience,
                     batch_size=min(n_fit,self.batch_size), solver='adam')
        return e


class ConstantClassifier:
    def __init__(self,label):
        self.label=label
        self.classes_=np.asarray([label])

    def predict(self,x):
        return np.repeat(self.label,len(x))

    def predict_proba(self,x):
        return np.ones((len(x),1))


@dataclass
class FittedHead:
    model: object
    scaler: object | None
    task: str
    classes: tuple[str,...]
    selection: dict
    candidates: list[dict]

    def predict(self,x):
        x=np.asarray(x,dtype=float)
        require(x.ndim==2 and np.isfinite(x).all(),'Invalid prediction features')
        return self.model.predict(x if self.scaler is None else self.scaler.transform(x))

    def predict_proba(self,x):
        require(self.task=='classification','Regression has no class probabilities')
        x=np.asarray(x,dtype=float)
        require(x.ndim==2 and np.isfinite(x).all(),'Invalid prediction features')
        if self.scaler is not None: x=self.scaler.transform(x)
        raw=self.model.predict_proba(x)
        out=np.zeros((len(x),len(self.classes)))
        for j,label in enumerate(self.model.classes_):
            out[:,self.classes.index(str(label))]=raw[:,j]
        return out


def parameter_accounting(model,family: str,fit_x: np.ndarray) -> dict:
    if isinstance(model,ConstantClassifier):
        return dict(trainable_head_parameters=0,constant_classifier_fallback=True)
    if family=='knn':
        return dict(trainable_head_parameters=None,stored_training_cells=len(fit_x),
                    feature_storage_bytes=int(fit_x.nbytes),
                    note='Feature storage only; not a total-memory measurement')
    if family=='mlp': count=sum(x.size for x in model.coefs_+model.intercepts_)
    else: count=np.asarray(model.coef_).size+np.asarray(model.intercept_).size
    return dict(trainable_head_parameters=int(count),constant_classifier_fallback=False)


def fit_head(fit_x,fit_y,validation: SourceValidation,*,fit_ids,task: str,
             candidates: list[Candidate],classes=(),seed: int=0) -> FittedHead:
    """Selection is fixed to patient/unit-aware source metrics, without callbacks.

    Validation answers and identities are assembled by cell ID. No target input
    is accepted until the returned head's predict call. Formal orchestration
    must separately verify source role membership against the supervision ledger.
    """
    require(isinstance(validation,SourceValidation),'Expected ID-bound source validation')
    require(task==validation.task,'Training and validation tasks differ')
    fi=unique_ids(fit_ids)
    vi=validation.cell_ids
    require(not set(fi).intersection(vi),'Source fit/validation overlap')
    fx=np.asarray(fit_x,dtype=float); fy=np.asarray(fit_y)
    vx=validation.x
    require(fx.ndim==2 and fx.shape[1]==vx.shape[1] and len(fx)==len(fi)==len(fy)
            and len(fi)>0 and fy.ndim==1,'Head features/answers/identities do not match')
    require(np.isfinite(fx).all(),'Nonfinite head input')
    classes=unique_ids(classes,'ontology classes')
    if task=='classification':
        require(classes==validation.classes and set(fy)<=set(classes),'Ontology mismatch')
    else:
        require(not classes,'Regression has no class ontology')
        fy=fy.astype(float)
        require(np.isfinite(fy).all(),'Nonfinite regression targets')
    validation.assert_scoreable()
    require(bool(candidates),'No source-selection candidates')
    grouped={}
    for candidate in candidates:
        effective=candidate.effective(task,len(fi))
        key=canonical_json(effective)
        grouped.setdefault(key,dict(effective=effective,requested=[]))['requested'].append(asdict(candidate))
    records=[]; best=None
    # Sort independently of incoming list/dict/set order; ties also compare this key explicitly.
    for key,item in sorted(grouped.items()):
        e=item['effective']; family=e['family']
        start=time.perf_counter()
        scaler=StandardScaler().fit(fx) if e['scale']=='standard' else None
        x=fx if scaler is None else scaler.transform(fx)
        v=vx if scaler is None else scaler.transform(vx)
        single=task=='classification' and len(np.unique(fy))==1
        if single: model=ConstantClassifier(fy[0])
        elif family=='linear':
            model=(LogisticRegression(C=e['C'],max_iter=e['max_iter'],random_state=seed)
                   if task=='classification' else Ridge(alpha=e['alpha']))
        elif family=='knn':
            cls=KNeighborsClassifier if task=='classification' else KNeighborsRegressor
            model=cls(n_neighbors=e['neighbors'],weights=e['weights'],n_jobs=1)
        else:
            cls=MLPClassifier if task=='classification' else MLPRegressor
            model=cls(hidden_layer_sizes=(e['width'],),random_state=seed,batch_size=e['batch_size'],
                      early_stopping=False,learning_rate_init=e['learning_rate'],alpha=e['alpha'],solver='adam')
        epochs=0; selected_epoch=None; updates=0; selected_updates=None
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            if family=='mlp' and not single:
                local_best=None; stale=0
                for epoch in range(e['max_epochs']):
                    before=int(model._optimizer.t) if hasattr(model,'_optimizer') else 0
                    if task=='classification': model.partial_fit(x,fy,classes=np.asarray(classes))
                    else: model.partial_fit(x,fy)
                    after=int(model._optimizer.t)
                    require(after-before==math.ceil(len(fi)/e['batch_size']),
                            'Unexpected optimizer update count; recheck installed sklearn behavior')
                    updates+=after-before; epochs+=1
                    score=validation.score(model.predict(v),cell_ids=vi)
                    require(np.isfinite(score),'Source validation score is nonfinite')
                    if local_best is None or score>local_best[0]:
                        local_best=(float(score),deepcopy(model),epoch+1,updates)
                        stale=0
                    else: stale+=1
                    if stale>=e['patience']: break
                score,model,selected_epoch,selected_updates=local_best
            else:
                if not single: model.fit(x,fy)
                score=validation.score(model.predict(v),cell_ids=vi)
                require(np.isfinite(score),'Source validation score is nonfinite')
        optimization_diagnostics=dict(protocol='not_logistic_solver',status='not_assessed_here')
        if family=='linear' and task=='classification' and not single:
            n_iter=np.atleast_1d(model.n_iter_).astype(int)
            limit=bool(np.any(n_iter>=e['max_iter']))
            warned=any(isinstance(w.message,ConvergenceWarning) for w in captured)
            optimization_diagnostics=dict(protocol='sklearn_logistic_solver',
                iterations=n_iter.tolist(),iteration_limit=e['max_iter'],
                iteration_limit_reached=limit,convergence_warning=warned,
                status='warning_or_iteration_limit' if warned or limit
                       else 'solver_stopped_without_convergence_warning')
        elif single:
            optimization_diagnostics=dict(protocol='constant_classifier',status='not_applicable')
        record=dict(candidate=e,requested_candidates=sorted(item['requested'],key=canonical_json),
                    source_validation_score=float(score),seconds=time.perf_counter()-start,
                    epochs_executed=epochs,selected_epoch=selected_epoch,
                    gradient_updates_executed=updates if family=='mlp' and not single else None,
                    selected_gradient_updates=selected_updates,
                    samples_seen_executed=epochs*len(fi) if family=='mlp' and not single else None,
                    selected_samples_seen=selected_epoch*len(fi) if selected_epoch else None,
                    validation_identity_sha256=validation.identity_sha256,
                    validation_identity_level=validation.identity_level,
                    selection_metric='macro_f1' if task=='classification' else 'effective_spearman',
                    tie_break=TIE_BREAK,warnings=[str(w.message) for w in captured],
                    warning_types=[type(w.message).__name__ for w in captured],
                    optimization_diagnostics=optimization_diagnostics,
                    **parameter_accounting(model,family,x))
        records.append(record)
        if best is None or score>best[0] or (score==best[0] and key<best[1]):
            best=(score,key,model,scaler,record)
    return FittedHead(best[2],best[3],task,classes,best[4],records)
