"""Independent arithmetic from saved predictions and frozen target/source answers.

No historical scores, runners, heads or alignment adapters are called here.
"""
from collections import Counter
import io
import gzip
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from common import *

LUNG_MAPPING={('Senosain CyTOF','CD4 T'):'CD4 T',('Senosain CyTOF','CD8 T'):'CD8 T',
              ('Cords IMC','CD4'):'CD4 T',('Cords IMC','CD8'):'CD8 T',('Zheng mIF','Th'):'CD4 T',
              ('Zheng mIF','Treg'):'CD4 T',('Zheng mIF','Tc'):'CD8 T',('Zheng mIF','DNT'):'DNT'}

def classification(truth, prediction, patients, classes):
    truth,prediction,patients=map(np.asarray,(truth,prediction,patients));c=len(classes)
    require(truth.ndim==1 and truth.shape==prediction.shape==patients.shape and len(truth)>0,'Classification shape')
    require(c==len(set(classes)) and set(truth)<=set(classes) and set(prediction)<=set(classes),'Classification ontology')
    require(not (set(patients)&{'','None','nan'}),'Missing patient')
    lookup={x:i for i,x in enumerate(classes)}
    pair=np.array([lookup[t]*c+lookup[p] for t,p in zip(truth,prediction)])
    matrices=[];detail=[]
    for patient in np.unique(patients):
        ix=np.flatnonzero(patients==patient)
        matrix=np.bincount(pair[ix],minlength=c*c).reshape(c,c)/len(ix)
        matrices.append(matrix);detail.append(dict(patient=str(patient),matrix=matrix.tolist()))
    cm=np.mean(matrices,axis=0);diag=cm.diagonal();den=cm.sum(0)+cm.sum(1);support=cm.sum(1)
    return dict(macro_f1=float(np.divide(2*diag,den,out=np.zeros(c),where=den>0).mean()),
                balanced_accuracy=float(np.divide(diag,support,out=np.zeros(c),where=support>0).mean()),
                accuracy=float(diag.sum()),confusion_matrix=cm.tolist(),patients=len(matrices),cells=len(truth),
                classes=list(classes),zero_division=0,patient_confusion_matrices=detail)

class Regression:
    def __init__(self, truth, units, patients, minimum_cells=30):
        truth=np.asarray(truth,dtype=float);units=np.asarray(units,dtype=str);patients=np.asarray(patients,dtype=str)
        require(truth.ndim==1 and truth.shape==units.shape==patients.shape and len(truth)>0,'Regression shape')
        require(np.isfinite(truth).all() and minimum_cells>=2,'Regression truth/support')
        require(not ((set(units)|set(patients))&{'','None','nan'}),'Missing unit/patient')
        self.n=len(truth);self.groups=[]
        for u in np.unique(units):
            ix=np.flatnonzero(units==u);ps=np.unique(patients[ix]);require(len(ps)==1,'Unit spans patients')
            ranks=rankdata(truth[ix],method='average')
            status='insufficient_truth_support' if len(ix)<minimum_cells else 'constant_truth' if np.ptp(truth[ix])==0 else 'ok'
            self.groups.append((ix,str(u),str(ps[0]),ranks,(ranks-.5)/len(ix),status))

    def score(self,prediction):
        pred=np.asarray(prediction,dtype=float);require(pred.shape==(self.n,),'Regression prediction shape')
        if not np.isfinite(pred).all():return dict(status='run_failure',reason='nonfinite_predictions',units=[])
        rows=[]
        for ix,u,p,ranks,percentiles,status in self.groups:
            values=pred[ix];r=dict(unit=u,patient=p,cells=len(ix),mae=float(np.abs(np.clip(values,0,1)-percentiles).mean()),rho=None,effective_rho=None)
            if status=='ok':
                if np.ptp(values)==0:status='constant_prediction';r['effective_rho']=0.0
                else:
                    rho=float(np.corrcoef(ranks,rankdata(values,method='average'))[0,1])
                    require(np.isfinite(rho),'Undefined rank correlation');r.update(rho=rho,effective_rho=rho)
            r['rho_status']=status;rows.append(r)
        def average(field,eligible=False):
            by_patient={}
            for r in rows:
                if r[field] is not None and (not eligible or r['effective_rho'] is not None):
                    by_patient.setdefault(r['patient'],[]).append(r[field])
            return (float(np.mean([np.mean(v) for v in by_patient.values()])) if by_patient else None),len(by_patient)
        rho,nr=average('effective_rho');mae,nm=average('mae');support,_=average('mae',True)
        return dict(status='ok',effective_spearman=rho,mae=mae,mae_rho_support=support,rho_patients=nr,mae_patients=nm,units=rows,
                    rho_status_counts=dict(Counter(r['rho_status'] for r in rows)))

def compare_tree(new, old, tolerance=2e-14, path='score'):
    """Require every field/support/status, not only the published scalar metrics."""
    if isinstance(new,dict):
        require(isinstance(old,dict) and new.keys()==old.keys(),'Score keys differ: '+path)
        return max((compare_tree(v,old[k],tolerance,path+'.'+k) for k,v in new.items()),default=0.0)
    if isinstance(new,list):
        require(isinstance(old,list) and len(new)==len(old),'Score lengths differ: '+path)
        return max((compare_tree(a,b,tolerance,path+f'[{i}]') for i,(a,b) in enumerate(zip(new,old))),default=0.0)
    if isinstance(new,float):
        require(isinstance(old,(int,float)) and np.isfinite(new) and np.isfinite(old),'Nonfinite score: '+path)
        diff=abs(new-old);require(diff<=tolerance,'Numerical score differs: '+path+' '+str(diff));return diff
    require(type(new)==type(old) and new==old,'Score field differs: '+path);return 0.0

class Cohort:
    def __init__(self,case,spec,source,records):
        self.spec=spec;self.case=case
        self.x=np.load(io.BytesIO(verified_bytes(source,spec['native'],records)),allow_pickle=False)
        self.md=pd.read_csv(io.BytesIO(verified_bytes(source,spec['metadata'],records)),compression='gzip',dtype=str,keep_default_na=False)
        if case=='lung_story':
            self.md=self.md.rename(columns={'cell_key':'cell_id'})
            self.md['patient_id']=self.md.dataset+'::'+self.md.patient_id
            self.md['identity_level']='patient'
            self.md['celltype']=[LUNG_MAPPING.get((d,p),b) for d,p,b in zip(self.md.dataset,self.md.provider_cell_type,self.md.broad_type)]
            require(int((self.md.dataset.eq('Senosain CyTOF')&self.md.parent_cluster.eq('C02')).sum())==3171,'Lung C02 scope changed')
        md=self.md
        require(md.cell_id.is_unique and self.x.shape==(len(md),len(spec['markers'])) and np.isfinite(self.x).all(),'Cohort dimensions/finite/IDs')
        require(digest(dict(pool_id=spec['pool_id'],cell_ids=list(md.cell_id),datasets=list(md.dataset),units=list(md.unit_key)))==spec['pool_sha256'],'Cohort pool differs')
        self.index={v:i for i,v in enumerate(md.cell_id)};self.arrays={c:md[c].to_numpy(dtype=str) for c in ('cell_id','dataset','unit_key','patient_id','celltype','identity_level')}
        self.targets={};self.baselines={}

    def target(self,target,marker):
        k=(target,marker)
        if k not in self.targets:
            mask=self.arrays['dataset']==target
            if not marker:mask &=np.isin(self.arrays['celltype'],self.spec['classes'])
            ix=np.flatnonzero(mask);require(len(ix)>0,'Empty target')
            level='patient' if (self.arrays['identity_level'][ix]=='patient').all() else 'unit_descriptive'
            patients=self.arrays['patient_id' if level=='patient' else 'unit_key'][ix]
            scorer=Regression(self.x[ix,self.spec['markers'].index(marker)],self.arrays['unit_key'][ix],patients) if marker else None
            self.targets[k]=(ix,level,patients,scorer)
        return self.targets[k]

    def check_ledger(self,row,ledger,plan):
        j=row['job'];marker=j.get('marker') or None
        require(ledger['sources']==[j['source']] and ledger['target']==j['target'] and j['source']!=j['target'],'Ledger direction')
        require(ledger['marker']==marker and ledger['seed']==j['seed'] and ledger['split_mode']==j['split_mode'],'Ledger task')
        require(ledger['pool_sha256']==self.spec['pool_sha256'] and digest(ledger['candidates'])==row['candidates_sha256'],'Ledger contract')
        require(ledger['mask']==([] if marker is None else [self.spec['canonical_markers'][marker]]),'Target mask')
        roles={k:ledger[k] for k in ('fit','validation','calibration')}
        require(digest(roles)==j['supervision_sha256'],'Supervision hash')
        flattened=sum(roles.values(),[])
        require(len(set(flattened))==len(flattened) and set(flattened)<=self.index.keys(),'Supervision overlap/unknown cell')
        ix=np.array([self.index[i] for i in flattened]);require((self.arrays['dataset'][ix]==j['source']).all(),'Target in source supervision')
        if not marker:require(np.isin(self.arrays['celltype'][ix],self.spec['classes']).all(),'Source outside class scope')
        require(len(roles['fit'])==j['fit_budget'] and len(roles['validation'])==j['validation_budget'],'Source budget')
        require(len(flattened)==j['total_supervision']==ledger['total_supervision'],'Total supervision')
        ids,level,patients,scorer=self.target(j['target'],marker)
        require(ledger['target_cell_ids']==self.arrays['cell_id'][ids].tolist(),'Target identity/order')
        identity='patient' if (self.arrays['identity_level'][ix[:len(roles['fit'])+len(roles['validation'])]]=='patient').all() else 'unit_descriptive'
        require(ledger['identity_level']==identity,'Source identity level')
        people=self.arrays['patient_id' if identity=='patient' else 'unit_key']
        require(ledger['supervised_identity_count']==len(set(people[ix])),'Supervised identity count')
        f=[self.index[i] for i in roles['fit']];v=[self.index[i] for i in roles['validation']]
        if j['split_mode'] in ('patient_disjoint_validation','unit_disjoint_descriptive_validation'):
            require(not(set(people[f])&set(people[v])),'Fit/validation identity overlap')
        if marker:
            cal=[self.index[i] for i in roles['calibration']];counts=Counter(self.arrays['unit_key'][cal])
            require(counts and set(counts.values())=={j['calibration_per_unit']},'Calibration unit budget')
        else:require(not roles['calibration'],'Unexpected J1 calibration')
        if plan is not None:
            for r,values in roles.items():require(values==plan.get(r,[]),'Frozen plan differs: '+r)
            if 'target_ids' in plan:require(plan['target_ids']==ledger['target_cell_ids'],'Plan target differs')
            if 'pool_sha256' in plan:require(plan['pool_sha256']==self.spec['pool_sha256'],'Plan pool differs')
        return ids,level,patients,scorer

    def baseline(self,ledger):
        """Source-only calibration then median source-fit percentile; no head fitting."""
        key=(ledger['marker'],digest({r:ledger[r] for r in ('fit','calibration')}))
        if key not in self.baselines:
            col=self.spec['markers'].index(ledger['marker']);cal=np.array([self.index[i] for i in ledger['calibration']]);fit=np.array([self.index[i] for i in ledger['fit']])
            cu=self.arrays['unit_key'][cal];fu=self.arrays['unit_key'][fit];values=np.empty(len(fit))
            for u in np.unique(fu):
                dist=np.sort(np.asarray(self.x[cal[cu==u],col],dtype=float));require(len(dist)>0,'Missing source calibration')
                mask=fu==u;y=self.x[fit[mask],col]
                values[mask]=(np.searchsorted(dist,y,'left')+np.searchsorted(dist,y,'right'))/(2*len(dist))
            self.baselines[key]=float(np.median(values))
        return self.baselines[key]

def score_job(row, cohort, source, records):
    from binding import check_job_proof
    check_job_proof(row,records,source)
    prefix=row['folder']+'/'
    ledger=bound_json(source,prefix+'ledger.json',records)
    receipt=bound_json(source,prefix+'predictions.json',records)
    require(phase1.bind(receipt['prediction_file'])==prefix+'predictions.npz','Prediction route differs')
    require(receipt['sha256']==records[prefix+'predictions.npz']['sha256'] and receipt['target_answers_included'] is False,'Prediction receipt differs')
    require(receipt['provenance']['ledger_sha256']==digest(ledger) and receipt['provenance']['pool_sha256']==cohort.spec['pool_sha256'],'Prediction provenance differs')
    plan=bound_json(source,row['plan'],records) if row['plan'] else None
    ix,level,patients,scorer=cohort.check_ledger(row,ledger,plan)
    with np.load(io.BytesIO(verified_bytes(source,prefix+'predictions.npz',records)),allow_pickle=False) as data:
        require(set(data.files)=={'cell_ids','predictions'},'Prediction NPZ keys')
        require(np.array_equal(data['cell_ids'],cohort.arrays['cell_id'][ix]),'Prediction row order')
        prediction=data['predictions'];require(prediction.shape==(len(ix),) and receipt['n_cells']==len(ix),'Prediction shape')
    if scorer:
        score=scorer.score(prediction)
        require(score['status']=='ok','Failed predictions')
        score['source_constant_baseline']=scorer.score(np.full(len(ix),cohort.baseline(ledger)))
    else:score=classification(cohort.arrays['celltype'][ix],prediction,patients,cohort.spec['classes'])
    # Historical scores are opened ONLY after new scores are computed.
    previous=bound_json(source,prefix+'RESULT.json',records)
    require(previous['prediction_receipt']==receipt and previous['selection']==receipt['provenance']['selection'],'Result prediction binding')
    require(previous['target_identity_level']==level,'Target identity level')
    for role,field in [('fit','fit_cells'),('validation','validation_cells'),('calibration','calibration_cells')]:require(previous[field]==len(ledger[role]),'Result supervision count')
    require(previous['total_supervision']==ledger['total_supervision'],'Result total supervision')
    delta=compare_tree(score,previous['score'])
    return dict(job=row['job'],score=score,max_abs_difference=delta,supervision_sha256=row['job']['supervision_sha256'],
                target_ids_sha256=digest(ledger['target_cell_ids']),prediction_sha256=receipt['sha256'],ledger_sha256=digest(ledger))
