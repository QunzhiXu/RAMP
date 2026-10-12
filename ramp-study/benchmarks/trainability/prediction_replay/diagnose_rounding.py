"""Explain the pre-existing Lung J1 RESULT-versus-report-table floating rounding."""
import io
import numpy as np
import pandas as pd
from common import *
import binding
from scores import Cohort

def main():
    source=PARENT.parents[2]/'final_xu';index=HERE/'bindings/current_predictions_20261011/catalog.json'
    anchor='82eafb7e4e41652b23844b53247081e3242163cc606004771734d2ba51ea7c4d'
    dest=output(HERE/'validation/lung_reporting_rounding_20261011',source,[index])
    try:
        catalog=binding.load(index,anchor);records=catalog['files'];cohort=Cohort('lung_story',catalog['cohorts']['lung_story'],source,records)
        table=pd.read_csv(SNAPSHOT/'tree'/phase1.ROOT/phase1.UPDATES['lung_story']/'SCORES_PER_JOB.csv',float_precision='round_trip')
        table=table[table.arm.eq('keep_c02')].set_index(['method','source','target','seed','family'])
        rows=[]
        for r in catalog['jobs']:
            j=r['job']
            if j['task']!='lung_story' or j['task_type']!='J1':continue
            ix,level,patients,_=cohort.target(j['target'],None);classes=cohort.spec['classes']
            people=pd.Series(patients);counts=people.value_counts();weights=1/people.map(counts).to_numpy(float)/len(counts)
            truth=pd.Categorical(cohort.arrays['celltype'][ix],categories=classes).codes
            with np.load(io.BytesIO(verified_bytes(source,r['folder']+'/predictions.npz',records)),allow_pickle=False) as data:
                require(np.array_equal(data['cell_ids'],cohort.arrays['cell_id'][ix]),'Target changed')
                pred=pd.Categorical(data['predictions'],categories=classes).codes
            c=len(classes);cm=np.bincount(c*truth+pred,weights=weights,minlength=c*c).reshape(c,c)
            tp=cm.diagonal();den=cm.sum(0)+cm.sum(1);support=cm.sum(1)
            weighted=dict(macro_f1=float(np.divide(2*tp,den,out=np.zeros(c),where=den>0).mean()),
                          balanced_accuracy=float(np.divide(tp,support,out=np.zeros(c),where=support>0).mean()),accuracy=float(tp.sum()/cm.sum()))
            old_result=bound_json(source,r['folder']+'/RESULT.json',records)['score']
            published=table.loc[(j['method'],j['source'],j['target'],j['seed'],j['family'])]
            for metric,value in weighted.items():
                rows.append(dict(job_id=j['id'],method=j['method'],metric=metric,historical_weighted_formula=value,
                                 published_table=float(published['patient_'+metric]),patient_matrix_result=old_result[metric],
                                 table_difference=abs(value-published['patient_'+metric]),result_difference=abs(value-old_result[metric])))
        frame=pd.DataFrame(rows);require(len(frame)==1620 and frame.table_difference.max()==0,'Rounding explanation does not reproduce full table')
        with (dest/'comparison.csv').open('x') as stream:frame.to_csv(stream,index=False)
        report=dict(status='passed',jobs=540,metric_rows=len(frame),historical_formula_vs_table_max_abs_difference=float(frame.table_difference.max()),
                    historical_formula_vs_RESULT_max_abs_difference=float(frame.result_difference.max()),
                    explanation='Historical Lung report accumulates per-cell 1/n_patient/n_patients weights in bincount and normalizes trace by total mass for accuracy; RESULT and independent replay average per-patient normalized count matrices. Algebraically equivalent but floating addition/division order differs.',
                    execution_or_scorer_changed=False,code=record(HERE/'diagnose_rounding.py'))
        write(dest/'report.json',report);write(dest/'COMPLETE.json',dict(status='complete',files={p.name:record(p) for p in dest.iterdir()}));print(report)
    except BaseException as e:write(dest/'FAILED.json',dict(error=repr(e)));raise

if __name__=='__main__':main()
