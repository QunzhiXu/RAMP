"""Read-only per-job comparison with the exact historical reporting inputs."""
import argparse
import pandas as pd
from common import *
import binding
import replay
import contract

KEYS=['task','task_type','method','marker','direction','family','seed','metric']

def compare(catalog,run,anchor):
    verified=replay.verify_run(run,catalog,anchor);root=SNAPSHOT/'tree'/phase1.ROOT
    selected={r['job']['id'] for r in catalog['jobs'] if r['manifest'] is None}
    base=contract.frame(root/phase1.BASE/'metrics.csv');pieces=[base[base.job_id.isin(selected)].copy()]
    for case,dirname in phase1.UPDATES.items():
        df=contract.frame(root/dirname/'SCORES_PER_JOB.csv')
        if case=='lung_story':df=df[df.arm=='keep_c02'].copy()
        if case=='hnscc_story':df=df[df.cohort=='full'].copy()
        df['task']=case;df['task_type']='J1';df['marker']='';df['direction']=df.source+' -> '+df.target
        for metric in ('macro_f1','balanced_accuracy','accuracy'):
            x=df.copy();x['metric']=metric;x['score']=x['patient_'+metric];pieces.append(x)
    df=contract.frame(root/phase1.J2/'summary/SCORES_PER_JOB.csv')
    for metric in ('effective_spearman','mae'):
        x=df.copy();x['metric']=metric;x['score']=x[metric];pieces.append(x)
    expected=pd.concat(pieces,ignore_index=True)[KEYS+['score']]
    actual=pd.concat([contract.frame(run/'branches'/f'{kind}__{case}'/'metrics.csv') for case,kind in replay.branch_names(catalog)],ignore_index=True)[KEYS+['score']]
    whole=contract.compare_scores(actual,expected,KEYS)
    a=actual.set_index(KEYS).sort_index();b=expected.set_index(KEYS).sort_index()
    table=a.rename(columns={'score':'new_score'}).join(b.rename(columns={'score':'historical_score'}))
    table['abs_difference']=(table.new_score-table.historical_score).abs();table=table.reset_index()
    groups=table.groupby(['task_type','task'],as_index=False).agg(rows=('abs_difference','size'),max_abs_difference=('abs_difference','max'))
    return dict(status='passed',scope='All per-job published metric inputs; no fitting',run=verified,comparison=whole,
                exact_equal_rows=int(table.abs_difference.eq(0).sum()),groups=groups.to_dict('records')),table

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',type=Path,required=True);p.add_argument('--anchor',required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    dest=output(a.output,Path(read(a.run/'REQUEST.json')['source_root']),[a.index,a.run])
    try:
        report,table=compare(binding.load(a.index,a.anchor),a.run,a.anchor)
        with (dest/'per_job_comparison.csv').open('x') as f:table.to_csv(f,index=False)
        write(dest/'report.json',report);write(dest/'COMPLETE.json',dict(status='complete',files={p.name:record(p) for p in dest.iterdir()}))
        print(json.dumps(report,indent=2))
    except BaseException as e:write(dest/'FAILED.json',dict(error=repr(e)));raise

if __name__=='__main__':main()
