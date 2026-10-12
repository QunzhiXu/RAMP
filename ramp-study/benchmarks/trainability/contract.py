"""Read-only reaggregation of existing Figure 3 scores. Never imports a historical runner."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
import inventory as inv
from safe_io import require, read, write, reserve, member

METHODS = {'ramp', 'unaligned', 'clipped_zscore', 'minmax', 'percentile',
           'harmony_direct', 'fastmnn', 'cycombine', 'cytonorm', 'maxfuse'}
FAMILIES = {'linear', 'knn', 'mlp'}
SEEDS = {20260919, 20260920, 20260921}
KEYS = ['task', 'task_type', 'family', 'metric', 'method']
GRID = ['method', 'family', 'seed', 'source', 'target']


def frame(path):
    return pd.read_csv(path, float_precision='round_trip', keep_default_na=False, low_memory=False)


def exact_grid(actual, expected, keys):
    def tuples(df):
        require(not df.duplicated(keys).any(), 'Duplicate job identities')
        return set(df[keys].itertuples(index=False, name=None))
    require(tuples(actual) == tuples(expected), 'Incomplete/extra job grid')


def finite(values):
    x = np.asarray(values, dtype=float)
    require(np.isfinite(x).all(), 'Nonfinite scores')
    return x


def compare_scores(actual, expected, keys=KEYS, tolerance=2e-14):
    exact_grid(actual, expected, keys)
    a = actual.set_index(keys).sort_index(); b = expected.set_index(keys).sort_index()
    delta = float(np.max(np.abs(finite(a.score) - finite(b.score))))
    require(delta <= tolerance, 'Score aggregation differs: ' + str(delta))
    for column in ('rows', 'rank'):
        if column in a and column in b:
            require(np.array_equal(a[column], b[column]), 'Count/rank differs')
    return {'rows': len(a), 'max_abs_difference': delta, 'tolerance': tolerance}


def nested(group):
    # Historical reporting.nested_average: repeats, markers, directions, tasks.
    x = group.copy()
    for keep in (['task', 'direction', 'marker'], ['task', 'direction'], ['task']):
        x = x.groupby(keep, as_index=False, sort=True, dropna=False).score.mean()
    return float(x.score.mean())


def aggregate(rows):
    result = []
    for (task, kind, family, metric, method), g in rows.groupby(KEYS, sort=True):
        result.append(dict(task=task, task_type=kind, family=family, metric=metric, method=method,
                           score=nested(g), rows=len(g)))
    model = pd.DataFrame(result)
    totals = model.groupby(['task_type', 'family', 'metric', 'method'], as_index=False).agg(score=('score', 'mean'), rows=('rows', 'sum'))
    totals['task'] = 'ALL'
    return pd.concat([model, totals], ignore_index=True)


def inspect(snapshot, anchor, source=None):
    verified = inv.verify(snapshot, anchor, source)
    root = Path(snapshot) / 'tree' / inv.ROOT
    suite = read(root / 'config/formal_v0_5/suite.json')
    configs = [read(Path(snapshot) / 'tree' / inv.bind(s['config'])) for s in suite['shards']]
    original_jobs = [j for c in configs for j in c['jobs'] if 'primary' in j['analysis_scopes'] and j['task_type'] in {'J1', 'J2'}]
    jobs = pd.DataFrame(original_jobs).fillna({'marker': ''})
    require(not jobs.id.duplicated().any(), 'Duplicate formal primary ID')
    selected = jobs[((jobs.task_type == 'J1') & (jobs.task == 'bone_marrow')) |
                    ((jobs.task_type == 'J2') & (jobs.task != 'lung_story'))].copy()
    metrics = frame(root / inv.BASE / 'metrics.csv')
    base_rows = metrics[metrics.job_id.isin(selected.id)].copy()
    wanted = []
    for j in selected.to_dict('records'):
        for metric in (['macro_f1', 'balanced_accuracy', 'accuracy'] if j['task_type']=='J1' else ['effective_spearman','mae']):
            wanted.append({'job_id': j['id'], 'metric': metric, 'task': j['task'], 'task_type': j['task_type'],
                           'method': j['method'], 'family': j['family'], 'seed': j['seed'],
                           'direction': j['source']+' -> '+j['target'], 'marker': j['marker']})
    expected = pd.DataFrame(wanted)
    exact_grid(base_rows, expected, list(expected.columns))
    finite(base_rows.score)
    require(base_rows.arm.eq('').all(), 'Unexpected score arm')
    require(set(base_rows.method) == METHODS, 'Methods changed')
    pieces = [base_rows]
    j1_stats = []; j1_updated = []
    for case, dirname in inv.UPDATES.items():
        manifest = read(root / dirname / 'MANIFEST.json')
        checks = read(root / dirname / 'CHECKS.json')
        require(checks['original_files_unchanged'] is True, 'Historical input check failed')
        require(read(root / dirname / 'RUN_COMPLETE.json')['all_passed'] is True, 'Historical update incomplete')
        df = frame(root / dirname / 'SCORES_PER_JOB.csv')
        if case == 'lung_story': df = df[df.arm == 'keep_c02'].copy()
        if case == 'hnscc_story': df = df[df.cohort == 'full'].copy()
        reference = jobs[(jobs.task == case) & (jobs.task_type == 'J1')]
        exact_grid(df, reference, GRID)
        grouped = df.groupby(GRID[1:], sort=True)
        require(grouped.size().eq(10).all(), 'Missing paired method')
        for k in ('supervision_sha256', 'target_ids_sha256'):
            require(df[k].str.fullmatch('[0-9a-f]{64}').all() and grouped[k].nunique().eq(1).all(), 'Method-dependent support: '+k)
        for k, allowed in [('method',METHODS),('family',FAMILIES),('seed',SEEDS)]:
            require(set(df[k]) == allowed, 'Wrong ' + k)
        df['task'] = case; df['task_type'] = 'J1'; df['marker'] = ''
        df['direction'] = df.source + ' -> ' + df.target
        j1_updated.append(df)
        for metric in ('macro_f1','balanced_accuracy','accuracy'):
            m = df.copy(); m['metric'] = metric; m['score'] = finite(m['patient_'+metric]); pieces.append(m)
        j1_stats.append({'task':case,'jobs':len(df),'directions':df.direction.nunique(),
                         'classes':manifest['classes'],'selection':'keep_c02' if case=='lung_story' else 'full',
                         'exploratory':manifest.get('exploratory'),
                         'split_modes':sorted(df.split_mode.unique()) if 'split_mode' in df else 'see frozen plans',
                         'scope_history':manifest.get('scope_selection_history',manifest.get('sampling'))})
    combined_updated = pd.concat(j1_updated, ignore_index=True)
    release_updated = frame(root / inv.J1 / 'UPDATED_J1_JOBS.csv')
    exact_grid(combined_updated, release_updated, ['task'] + GRID)
    for metric in ('macro_f1','balanced_accuracy','accuracy'):
        a = combined_updated.rename(columns={'patient_'+metric:'score'})
        b = release_updated.rename(columns={'patient_'+metric:'score'})
        compare_scores(a,b,['task']+GRID)
    newlung = read(root / inv.J2 / 'MANIFEST.json')
    scores = frame(root / inv.J2 / 'summary/SCORES_PER_JOB.csv')
    newjobs = pd.DataFrame(newlung['jobs'])
    exact_grid(scores.rename(columns={'job_id':'id'}), newjobs, ['id','task','task_type','marker']+GRID)
    require(len(newjobs)==4860 and newjobs.id.is_unique, 'Wrong Lung job grid')
    require(scores.target_cells.eq(28000).all(), 'Wrong Lung target support')
    paired = scores.groupby(['marker','source','target','seed','family'])
    for k in ('supervision_sha256','target_ids_sha256'):
        require(paired[k].nunique().eq(1).all(), 'Lung method-specific support')
    for metric in ('effective_spearman','mae'):
        m = scores.copy();m['metric']=metric;m['score']=finite(m[metric]);pieces.append(m)
    rows = pd.concat(pieces,ignore_index=True)
    all_model = aggregate(rows)
    published = pd.DataFrame(read(root / inv.FINAL / 'primary_means.json'))
    comparison = compare_scores(all_model,published)
    primary = all_model[all_model.task != 'ALL']
    bench = primary.groupby(['task_type','task','metric','method'],as_index=False).score.mean()
    overall = bench.groupby(['task_type','metric','method'],as_index=False).score.mean()
    tables = {'primary_means': comparison,
        'PER_BENCHMARK':compare_scores(bench,frame(root/inv.FINAL/'PER_BENCHMARK.csv'),['task_type','task','metric','method']),
        'OVERALL':compare_scores(overall,frame(root/inv.FINAL/'OVERALL.csv'),['task_type','metric','method'])}
    for kind in ('J1','J2'):
        b = bench[bench.task_type==kind].drop(columns='task_type').copy()
        b['rank'] = b.groupby(['task','metric']).score.transform(lambda s:s.rank(ascending=s.name[1]=='mae',method='min')).astype(int)
        o = b.groupby(['method','metric'],as_index=False).score.mean()
        o['rank'] = o.groupby('metric').score.transform(lambda s:s.rank(ascending=s.name=='mae',method='min')).astype(int)
        tables[kind+'_PER_BENCHMARK']=compare_scores(b,frame(root/inv.FINAL/(kind+'_PER_BENCHMARK.csv')),['task','metric','method'])
        tables[kind+'_OVERALL']=compare_scores(o,frame(root/inv.FINAL/(kind+'_OVERALL.csv')),['method','metric'])
    # Exact declared grids, including feasibility exclusions, not only an expected row count.
    active_j2 = pd.concat([jobs[(jobs.task_type=='J2')&(jobs.task!='lung_story')],newjobs],ignore_index=True)
    require(len(active_j2)==29310 and not active_j2.duplicated(['task','marker']+GRID).any(),'Wrong current J2 grid')
    for key,g in active_j2.groupby(['task','marker','source','target','family','seed']):
        require(set(g.method)==METHODS and len(g)==10,'J2 method missing from a declared job')
        require(g.supervision_sha256.nunique()==1,'Different supervised source support')
    exclusions=read(root/'validation/formal_preflight_v0_5/EXCLUSIONS.json')
    primary_exclusions=[r for r in exclusions if r['job']['task_type']=='J2' and 'primary' in r['job']['analysis_scopes']]
    exclude_short=[{k:r['job'].get(k) for k in ['task','source','target','marker','family','seed']} | {'reason':r['reason']} for r in primary_exclusions]
    grid_keys=['task','marker']+GRID
    full_grid=set()
    for case,g in active_j2.groupby('task'):
        platforms=sorted(set(g.source)|set(g.target))
        for marker,method,family,seed,(src,dst) in itertools.product(sorted(g.marker.unique()),sorted(METHODS),sorted(FAMILIES),sorted(SEEDS),itertools.permutations(platforms,2)):
            full_grid.add((case,marker,method,family,seed,src,dst))
    omitted=set()
    for entry in primary_exclusions:
        j=entry['job']
        if j['task'] not in set(active_j2.task):continue
        require(entry['applies_to_all_eligible_methods'] is True,'Method-specific feasibility exclusion')
        for method in METHODS:omitted.add((j['task'],j['marker'],method,j['family'],j['seed'],j['source'],j['target']))
    actual=set(active_j2[grid_keys].itertuples(index=False,name=None))
    require(full_grid-actual==omitted and not (actual-full_grid) and not (actual&omitted),'Feasibility exclusions do not explain grid')
    receipts=[]
    for path in read(Path(snapshot)/'catalog.json')['j2_representation_receipts']:
        folder = Path(snapshot)/'tree'/path
        start,done = read(folder/'START.json'),read(folder/'COMPLETE.json')
        canonical = done['input_contract']['canonical_markers'][Path(path).parent.name]
        require(start['masked']==[canonical] and start['key']==done['key'],'Incorrect marker mask receipt')
        require(start['pool_sha256']==done['pool_sha256'] and done['labels_used'] is False,'Representation contract differs')
        require(done['method']==Path(path).name,'Method receipt mismatch')
        if (folder/'MASK_BOUNDARY_AUDIT.json').exists():
            boundary=read(folder/'MASK_BOUNDARY_AUDIT.json')
            require(boundary['passed'] is True and boundary['canonical_mask']==[canonical] and boundary['all_platforms_masked'] is True,'Lung mask audit differs')
            require(canonical not in [done['input_contract']['canonical_markers'][m] for m in boundary['feature_names']], 'Target feature present')
        receipts.append({'path':path,'shape':done['shape'],'method':done['method'],'canonical_mask':canonical})
    require(len(receipts)==790,'Wrong J2 representation receipt grid')
    task_j2=[]
    for case,g in active_j2.groupby('task'):
        task_j2.append({'task':case,'jobs':len(g),'markers':sorted(g.marker.unique()),
                        'split_modes':g.split_mode.value_counts().to_dict()})
    bm = jobs[(jobs.task=='bone_marrow')&(jobs.task_type=='J1')]
    count_j1=len(bm)+sum(r['jobs'] for r in j1_stats)
    dirs=int(bm[['source','target']].drop_duplicates().shape[0])+sum(r['directions'] for r in j1_stats)
    require(count_j1==3060 and dirs==34,'Wrong J1 totals')
    require(sum(len(r['markers']) for r in task_j2)==79,'Wrong benchmark-marker count')
    result={'status':'passed','scope':'source/receipt and existing-score reaggregation only; no prediction rescoring or fitting',
        'snapshot':verified,'j1_jobs':count_j1,'j1_directions':dirs,'j1_updates':j1_stats,
        'j2_jobs':len(active_j2),'benchmark_marker_combinations':79,'j2_tasks':task_j2,
        'feasibility_omitted_jobs_current_six_tasks':len(omitted),
        'primary_j2_exclusions':exclude_short,'j2_receipts_checked':len(receipts),
        'representation_values_rehashed':False,'predictions_rescored':False,
        'score_rows_reaggregated':len(rows),'comparisons':tables}
    return result, bench, overall


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--anchor',required=True);p.add_argument('--source-root',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();out=None
    if a.output:out=reserve(a.output,[a.snapshot]+([a.source_root] if a.source_root else []))
    try:
        result,bench,overall=inspect(a.snapshot,a.anchor,a.source_root)
        if out:
            for name,df in [('PER_BENCHMARK.csv',bench),('OVERALL.csv',overall)]:
                with (out/name).open('x') as stream:df.to_csv(stream,index=False)
            write(out/'report.json',result)
        print(json.dumps({k:result[k] for k in ('status','j1_jobs','j2_jobs','score_rows_reaggregated','comparisons')},indent=2))
    except BaseException as error:
        if out:write(out/'FAILED.json',{'error':repr(error)})
        raise


if __name__=='__main__':main()
