"""Descriptive class-support audit of frozen J1 predictions; never refit or select jobs.

All writes stay beside this script. Rules use class counts, not method scores.
Selected-class scores average existing per-class recall/F1, preserving the entire
confusion matrix and all false positives/negatives. They are not scores from a
new classifier trained or tested on a filtered population.
"""
from pathlib import Path
import collections
import csv
import gzip
import hashlib
import json
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TASKS = ['bone_marrow', 'tnbc', 'melanoma', 'lung_cancer', 'lung_story',
         'melanoma_story', 'hnscc_story']
METHODS = ['ramp', 'unaligned', 'harmony_direct', 'cycombine', 'cytonorm',
           'fastmnn', 'maxfuse', 'clipped_zscore', 'minmax', 'percentile']
FAMILIES = ['linear', 'knn', 'mlp']
SEEDS = [20260919, 20260920, 20260921]
THRESHOLDS = [5, 10, 20]
# A simple hierarchy sensitivity analysis; all unlisted labels stay unchanged.
# Ambiguous NK/gdT is NOT relabeled as pure NK or T.
MERGES = {
    'bone_marrow': {'T lineage': ['CD4 T', 'CD8 T']},
    'tnbc': {'T lineage': ['CD4 T', 'CD8 T']},
    'melanoma': {},
    'lung_cancer': {'T lineage': ['CD4 T', 'CD8 T']},
    'lung_story': {},
    'melanoma_story': {'T lineage': ['CD8 T', 'Other T / CD8-']},
    'hnscc_story': {'T lineage': ['CD4 T', 'CD8 T'], 'B lineage': ['B', 'Plasma']},
}


def dump(name, obj):
    (HERE/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')


def write_csv(name, rows):
    assert rows, name
    with (HERE/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def primary(config):
    jobs = [j for j in config['jobs'] if j['task_type']=='J1'
            and 'primary' in j['analysis_scopes']]
    assert all(j['fit_budget']==500 and j['width']==32 and not j.get('labelled_patient_count') for j in jobs)
    assert {j['family'] for j in jobs} == set(FAMILIES)
    assert {j['seed'] for j in jobs} == set(SEEDS)
    return jobs


def metrics(cm):
    n = len(cm)
    recall=[]; f1=[]
    for i in range(n):
        support=sum(cm[i]); pred=sum(row[i] for row in cm); tp=cm[i][i]
        recall.append(tp/support if support else 0.)
        f1.append(2*tp/(support+pred) if support+pred else 0.)
    return {'balanced_accuracy': recall, 'macro_f1': f1}


def grouped_cm(cm, classes, mapping):
    groups=sorted(set(mapping.values())); loc={g:i for i,g in enumerate(groups)}
    out=[[0. for _ in groups] for _ in groups]
    for i,a in enumerate(classes):
        for j,b in enumerate(classes): out[loc[mapping[a]]][loc[mapping[b]]]+=cm[i][j]
    assert abs(sum(map(sum,cm))-sum(map(sum,out)))<1e-12
    return groups,out


def class_sets(task, classes, counts, fits, datasets):
    def supported(members, threshold):
        return (min(sum(counts[d][c] for c in members) for d in datasets)>=100
                and min(sum(cnt[c] for c in members) for cnt in fits.values())>=threshold)
    sets={'all_shared': list(classes)}
    for k in THRESHOLDS:
        sets[f'support_ge{k}']=[c for c in classes if supported([c],k)]
    mapping={c:c for c in classes}
    for group, members in MERGES[task].items():
        assert set(members)<=set(classes)
        for c in members: mapping[c]=group
    groups=sorted(set(mapping.values()))
    sets['coarse_all']=groups
    sets['coarse_support_ge10']=[g for g in groups if supported([c for c in classes if mapping[c]==g],10)]
    assert all(len(v)>=2 for v in sets.values()), (task,sets)
    return sets,mapping


def aggregate(rows, keys):
    groups=collections.defaultdict(list)
    for row in rows: groups[tuple(row[k] for k in keys)].append(row['score'])
    return [dict(zip(keys,key), score=statistics.mean(v)) for key,v in sorted(groups.items())]


def main():
    # Written before any comparison-method result is opened. These are exploratory
    # rules defined after seeing original overall results, NOT preregistration.
    dump('RULES.json', dict(exploratory=True, training_budget=500, minimum_full_pool_per_platform=100,
         minimum_fit_per_source_and_seed=THRESHOLDS, mergers=MERGES,
         selection_uses_method_performance=False, refit=False, original_outputs_unchanged=True,
         selected_class_scoring='Average selected per-class metrics from full original confusion matrix; no rows/columns removed.',
         aggregation='seeds, directions, model families, benchmarks each equally weighted at their respective levels'))
    counts_out=[]; fit_out=[]; support_out=[]; jobs_out=[]; class_out=[]; sets_out={}
    config_hashes={}; primary_hash=hashlib.sha256(); jobs_read=0; baseline_error=0.
    for task in TASKS:
        inp=ROOT/'data/real_inputs_v0_3_r2'/task
        receipt=json.loads((inp/'manifest.json').read_text()); classes=receipt['classes']
        with gzip.open(inp/'metadata.csv.gz','rt') as f: md={r['cell_id']:r for r in csv.DictReader(f)}
        datasets=sorted({r['dataset'] for r in md.values()})
        counts={d:collections.Counter() for d in datasets}; patients=collections.defaultdict(set)
        for row in md.values():
            counts[row['dataset']][row['celltype']]+=1
            patients[row['dataset'],row['celltype']].add(row['patient_id'])
        for d in datasets:
            eligible=sum(counts[d][c] for c in classes)
            for cl in classes:
                counts_out.append(dict(task=task,dataset=d,celltype=cl,cells=counts[d][cl],
                    fraction_of_eligible=counts[d][cl]/eligible,patients_or_units=len(patients[d,cl])))
        rc=json.loads((ROOT/f'config/formal_v0_5/{task}__ramp.json').read_text())
        rjobs=primary(rc); reference={}; fits={}; fit_patients={}; supervision_by_source={}
        for j in rjobs:
            key=(j['source'],j['target'],j['seed'],j['family']);reference[key]=j['supervision_sha256']
            skey=j['source'],j['seed']
            if skey in supervision_by_source:
                assert supervision_by_source[skey]==j['supervision_sha256']
            supervision_by_source[skey]=j['supervision_sha256']
            if skey in fits: continue
            ledger=json.loads((Path(rc['output'])/'jobs'/j['id']/'ledger.json').read_text())
            cnt=collections.Counter(md[x]['celltype'] for x in ledger['fit']);fits[skey]=cnt
            pats=collections.defaultdict(set)
            for x in ledger['fit']:pats[md[x]['celltype']].add(md[x]['patient_id'])
            fit_patients[skey]=pats
            for cl in classes:
                fit_out.append(dict(task=task,source=skey[0],seed=skey[1],celltype=cl,
                    fit_cells=cnt[cl],fit_patients_or_units=len(pats[cl])))
        sets,mapping=class_sets(task,classes,counts,fits,datasets);sets_out[task]=sets
        for cl in classes:
            support_out.append(dict(task=task,celltype=cl,min_pool_cells=min(counts[d][cl] for d in datasets),
                min_pool_fraction=min(counts[d][cl]/sum(counts[d][c] for c in classes) for d in datasets),
                min_fit_cells=min(c[cl] for c in fits.values()),
                min_fit_patients_or_units=min(len(p[cl]) for p in fit_patients.values()),
                included_support_ge10=cl in sets['support_ge10']))
        for method in METHODS:
            cp=ROOT/f'config/formal_v0_5/{task}__{method}.json';config_hashes[str(cp)]=sha(cp)
            cfg=json.loads(cp.read_text());jobs=primary(cfg)
            assert len(jobs)==len(rjobs)
            for j in jobs:
                key=j['source'],j['target'],j['seed'],j['family']
                assert j['supervision_sha256']==reference[key], (task,method,key)
                path=Path(cfg['output'])/'jobs'/j['id']/'RESULT.json'
                raw=path.read_bytes();primary_hash.update(str(path).encode());primary_hash.update(raw)
                result=json.loads(raw);s=result['score'];cm=s['confusion_matrix']
                assert s['classes']==classes
                met=metrics(cm);gclasses,gcm=grouped_cm(cm,classes,mapping);gmet=metrics(gcm)
                base=dict(task=task,method=method,family=j['family'],source=j['source'],target=j['target'],seed=j['seed'])
                for metric,values in met.items():
                    baseline_error=max(baseline_error,abs(statistics.mean(values)-s[metric]))
                    for cl,value in zip(classes,values):class_out.append(dict(base,celltype=cl,metric=metric,score=value))
                for rule,chosen in sets.items():
                    use_classes,use_metrics=(gclasses,gmet) if rule.startswith('coarse_') else (classes,met)
                    for metric,values in use_metrics.items():
                        score=statistics.mean(values[use_classes.index(cl)] for cl in chosen)
                        jobs_out.append(dict(base,rule=rule,metric=metric,classes=len(chosen),score=score))
                jobs_read+=1
        print(task,'done', 'supported:',sets['support_ge10'],flush=True)
    assert baseline_error<1e-12
    # Keep each hierarchical weight explicit, even though all current jobs are balanced.
    family=aggregate(jobs_out,['task','method','family','rule','metric','source','target'])
    family=aggregate(family,['task','method','family','rule','metric'])
    task_scores=aggregate(family,['task','method','rule','metric'])
    overall=aggregate(task_scores,['method','rule','metric'])
    class_family=aggregate(class_out,['task','method','family','source','target','celltype','metric'])
    class_family=aggregate(class_family,['task','method','family','celltype','metric'])
    class_scores=aggregate(class_family,['task','method','celltype','metric'])
    with (ROOT/'Figure/Panel B/per_benchmark_scores.csv').open() as f:
        old={(r['task'],r['method'],r['metric']):float(r['score']) for r in csv.DictReader(f)}
    error=max(abs(r['score']-old[r['task'],r['method'],r['metric']]) for r in task_scores if r['rule']=='all_shared')
    assert error<1e-12,error
    for name,rows in [('CLASS_COUNTS.csv',counts_out),('FIT_COUNTS.csv',fit_out),('CLASS_SUPPORT.csv',support_out),
                      ('PER_JOB_SCORES.csv',jobs_out),('PER_CLASS_SCORES.csv',class_scores),
                      ('PER_BENCHMARK_SCORES.csv',task_scores),('OVERALL_SCORES.csv',overall)]:write_csv(name,rows)
    dump('CLASS_SETS.json',sets_out)
    dump('CHECKS.json',dict(primary_jobs_read=jobs_read,all_methods_identical_supervision=True,
        maximum_original_job_score_error=baseline_error,maximum_panel_b_reproduction_error=error,
        result_contents_sha256=primary_hash.hexdigest(),config_sha256=config_hashes,
        no_refits=True,formal_outputs_modified=False))
    for rule in sets_out[TASKS[0]]:
        print('\nRULE',rule)
        for metric in ['balanced_accuracy','macro_f1']:
            rr=sorted([r for r in overall if r['rule']==rule and r['metric']==metric],key=lambda r:-r['score'])
            print(metric,[(r['method'],round(r['score'],4)) for r in rr])


if __name__=='__main__':main()
