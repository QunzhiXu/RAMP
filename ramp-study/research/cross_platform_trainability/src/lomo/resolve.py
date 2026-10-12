"""Locate the artefacts each R method needs, for every (task, method).

Each method family uses its own file naming; the patterns are explicit below
rather than guessed.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import paths

SRC = paths.SRC
STORY = {'lung_story':'story_lung_HLA_ABC','melanoma_story':'figure_melanoma','hnscc_story':'figure_HNSCC'}
BENCH = ['bone_marrow','tnbc','melanoma','lung_cancer']
TASKS = BENCH + list(STORY)
RMETHODS = ['fastmnn','cycombine','cytonorm','harmony']

# per method: (root suffix, input glob, markers, manifest, config candidates)
PATTERNS = {
 'fastmnn':  ('', ['preprocessing/data/combined*'], 'preprocessing/markers.txt',
              'preprocessing/preprocessing_manifest.json',
              # melanoma keeps the runner's own benchmark config at config.json
              # and the algorithm config beside it, so try that name first
              ['alignment/config.json','algorithm_config.json','config.json']),
 'cycombine':('', ['preprocessing/data/combined*','preprocessing/analysis_ready*'],
              'preprocessing/markers.txt','preprocessing/preprocessing_manifest.json',
              ['alignment/config.json','config.json']),
 'cytonorm': ('', ['preprocessing/analysis_ready*'], 'preprocessing/markers.txt',
              'preprocessing/preprocessing.json',
              ['alignment/executed_config.json','config.json','manifest.json']),
 'harmony':  ('source_scale', ['preprocessing/data/*combined*','preprocessing/data/*analysis_ready*'],
              'preprocessing/markers.txt','preprocessing/manifest.json',
              ['alignment/config.json']),
}


def find(task, method):
    suffix, ing, mk, man, cfgs = PATTERNS[method]
    if task in BENCH:
        r = SRC/'benchmark_results'/task/method
        if suffix: r = r/suffix
    else:
        r = SRC/STORY[task]/'Comparison'/'methods'/method
    out = {'root': r, 'exists': r.exists()}
    if not r.exists(): return out
    if task in BENCH:
        cand = [p for g in ing for p in r.glob(g)]
        out['input'] = cand[0] if cand else None
        out['markers'] = (r/mk) if (r/mk).exists() else None
        out['manifest'] = (r/man) if (r/man).exists() else None
        want = {'fastmnn': 'final_xu.fastmnn_repro.v1',
                'cycombine': 'final_xu.cycombine_repro.v1',
                'cytonorm': 'final_xu.cytonorm.v1',
                'harmony': 'final_xu.harmony_direct.v1'}[method]
        out['config'] = next(
            (r/c for c in cfgs if (r/c).exists()
             and json.loads((r/c).read_text()).get('schema_version') == want), None)
    else:
        shared = SRC/STORY[task]/'Comparison'/'input'
        out['input'] = shared/'analysis_ready.csv.gz' if (shared/'analysis_ready.csv.gz').exists() else None
        out['markers'] = shared/'markers.txt' if (shared/'markers.txt').exists() else None
        out['manifest'] = shared/'manifest.json' if (shared/'manifest.json').exists() else None
        rc = SRC/STORY[task]/'Comparison'/'run_config.json'
        out['config'] = rc if rc.exists() else None
    return out


if __name__ == '__main__':
    import pandas as pd
    rows=[]
    for t in TASKS:
        for m in RMETHODS:
            f=find(t,m)
            rows.append(dict(task=t,method=m,
                input=f.get('input') is not None, markers=f.get('markers') is not None,
                manifest=f.get('manifest') is not None, config=f.get('config') is not None))
    df=pd.DataFrame(rows); df['ok']=df[['input','markers','manifest','config']].all(1)
    print(df.to_string(index=False)); print(f'\nresolved: {df.ok.sum()}/{len(df)}')
    for _,r in df[~df.ok].iterrows():
        print('  MISS', r.task, r.method, [c for c in ['input','markers','manifest','config'] if not r[c]])
