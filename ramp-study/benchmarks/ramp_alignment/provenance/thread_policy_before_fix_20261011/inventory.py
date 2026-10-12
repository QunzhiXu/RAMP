"""Freeze the seven historical RAMP contracts; only six require a new fit here."""
import argparse
import ast
from pathlib import Path
import safe_io as io

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
LEGACY = STUDY.parent/'final_xu'
SNAPSHOT = HERE/'frozen_20261011'
REGISTRY_SHA = '4f477c04dc32e98480d2d74ef29927c3e324813b64ca3c832d1a31f4c9f07b22'
CATALOG_SHA = 'cdd905472a93bac8e835e0f3506470a9a84a13971dfcde5fa12f5cc0746ee535'
CANONICAL_SHA = '2501055cf269e383179b602bd99ff39171c8d283bc6332cd4143eed30c4a68c1'
HELPER_SHA = '96845657ec9dbe652073cf7d2e5666c96141168c03604d6ba7a13075d5b4969d'
CASES = ('bone_marrow', 'tnbc', 'melanoma', 'lung_cancer', 'lung_story', 'melanoma_story', 'hnscc_story')
EXECUTABLE = CASES[1:]
AUDITS = ('coupling_audit', 'strength_audit', 'evidence_audit', 'reference_weight_audit', 'profile_audit', 'algorithm_audit')
LUNG = 'story_lung_HLA_ABC/releases/senosain140k_20260921'
INPUT_NAMES = {'raw':'raw.npy', 'fit_metadata':'fit_metadata.csv.gz', 'markers':'markers.txt'}


def literal(path, name):
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Missing literal '+name+' in '+str(path))


def layout(case):
    if case in CASES[:4]:
        base = f'benchmark_results/{case}/ramp/alignment'
        return base+'/run_alignment.py', base+'/data', base+'/data/ramp_manifest.json'
    if case == 'lung_story':
        return LUNG+'/scripts/align.py', LUNG+'/alignment', LUNG+'/inputs/DESIGN.json'
    if case == 'melanoma_story':
        base='figure_melanoma/analysis/alignment'
        return base+'/run_alignment.py', base+'/output', base+'/output/manifest.json'
    if case == 'hnscc_story':
        base='figure_HNSCC/alignment'
        return base+'/02_run_ramp_ver2.py', base+'/output', base+'/output/alignment_manifest.json'
    raise ValueError(case)


def output_contract(case):
    matrices = {'ramp_aligned.npy':['aligned','float32'], 'kmeanslow_global.npy':['baseline','float32']}
    if case in ('tnbc','lung_cancer'):
        matrices.update({'ramp_aligned_float64.npy':['aligned','float64'], 'kmeanslow_global_float64.npy':['baseline','float64']})
    if case == 'lung_story':
        matrices={'RAMP_SHARED9.npy':['aligned','float64'], 'baseline_SHARED9.npy':['baseline','float64']}
    if case == 'hnscc_story':
        matrices={'aligned_ramp_ver2.float32.npy':['aligned','float32'], 'baseline_kmeanslow.float32.npy':['baseline','float32']}
    if case == 'melanoma_story':
        matrices={'ramp_aligned.npy':['aligned','float32'], 'kmeanslow_baseline.npy':['baseline','float32']}
    audits={(a.upper() if case=='lung_story' else a)+'.csv':a for a in AUDITS}
    extras=['REFERENCE_PROFILES.npz','FROZEN_PLATFORM_MAPS.npz'] if case=='lung_story' else []
    return matrices, audits, extras


def collect(source, registry):
    import numpy as np
    import pandas as pd
    if io.record(registry)['sha256'] != REGISTRY_SHA: raise ValueError('Registry trust anchor changed')
    reg = io.read(registry)
    sources, copies, cases = {}, {}, []
    def track(name, copy_as=None):
        row=io.record(source/name);sources[name]=row
        if copy_as: copies[copy_as]={'source':name, **row}
        return row
    for name in ('__init__.py','ramp.py','ramp_ver2.py','run_ramp.py'):
        row=track('algorithm/ramp/'+name, 'vendor/ramp/'+name)
        io.check(STUDY/'vendor/ramp'/name,row)
    if sources['algorithm/ramp/ramp_ver2.py']['sha256'] != CANONICAL_SHA or sources['algorithm/ramp/ramp.py']['sha256'] != HELPER_SHA:
        raise ValueError('Canonical RAMP changed')
    for name in ('__init__.py','ramp.py','ramp_ver2.py'):
        row=track(LUNG+'/vendor/ramp/'+name, 'sources/'+LUNG+'/vendor/ramp/'+name)
        if name!='__init__.py' and row != sources['algorithm/ramp/'+name]: raise ValueError('Lung numerical vendor differs')
        if name=='__init__.py' and (source/LUNG/'vendor/ramp'/name).read_bytes()!=b'':raise ValueError('Lung historical package initializer changed')
    track(LUNG+'/scripts/frozen_maps.py','helpers/frozen_maps.py')
    track(LUNG+'/scripts/runtime.py','sources/'+LUNG+'/scripts/runtime.py')
    for r in reg['cases']:
        case=r['case']; script, result_dir, manifest=layout(case)
        for p in (script,manifest,r['source_manifest']): track(p,'sources/'+p)
        if case.endswith('_story'):
            track(str(Path(r['input_directory']).parent/'scripts/prepare.py'), 'sources/'+str(Path(r['input_directory']).parent/'scripts/prepare.py'))
        for role,row in r['files'].items():
            io.check(source/row['path'],row);sources[row['path']]={k:row[k] for k in ('bytes','sha256')}
        m=io.read(source/manifest)
        family=m['family']
        if case!='lung_story' and literal(source/script,'FAMILY')!=family:raise ValueError('Family disagrees with code')
        if case in CASES[:4] and m['algorithm_source']['sha256']!=CANONICAL_SHA:raise ValueError('Historical source hash differs')
        policy=literal(source/script,'PAIR_FIT_POLICY') if case=='lung_cancer' else {'minimum':4000,'per_unit':100,'maximum':8000}
        matrices,audits,extras=output_contract(case)
        refs={}
        for n in [*matrices,*audits,*extras]:refs[n]={'source':result_dir+'/'+n, **track(result_dir+'/'+n)}
        for n,(attr,dtype) in matrices.items():
            x=np.load(source/result_dir/n,allow_pickle=False)
            if list(x.shape)!=r['expected_shape'] or str(x.dtype)!=dtype or not np.isfinite(x).all():raise ValueError('Historical output contract differs')
        coupling=pd.read_csv(source/result_dir/next(n for n,a in audits.items() if a=='coupling_audit'))
        if set(coupling.family)!={family}:raise ValueError('Coupling family differs')
        for k,v in policy.items():
            if set(coupling['pair_fit_'+k])!={v}:raise ValueError('Pair policy differs')
        aligned=next(n for n,(a,d) in matrices.items() if a=='aligned')
        if case.endswith('_story'):
            p=r['input_directory']+'/ramp.npy';track(p)
            if sources[p]!={k:refs[aligned][k] for k in ('bytes','sha256')}:raise ValueError('Comparison does not use this alignment')
        cases.append({'case':case,'reference_only':case=='bone_marrow','family':family,'pair_fit':policy,
                      'seed':42,'algorithm_version':'2.3.0','shape':r['expected_shape'],'input_dtype':r['observed']['dtype'],
                      'datasets':r['observed']['dataset_order'],'dataset_cells':r['observed']['dataset_cells'],
                      'dataset_units':r['observed']['dataset_units'],'markers':r['observed']['markers'],
                      'fit_columns':r['fit_columns'],'inputs':{INPUT_NAMES[k]:r['files'][k] for k in INPUT_NAMES},
                      'matrices':matrices,'audits':audits,'extras':extras,'references':refs,
                      'source_script':script,'historical_manifest':manifest})
    return {'registry_sha256':REGISTRY_SHA,'source_root':str(source),'cases':cases,'sources':sources,'copies':copies}


def freeze(source, output, registry):
    source=Path(source).resolve(strict=True);registry=Path(registry).resolve(strict=True)
    c=collect(source,registry)
    out=io.reserve(output,(source,STUDY/'vendor',registry,STUDY/'benchmarks/bone_marrow'))
    try:
        io.copy(registry,out/'registry.json',io.record(registry))
        for n,row in c['copies'].items():io.copy(source/row['source'],out/n,row)
        for n,row in c['sources'].items():io.check(source/n,row)
        if io.record(registry)['sha256']!=REGISTRY_SHA:raise ValueError('Registry changed during freeze')
        io.write(out/'catalog.json',c)
        expected={n:{k:r[k] for k in ('bytes','sha256')} for n,r in c['copies'].items()}
        expected['registry.json']=io.record(registry)
        io.payload(out,expected,('catalog.json',))
        io.write(out/'COMPLETE.json',{'status':'complete','catalog_sha256':io.record(out/'catalog.json')['sha256']})
    except BaseException as e:io.failed(out,e);raise
    return io.record(out/'catalog.json')


def catalog():
    if io.record(SNAPSHOT/'catalog.json')['sha256']!=CATALOG_SHA:raise ValueError('Catalog trust anchor differs')
    c=io.read(SNAPSHOT/'catalog.json');done=io.read(SNAPSHOT/'COMPLETE.json')
    if done!={'status':'complete','catalog_sha256':CATALOG_SHA}:raise ValueError('Incomplete snapshot')
    if io.record(SNAPSHOT/'registry.json')['sha256']!=REGISTRY_SHA:raise ValueError('Registry changed')
    expected={n:{k:r[k] for k in ('bytes','sha256')} for n,r in c['copies'].items()}
    expected['registry.json']=io.record(SNAPSHOT/'registry.json')
    io.payload(SNAPSHOT,expected,('catalog.json','COMPLETE.json'))
    return c


def spec(case):
    if case not in CASES:raise ValueError('Unknown case: '+str(case))
    return next(s for s in catalog()['cases'] if s['case']==case)


def verify_source(source):
    c=catalog();source=Path(source).resolve(strict=True)
    for n,row in c['sources'].items():io.check(source/n,row)
    return {'source_files':len(c['sources']),'frozen_copies':len(c['copies']),'cases':len(c['cases'])}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['freeze','verify'])
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--registry',type=Path,default=STUDY/'benchmarks/integration_inputs/registry_20261010.json')
    a=p.parse_args()
    if a.mode=='freeze':
        if a.output is None:p.error('--output required')
        result=freeze(a.source_root,a.output,a.registry)
    else:result=verify_source(a.source_root)
    print(result)


if __name__=='__main__':main()
