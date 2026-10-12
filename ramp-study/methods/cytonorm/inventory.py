"""Freeze historical CytoNorm correction sources. No fitting or preprocessing."""
import argparse
import hashlib
import json
from pathlib import Path

import cytonorm_io as io

HERE=Path(__file__).resolve().parent
REGISTRY=HERE.parents[1]/'benchmarks/integration_inputs/registry_20261010.json'
REGISTRY_SHA='4f477c04dc32e98480d2d74ef29927c3e324813b64ca3c832d1a31f4c9f07b22'
SNAPSHOT=HERE/'frozen_20261011'
CATALOG_SHA='532c2950133dce0d6435f7debbaed059a5a7a806bd5be396945463458814dc12'
LEGACY_SOURCE=Path('/Users/qxu6/Documents/New project/final_xu')
TECHNICAL=('bone_marrow','tnbc','melanoma','lung_cancer')
STORIES={'hnscc_story':'figure_HNSCC/Comparison','melanoma_story':'figure_melanoma/Comparison',
         'lung_story':'story_lung_HLA_ABC/Comparison_senosain140k_20260921'}
PARAMETERS={'schema_version':'final_xu.cytonorm.v1',
 'method_label':'CytoNorm 2.0.6 (batch aggregates; benchmark-only preprocessing)',
 'seed':42,'aggregate_cells_per_unit':2000,
 'flowsom':{'nCells':1000000,'xdim':15,'ydim':15,'nClus':10,'rlen':10,'scale':False},
 'quantile':{'nQ':99,'goal':'mean','limit':None}}
GUARD_OLD='if (length(batches) != 3L) stop("Expected exactly three dataset/platform batches")'
GUARD_STORY='if (!(length(batches) %in% c(2L, 3L))) stop("Expected two or three dataset/platform batches")'


def registry():
    data=REGISTRY.read_bytes()
    if hashlib.sha256(data).hexdigest()!=REGISTRY_SHA:raise ValueError('Accepted registry changed')
    return json.loads(data),{'bytes':len(data),'sha256':REGISTRY_SHA}


class Builder:
    def __init__(self,source):
        self.source=Path(source).resolve(strict=True);self.files={};self.links=[]

    def rel(self,path):
        path=Path(path)
        if path.is_absolute():
            if path.is_relative_to(self.source):path=path.relative_to(self.source)
            elif path.is_relative_to(LEGACY_SOURCE):path=path.relative_to(LEGACY_SOURCE)
            else:raise ValueError('Unbound historical path: '+str(path))
        if '..' in path.parts:raise ValueError('Unsafe path')
        return path.as_posix()

    def add(self,path,copy=False):
        name=self.rel(path)
        if name not in self.files:self.files[name]={'source':name,**io.record(io.within(self.source,name))}
        if copy:
            if name=='algorithm/CytoNorm/.Rlib/runtime_lock.csv':dest='environment/runtime_lock.csv'
            elif name.startswith('algorithm/CytoNorm/'):dest='adapter/'+str(Path(name).relative_to('algorithm/CytoNorm'))
            else:dest='study_sources/'+name
            self.files[name]['copy']=dest
        return name

    def json(self,path):return io.read_json(self.source/self.add(path,True))

    def link(self,manifest,fields,target):
        name=self.add(manifest,True);dest=self.add(target)
        value=io.read_json(self.source/name)
        for f in fields:value=value[f]
        if value!=self.files[dest]['sha256']:raise ValueError(f'Broken historical hash: {name} {fields} -> {dest}')
        self.links.append({'manifest':name,'fields':fields,'target':dest})


def check_input(root,spec,canonical=True):
    import numpy as np
    root=Path(root);df=io.table(root/spec['input']);markers=(root/spec['marker_file']).read_text().splitlines()
    ident=['cell_id','dataset','unit_key']
    if list(df)!=ident+spec['markers'] or markers!=spec['markers']:raise ValueError('Input column/marker order differs')
    if len(df)!=spec['cells'] or df.cell_id.duplicated().any() or (df[ident]=='').any().any():raise ValueError('Invalid cell identity')
    if list(dict.fromkeys(df.dataset))!=spec['datasets'] or df.unit_key.nunique()!=spec['units'] or (df.groupby('unit_key').dataset.nunique()!=1).any():raise ValueError('Dataset/unit identity differs')
    x=df[markers].to_numpy(float)
    if not np.isfinite(x).all():raise ValueError('Nonfinite input')
    if canonical and not df[ident].equals(io.table(root/spec['canonical']['evaluation_metadata'])[ident]):raise ValueError('Canonical row identity differs')
    if spec['kind']=='story' and (not np.array_equal(np.load(root/spec['ids'],allow_pickle=False),df.cell_id) or not io.table(root/spec['metadata']).equals(df[ident])):raise ValueError('Story identity differs')
    return {'cells':len(df),'markers':len(markers),'units':int(df.unit_key.nunique()),
        'dataset_counts':df.groupby('dataset',sort=False).size().to_dict(),
        'units_by_dataset':df.groupby('dataset',sort=False).unit_key.nunique().to_dict(),
        'zero_l2_cells':int(np.sum(np.sum(x*x,axis=1)==0)),
        'input_ranges':{d:[float(p[markers].min().min()),float(p[markers].max().max())] for d,p in df.groupby('dataset',sort=False)}}


def identity(root,spec):
    import numpy as np
    import pandas as pd
    root=Path(root);counts=check_input(root,spec);inp=io.table(root/spec['input']);out=io.table(root/spec['output'])
    if list(out)!=['cell_id',*spec['markers']] or not np.array_equal(inp.cell_id,out.cell_id) or not np.isfinite(out.iloc[:,1:].to_numpy(float)).all():raise ValueError('Historical corrected output differs')
    ident=['cell_id','dataset','unit_key'];base=root/spec['result_dir']
    if not io.table(base/'row_metadata.csv.gz').equals(inp[ident]):raise ValueError('Historical row metadata differs')
    clusters=io.table(base/'cell_clusters.csv.gz')
    if not clusters[ident].equals(inp[ident]) or not clusters.cluster.between(1,10).all():raise ValueError('Historical cluster identity differs')
    agg=io.table(base/'aggregate_cells.csv.gz')
    if list(agg)!=ident or agg.cell_id.duplicated().any() or not agg.equals(inp.set_index('cell_id').loc[agg.cell_id].reset_index()[ident]):raise ValueError('Aggregate control identity differs')
    want=inp.groupby('unit_key').size().clip(upper=2000)
    if not agg.groupby('unit_key').size().equals(want):raise ValueError('Per-unit aggregate cap differs')
    record=spec['integrity']
    for k,v in {'cells':len(inp),'markers':len(spec['markers']),'units':spec['units'],'batches':spec['datasets'],
                'aggregate_cells':len(agg),'seed':42,'spline_replay_max_absolute_error':0,
                'row_identity_verified':True,'source_labels_used':False,'nonfinite_outputs':0}.items():
        if record[k]!=v:raise ValueError('Historical integrity field differs: '+k)
    if spec['kind']=='story':
        values=np.load(root/spec['values'],allow_pickle=False)
        parsed=pd.read_csv(root/spec['output'],low_memory=False).iloc[:,1:].to_numpy(float)
        if values.dtype!=np.float64 or not np.array_equal(values,parsed):raise ValueError('Story NPY conversion differs')
    return {**counts,'ordered_identity_matches':True,'aggregate_cells':len(agg),'output_type':'corrected_marker_values'}


def build(source):
    reg,reg_record=registry();cases=reg['cases'];cases=list(cases.values()) if isinstance(cases,dict) else cases
    b=Builder(source);branches=[];alg=Path('algorithm/CytoNorm')
    for p in sorted((b.source/alg).rglob('*')):
        if p.is_file() and not any(part.startswith('.') or part=='__pycache__' for part in p.relative_to(b.source/alg).parts):b.add(p,True)
    runtime=b.add(alg/'.Rlib/runtime_lock.csv',True);lock=b.json(alg/'upstream_lock.json');design=b.json(alg/'config.json')
    if {k:v for k,v in design.items() if k!='cases'}!=PARAMETERS:raise ValueError('Algorithm design differs')
    for i,row in enumerate(lock['archives']):b.link(alg/'upstream_lock.json',['archives',i,'sha256'],alg/'vendor'/row['file'])
    b.add('benchmark_results/cytonorm_benchmark_runner.py',True)
    helper=(b.source/alg/'cytonorm_repro.R').read_bytes()
    for case in cases:
        name=case['case'];canonical={}
        for role,row in case['files'].items():io.check(io.within(b.source,row['path']),row);canonical[role]=b.add(row['path'])
        common={'case':name,'cells':case['expected_shape'][0],'units':case['expected_units'],
            'markers':case['observed']['markers'],'datasets':case['observed']['dataset_order'],
            'canonical':canonical,'parameters':PARAMETERS,'runtime_lock':runtime,'profile':'canonical'}
        if name in TECHNICAL:
            base=Path('benchmark_results')/name/'cytonorm';pp=base/'preprocessing';al=base/'alignment/data'
            pub=b.json(base/'manifest.json');done=b.json(base/'COMPLETE.json');pm=b.json(pp/'preprocessing.json')
            cfg=b.json(base/'alignment/executed_config.json');rdone=b.json(al/'COMPLETE.json');integrity=b.json(al/'integrity.json')
            if done['status']!='COMPLETE' or rdone['status']!='COMPLETE' or pm['case']!=name or pm['labels_read'] is not False:raise ValueError('Incomplete historical stage')
            if pm['design']!=design or pub['integrity']!=integrity or {k:cfg[k] for k in PARAMETERS}!=PARAMETERS:raise ValueError('Historical design differs')
            for f,h in done['outputs'].items():
                if f.startswith(('preprocessing/','alignment/')) or f in ('manifest.json','README.md'):
                    b.link(base/'COMPLETE.json',['outputs',f],base/f)
                    if Path(f).suffix in ('.json','.md','.txt'):b.add(base/f,True)
            for f in rdone['outputs']:b.link(al/'COMPLETE.json',['outputs',f],al/f)
            for f in pm['outputs']:b.link(pp/'preprocessing.json',['outputs',f],pp/f)
            for i,row in enumerate(pm['sources']):b.link(pp/'preprocessing.json',['sources',i,'sha256'],row['path'])
            b.link(pp/'preprocessing.json',['adapter_sha256'],alg/'benchmark.py')
            for i,row in enumerate(pub['source_inputs']):b.link(base/'manifest.json',['source_inputs',i,'sha256'],row['path'])
            for i,row in enumerate(pub['source_completion_files']):b.link(base/'manifest.json',['source_completion_files',i,'sha256'],row['path'])
            b.link(base/'manifest.json',['publisher_sha256'],'benchmark_results/cytonorm_benchmark_runner.py')
            for key,target in [('input_sha256',pp/'analysis_ready.csv.gz'),('markers_sha256',pp/'markers.txt'),
                ('preprocessing_sha256',pp/'preprocessing.json'),('upstream_lock_sha256',alg/'upstream_lock.json'),('runtime_lock_sha256',runtime)]:
                b.link(base/'alignment/executed_config.json',[key],target)
            # Bind the historical algorithm_test path as well as the published byte copy.
            b.link(base/'alignment/executed_config.json',['preprocessing_sha256'],cfg['preprocessing_file'])
            for key in cfg['code_hashes']:b.link(base/'alignment/executed_config.json',['code_hashes',key],alg/key)
            spec={**common,'kind':'technical','base':str(base),'result_dir':str(al),'integrity':integrity,
                'input':str(pp/'analysis_ready.csv.gz'),'marker_file':str(pp/'markers.txt'),
                'config':str(base/'alignment/executed_config.json'),'input_manifest':str(pp/'preprocessing.json'),
                'output':str(al/'corrected_expression.csv.gz'),'preprocessing':design['cases'][name]['transforms']}
        else:
            base=Path(STORIES[name]);al=base/'methods/cytonorm';im=b.json(base/'input/manifest.json');cfg=b.json(base/'run_config.json')
            rc=b.json(al/'R_COMPLETE.json');done=b.json(al/'COMPLETE.json')
            if cfg['markers']!=common['markers'] or cfg['datasets']!=common['datasets'] or cfg['seed']!=42:raise ValueError('Story config differs')
            if rc['method']!='cytonorm' or rc['parameters']!=PARAMETERS or done['runtime']!=rc or done['cells']!=common['cells'] or done['dimensions']!=len(common['markers']):raise ValueError('Story completion differs')
            for f in ('prepare.py','run_r.R','run_r_methods.py','cytonorm_repro.R'):b.add(base/'scripts'/f,True)
            story=(b.source/base/'scripts/cytonorm_repro.R').read_bytes()
            if helper.count(GUARD_OLD.encode())!=1 or story!=helper.replace(GUARD_OLD.encode(),GUARD_STORY.encode()):raise ValueError('Unexpected story helper change')
            for f in im['output_sha256']:b.link(base/'input/manifest.json',['output_sha256',f],base/'input'/f)
            for path in im['source_sha256']:b.link(base/'input/manifest.json',['source_sha256',path],path)
            b.link(base/'run_config.json',['input_manifest_sha256'],base/'input/manifest.json')
            for key,target in [('input_sha256',base/'input/analysis_ready.csv.gz'),('helper_sha256',base/'scripts/cytonorm_repro.R'),('output_sha256',al/'corrected_expression.csv.gz')]:b.link(al/'R_COMPLETE.json',[key],target)
            for key,target in [('values_sha256',al/'values.npy'),('cell_ids_sha256',base/'input/cell_ids.npy'),('input_manifest_sha256',base/'input/manifest.json')]:b.link(al/'COMPLETE.json',[key],target)
            for p in sorted((b.source/al).iterdir()):
                if p.is_file():b.add(p,copy=p.suffix in ('.json','.txt'))
            spec={**common,'kind':'story','base':str(base),'result_dir':str(al),'integrity':rc['result'],
                'input':str(base/'input/analysis_ready.csv.gz'),'marker_file':str(base/'input/markers.txt'),
                'metadata':str(base/'input/fit_metadata.csv.gz'),'ids':str(base/'input/cell_ids.npy'),
                'config':str(base/'run_config.json'),'input_manifest':str(base/'input/manifest.json'),
                'output':str(al/'corrected_expression.csv.gz'),'values':str(al/'values.npy'),'preprocessing':im['preprocessing']}
        spec['identity_check']=identity(b.source,spec);branches.append(spec)
    for row in b.files.values():io.check(io.within(b.source,row['source']),row)
    if registry()[1]!=reg_record:raise ValueError('Registry changed during collection')
    return {'schema':1,'status':'complete','registry':reg_record,'branches':branches,
            'files':sorted(b.files.values(),key=lambda r:r['source']),'links':b.links,
            'scope':'Frozen correction-stage provenance; not a provider or evaluation dependency closure'}


def verify_payload(root,c):
    if c.get('status')!='complete' or c['registry']['sha256']!=REGISTRY_SHA:raise ValueError('Invalid catalog')
    io.check(root/'registry.json',c['registry'])
    io.payload(root,{r['copy']:r for r in c['files'] if 'copy' in r},('registry.json','catalog.json'))


def freeze(source,output):
    source=Path(source).resolve(strict=True);codes={n:io.record(HERE/n) for n in ('inventory.py','cytonorm_io.py')};c=build(source)
    out=io.reserve(output,(source,REGISTRY,*(HERE/n for n in codes)))
    try:
        for r in c['files']:
            if 'copy' in r:io.copy_new(io.within(source,r['source']),out/r['copy'],r)
        io.copy_new(REGISTRY,out/'registry.json',c['registry'])
        for r in c['files']:io.check(io.within(source,r['source']),r)
        for n,r in codes.items():io.check(HERE/n,r)
        io.check(REGISTRY,c['registry']);io.create_json(out/'catalog.json',c);verify_payload(out,c)
        io.create_json(out/'COMPLETE.json',{'status':'complete','manifest_sha256':io.sha(out/'catalog.json')})
    except BaseException as e:io.failure(out,e);raise
    return out


def verify_local(snapshot=None):
    root=SNAPSHOT if snapshot is None else Path(snapshot)
    if CATALOG_SHA is None or io.sha(root/'catalog.json')!=CATALOG_SHA:raise ValueError('Frozen catalog anchor differs')
    c=io.complete(root,'catalog.json')
    if c.get('status')!='complete' or c['registry']['sha256']!=REGISTRY_SHA:raise ValueError('Invalid catalog')
    io.check(root/'registry.json',c['registry'])
    io.payload(root,{r['copy']:r for r in c['files'] if 'copy' in r},('registry.json','catalog.json','COMPLETE.json'))
    return c


def verify_source(source,snapshot=None):
    c=verify_local(snapshot)
    for r in c['files']:io.check(io.within(source,r['source']),r)
    for r in c['links']:
        value=io.read_json(io.within(source,r['manifest']))
        for f in r['fields']:value=value[f]
        if value!=io.sha(io.within(source,r['target'])):raise ValueError('Historical relation differs')
    for s in c['branches']:
        if identity(source,s)!=s['identity_check']:raise ValueError('Cohort identity differs')
    return {'files':len(c['files']),'copies':sum('copy' in r for r in c['files']),'hash_relations':len(c['links']),'branches':len(c['branches'])}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('freeze','verify'))
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,default=SNAPSHOT)
    a=p.parse_args();print(freeze(a.source_root,a.output) if a.command=='freeze' else verify_source(a.source_root,a.output))


if __name__=='__main__':main()
