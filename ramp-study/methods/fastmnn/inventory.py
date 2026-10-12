"""Freeze seven historical fastMNN embedding branches; never fit or preprocess."""
import argparse
import hashlib
import json
from pathlib import Path

import fastmnn_io as io

HERE = Path(__file__).resolve().parent
REGISTRY = HERE.parents[1]/'benchmarks/integration_inputs/registry_20261010.json'
REGISTRY_SHA = '4f477c04dc32e98480d2d74ef29927c3e324813b64ca3c832d1a31f4c9f07b22'
SNAPSHOT = HERE/'frozen_20261010'
CATALOG_SHA = '04c2cce350fe085533b1bf3bb8e4097e2c5277e7d6a7091bf375293c9844c15d'
LEGACY_SOURCE = Path('/Users/qxu6/Documents/New project/final_xu')
TECHNICAL = ('bone_marrow','tnbc','melanoma','lung_cancer')
STORIES = {'hnscc_story':'figure_HNSCC/Comparison','melanoma_story':'figure_melanoma/Comparison',
           'lung_story':'story_lung_HLA_ABC/Comparison_senosain140k_20260921'}
PARAMETERS = dict(seed=42,rng_kind='Mersenne-Twister',normal_kind='Inversion',sample_kind='Rejection',
    k=20,prop_k=None,cos_norm=True,ndist=3,requested_d=50,
    d_policy='request 50; batchelor realizes min(50, n_markers - 1, n_cells - 1)',deferred=True,
    weights=None,get_variance=True,merge_order=None,auto_merge=False,min_batch_skip=0,subset_row=None,
    correct_all=False,pca_solver='IrlbaParam',neighbor_solver='KmknnParam',parallel_solver='SerialParam')
ADAPTER_FILES = ('fastmnn_repro.R','run_fastmnn.R','bootstrap_library.R','upstream_lock.json','runtime_lock.csv',
    'config.example.json','preprocessing_manifest.example.json','README.md','PREPROCESSING_PROTOCOL.md',
    'PARAMETER_PROVENANCE.md','vendor/LICENSE_NOTICE.md','vendor/batchelor_1.18.1.tar.gz',
    'tests/smoke_test.R','tests/integrity_contract_test.R','tests/cli_contract_test.R')


def registry():
    data=REGISTRY.read_bytes()
    if hashlib.sha256(data).hexdigest()!=REGISTRY_SHA: raise ValueError('Accepted registry changed')
    return json.loads(data),{'bytes':len(data),'sha256':REGISTRY_SHA}


class Builder:
    def __init__(self,source):
        self.source=Path(source).resolve(strict=True);self.files={};self.links=[]

    def rel(self,path):
        path=Path(path)
        if path.is_absolute():
            if path.is_relative_to(self.source): path=path.relative_to(self.source)
            elif path.is_relative_to(LEGACY_SOURCE): path=path.relative_to(LEGACY_SOURCE)
            else: raise ValueError('Unbound historical absolute path: '+str(path))
        if '..' in path.parts: raise ValueError('Unsafe source path')
        return path.as_posix()

    def resolve(self,directory,path):
        p=Path(path)
        if p.is_absolute(): return self.rel(p)
        return (self.source/directory/p).resolve(strict=True).relative_to(self.source).as_posix()

    def add(self,path,copy=False):
        name=self.rel(path)
        if name not in self.files: self.files[name]={'source':name,**io.record(io.within(self.source,name))}
        if copy:
            self.files[name]['copy']=('adapter/'+str(Path(name).relative_to('algorithm/fastmnn'))
                if name.startswith('algorithm/fastmnn/') else 'study_sources/'+name)
        return name

    def json(self,path):return io.read_json(self.source/self.add(path,copy=True))

    def link(self,manifest,fields,target):
        name=self.add(manifest,copy=True);dest=self.add(target)
        value=io.read_json(self.source/name)
        for field in fields:value=value[field]
        if value!=self.files[dest]['sha256']: raise ValueError(f'Historical hash mismatch: {name} {fields} -> {dest}')
        self.links.append({'manifest':name,'fields':fields,'target':dest})

    def paired_fields(self,manifest,value,directory,fields=()):
        """Resolve explicit JSON file/hash pairs, never search prose for a hash."""
        if isinstance(value,list):
            for i,row in enumerate(value):self.paired_fields(manifest,row,directory,(*fields,i))
        elif isinstance(value,dict):
            for key,path in value.items():
                hkey=key[:-5]+'_sha256' if key.endswith('_file') else None
                if key=='path' and 'sha256' in value:hkey='sha256'
                if hkey in value and isinstance(path,str):
                    self.link(manifest,list(fields)+( [hkey]),self.resolve(directory,path))
                if isinstance(path,(dict,list)):self.paired_fields(manifest,path,directory,(*fields,key))


def check_input(root,spec,canonical=True):
    import numpy as np
    root=Path(root);inp=io.table(root/spec['input']);markers=(root/spec['marker_file']).read_text().splitlines()
    identity=['cell_id','dataset','unit_key']
    if list(inp.columns)!=spec['input_columns'] or markers!=spec['markers']:raise ValueError('Marker/input-column order differs')
    if len(inp)!=spec['cells'] or inp.cell_id.duplicated().any() or (inp[identity]=='').any().any():raise ValueError('Cell identity differs')
    if (list(dict.fromkeys(inp.dataset))!=spec['datasets'] or inp.unit_key.nunique()!=spec['units'] or
        (inp.groupby('unit_key').dataset.nunique()!=1).any()):raise ValueError('Dataset/unit identity differs')
    x=inp[markers].to_numpy(float)
    if not np.isfinite(x).all():raise ValueError('Non-finite input')
    if canonical and not inp[identity].equals(io.table(root/spec['canonical']['evaluation_metadata'])[identity]):
        raise ValueError('Canonical cell/dataset/unit order differs')
    if spec['kind']=='story':
        if (not np.array_equal(np.load(root/spec['ids'],allow_pickle=False),inp.cell_id) or
            not io.table(root/spec['metadata']).equals(inp[identity])):raise ValueError('Story IDs/metadata differ')
    return {'cells':len(inp),'markers':len(markers),'units':int(inp.unit_key.nunique()),
        'dataset_counts':inp.groupby('dataset',sort=False).size().to_dict(),
        'zero_l2_cells':int(np.sum(np.sum(x*x,axis=1)==0)),
        'input_ranges':{d:[float(p[markers].min().min()),float(p[markers].max().max())] for d,p in inp.groupby('dataset',sort=False)}}


def identity(root,spec):
    import numpy as np
    import pandas as pd
    counts=check_input(root,spec);root=Path(root);inp=io.table(root/spec['input']);out=io.table(root/spec['output'])
    if list(out.columns)!=['cell_id',*spec['embedding_columns']] or not np.array_equal(inp.cell_id,out.cell_id):raise ValueError('Historical embedding identity differs')
    if not np.isfinite(out.iloc[:,1:].to_numpy(float)).all():raise ValueError('Non-finite embedding')
    meta=io.table(root/spec['output_metadata'])
    if (list(meta.columns)!=['cell_id','batch','sample'] or not np.array_equal(meta.cell_id,inp.cell_id) or
        not np.array_equal(meta.batch,inp.dataset) or not np.array_equal(meta['sample'],inp.unit_key)):raise ValueError('Historical provenance differs')
    if spec['kind']=='story':
        old=np.load(root/spec['values'],allow_pickle=False)
        parsed=pd.read_csv(root/spec['output'],low_memory=False).iloc[:,1:].to_numpy(float)
        if old.dtype!=np.float64 or not np.array_equal(old,parsed):raise ValueError('Historical NPY/CSV conversion differs')
    return {**counts,'dimensions':spec['dimensions'],'output_type':'corrected_latent_embedding','ordered_identity_matches':True}


def build(source):
    reg,reg_record=registry();cases=reg['cases'];cases=list(cases.values()) if isinstance(cases,dict) else cases
    b=Builder(source);branches=[]
    for name in ADAPTER_FILES:b.add('algorithm/fastmnn/'+name,copy=True)
    for name in ('fastmnn_benchmark_runner.py','audit_fastmnn_benchmarks.py','fastmnn_reproducibility_audit.json'):
        b.add('benchmark_results/'+name,copy=True)
    diagnostic='Main Figure/fastmnn_reconstructed_audit_20260914'
    for name in ('README.md','run_audit.py','evaluate_reconstructed.py','official_convertPCsToSCE.R','manifest.json'):
        b.add(diagnostic+'/'+name,copy=True)
    for case in cases:
        name=case['case'];canonical={}
        for role,row in case['files'].items():
            io.check(io.within(b.source,row['path']),row);canonical[role]=b.add(row['path'])
        markers=case['observed']['markers'];dim=min(50,len(markers)-1,case['expected_shape'][0]-1)
        common=dict(case=name,cells=case['expected_shape'][0],units=case['expected_units'],markers=markers,
                    datasets=case['observed']['dataset_order'],canonical=canonical,parameters=PARAMETERS,
                    dimensions=dim,output_type='corrected_latent_embedding',profile='canonical')
        if name in TECHNICAL:
            base=Path('benchmark_results')/name/'fastmnn';pp=base/'preprocessing';al=base/'alignment/data'
            for directory in (base,pp):
                for p in sorted((b.source/directory).iterdir()):
                    if p.is_file() and p.suffix in ('.R','.py','.json','.txt','.md'):b.add(p,copy=True)
            pm_path=pp/'preprocessing_manifest.json';bm_path=pp/'preprocessing_build_manifest.json'
            pm=b.json(pm_path);bm=b.json(bm_path);am=b.json(al/'manifest.json')
            cfg_path=b.rel(am['config_file']['path']);cfg=b.json(cfg_path)
            if bm.get('status')!='complete' or am.get('status')!='complete' or b.json(pp/'COMPLETE.json').get('status')!='complete' or b.json(al/'COMPLETE.json').get('status')!='complete':raise ValueError('Incomplete historical stage')
            for key,target in [('preprocessing_manifest_sha256',pm_path),('preprocessing_build_manifest_sha256',bm_path),('algorithm_config_sha256',cfg_path)]:
                b.link(pp/'COMPLETE.json',[key],target)
            b.link(al/'COMPLETE.json',['manifest_sha256'],al/'manifest.json')
            if cfg['algorithm_parameters']!=PARAMETERS or am['algorithm']['resolved_parameters']!=PARAMETERS:raise ValueError('Noncanonical parameters')
            if (cfg['batch_levels']!=common['datasets'] or am['input']['markers']!=markers or
                am['algorithm']['actual_d']!=dim or am['algorithm']['reconstructed_assay_exported'] is not False):raise ValueError('Wrong embedding contract')
            b.paired_fields(pm_path,pm,pp)
            b.paired_fields(bm_path,bm,pp)
            input_file=pp/'data/combined_analysis_ready.csv.gz';marker=pp/'markers.txt'
            for fields,target in [(['input','sha256'],input_file),(['input','marker_file_sha256'],marker),
                (['input','preprocessing_record_sha256'],pm_path),(['config_file','sha256'],cfg_path)]:b.link(al/'manifest.json',fields,target)
            for key,target in [('input_sha256',input_file),('marker_file_sha256',marker),('preprocessing_record_sha256',pm_path)]:b.link(cfg_path,['input_scale',key],target)
            for key,row in am['implementation'].items():b.link(al/'manifest.json',['implementation',key,'sha256'],row['path'])
            for i,row in enumerate(am['output_files']):
                b.link(al/'manifest.json',['output_files',i,'sha256'],al/row['file'])
                if Path(row['file']).suffix in ('.json','.txt'):b.add(al/row['file'],copy=True)
            spec={**common,'kind':'technical','base':str(base),'input':str(input_file),'marker_file':str(marker),
                'record':str(pm_path),'config':cfg_path,'input_manifest':str(bm_path),'output_manifest':str(al/'manifest.json'),
                'input_columns':['cell_id','dataset','donor','unit_key',*markers],
                'output':str(al/'fastmnn_corrected_embedding.csv.gz'),'output_metadata':str(al/'cell_metadata_label_free.csv.gz'),
                'embedding_columns':[f'fastMNN_{i:02d}' for i in range(1,dim+1)],'preprocessing':pm['datasets']}
        else:
            base=Path(STORIES[name]);im=b.json(base/'input/manifest.json');cfg=b.json(base/'run_config.json')
            rc=b.json(base/'methods/fastmnn/R_COMPLETE.json');done=b.json(base/'methods/fastmnn/COMPLETE.json')
            expected={'cells':common['cells'],'dimensions':dim,'row_identity_verified':True,'output_type':'embedding'}
            if (cfg['datasets']!=common['datasets'] or cfg['markers']!=markers or cfg['seed']!=42 or
                rc['method']!='fastmnn' or rc['parameters']!=PARAMETERS or rc['result']!=expected or
                done['runtime']!=rc or done['cells']!=common['cells'] or done['dimensions']!=dim):raise ValueError('Story contract differs')
            for f in ('run_r.R','run_r_methods.py','prepare.py'):b.add(base/'scripts'/f,copy=True)
            for filename in im['output_sha256']:b.link(base/'input/manifest.json',['output_sha256',filename],base/'input'/filename)
            for path in im['source_sha256']:b.link(base/'input/manifest.json',['source_sha256',path],path)
            b.link(base/'run_config.json',['input_manifest_sha256'],base/'input/manifest.json')
            for key,target in [('input_sha256',base/'input/analysis_ready.csv.gz'),('helper_sha256','algorithm/fastmnn/fastmnn_repro.R'),('output_sha256',base/'methods/fastmnn/corrected_expression.csv.gz')]:b.link(base/'methods/fastmnn/R_COMPLETE.json',[key],target)
            for key,target in [('values_sha256',base/'methods/fastmnn/values.npy'),('cell_ids_sha256',base/'input/cell_ids.npy'),('input_manifest_sha256',base/'input/manifest.json')]:b.link(base/'methods/fastmnn/COMPLETE.json',[key],target)
            for path in sorted((b.source/base/'methods/fastmnn').iterdir()):
                if path.is_file():b.add(path,copy=path.suffix in ('.json','.txt'))
            spec={**common,'kind':'story','base':str(base),'config':str(base/'run_config.json'),
                'input':str(base/'input/analysis_ready.csv.gz'),'marker_file':str(base/'input/markers.txt'),
                'metadata':str(base/'input/fit_metadata.csv.gz'),'ids':str(base/'input/cell_ids.npy'),
                'input_manifest':str(base/'input/manifest.json'),'output_manifest':str(base/'methods/fastmnn/R_COMPLETE.json'),
                'output':str(base/'methods/fastmnn/corrected_expression.csv.gz'),'values':str(base/'methods/fastmnn/values.npy'),
                'output_metadata':str(base/'methods/fastmnn/cell_metadata.csv.gz'),
                'input_columns':['cell_id','dataset','unit_key',*markers],
                'embedding_columns':[f'MNN{i}' for i in range(1,dim+1)],'preprocessing':im['preprocessing']}
        spec['identity_check']=identity(b.source,spec);branches.append(spec)
    for row in b.files.values():io.check(io.within(b.source,row['source']),row)
    if registry()[1]!=reg_record:raise ValueError('Registry changed during collection')
    return {'schema':1,'status':'complete','registry':reg_record,'branches':branches,
        'files':sorted(b.files.values(),key=lambda r:r['source']),'links':b.links,
        'scope':'Frozen analysis-ready inputs and historical corrected embeddings; not provider dependency closure',
        'diagnostic_only_deferred':diagnostic}


def verify_payload(root,catalog):
    if catalog.get('status')!='complete' or catalog['registry']['sha256']!=REGISTRY_SHA:raise ValueError('Invalid catalog')
    io.check(root/'registry.json',catalog['registry'])
    io.payload(root,{r['copy']:r for r in catalog['files'] if 'copy' in r},('registry.json','catalog.json'))


def freeze(source,output):
    source=Path(source).resolve(strict=True);catalog=build(source)
    output=io.reserve(output,(source,REGISTRY,HERE/'inventory.py',HERE/'fastmnn_io.py'))
    try:
        for row in catalog['files']:
            if 'copy' in row:io.copy_new(io.within(source,row['source']),output/row['copy'],row)
        io.copy_new(REGISTRY,output/'registry.json',catalog['registry'])
        for row in catalog['files']:io.check(io.within(source,row['source']),row)
        io.check(REGISTRY,catalog['registry']);io.create_json(output/'catalog.json',catalog)
        verify_payload(output,catalog)
        io.create_json(output/'COMPLETE.json',{'status':'complete','manifest_sha256':io.sha(output/'catalog.json')})
    except BaseException as error:io.failure(output,error);raise
    return output


def verify_local(snapshot=None):
    root=SNAPSHOT if snapshot is None else Path(snapshot)
    if CATALOG_SHA is None or io.sha(root/'catalog.json')!=CATALOG_SHA:raise ValueError('Frozen catalog anchor differs')
    c=io.complete(root,'catalog.json')
    if c.get('status')!='complete' or c['registry']['sha256']!=REGISTRY_SHA:raise ValueError('Invalid catalog status/registry')
    io.check(root/'registry.json',c['registry'])
    io.payload(root,{r['copy']:r for r in c['files'] if 'copy' in r},('registry.json','catalog.json','COMPLETE.json'))
    return c


def verify_source(source,snapshot=None):
    c=verify_local(snapshot)
    for row in c['files']:io.check(io.within(source,row['source']),row)
    for link in c['links']:
        value=io.read_json(io.within(source,link['manifest']))
        for field in link['fields']:value=value[field]
        if value!=io.sha(io.within(source,link['target'])):raise ValueError('Historical JSON relation differs')
    for spec in c['branches']:
        if identity(source,spec)!=spec['identity_check']:raise ValueError('Recorded cohort identity differs')
    return {'files':len(c['files']),'copies':sum('copy' in r for r in c['files']),
            'hash_relations':len(c['links']),'branches':len(c['branches'])}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('freeze','verify'))
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,default=SNAPSHOT)
    a=p.parse_args();print(freeze(a.source_root,a.output) if a.command=='freeze' else verify_source(a.source_root,a.output))


if __name__=='__main__':main()
