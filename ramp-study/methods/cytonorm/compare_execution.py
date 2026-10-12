"""Strict read-only historical comparison; never imported by fitting."""
import copy
import math
from pathlib import Path
import re
import subprocess

import cytonorm_io as io
import execution as ex
import inventory as inv
import study


def sessions(old,new):
    normalized=[];observed=[]
    for text in (old,new):
        lines=text.splitlines();indices=[i for i,line in enumerate(lines) if line.startswith('Running under: ')]
        if len(indices)!=1 or not lines[indices[0]].startswith('Running under: macOS '):raise ValueError('Unexpected OS line')
        observed.append(lines[indices[0]]);lines[indices[0]]='<recorded macOS release>';normalized.append(lines)
    if normalized[0]!=normalized[1]:raise ValueError('Session differs beyond macOS release')
    return {'historical':observed[0],'current':observed[1]}


def elapsed(old,new,nested=False):
    copies=[copy.deepcopy(old),copy.deepcopy(new)];times=[]
    for value in copies:
        target=value['runtime'] if nested else value;time=target['elapsed_seconds']
        if isinstance(time,bool) or not isinstance(time,(int,float)) or not math.isfinite(time) or time<0:raise ValueError('Invalid elapsed time')
        times.append(time);target['elapsed_seconds']='<recorded elapsed seconds>'
    if copies[0]!=copies[1]:raise ValueError('Completion differs beyond elapsed time')
    return {'historical_seconds':times[0],'current_seconds':times[1]}


def warnings_relocation(old,new,historical_prefix,current_prefix):
    normalized=[];observed=[]
    for text,prefix in ((old,historical_prefix),(new,current_prefix)):
        pattern=re.compile(r"^cannot remove file '"+re.escape(str(prefix))+
            r"\.staging-([0-9]+)/work/normalize/(Norm_batch_0[123]_fsom(?:[1-9]|10)\.fcs)', reason 'No such file or directory'$")
        lines=[];paths=[]
        for line in text.splitlines():
            match=pattern.fullmatch(line)
            if match:
                paths.append({'pid':match[1],'file':match[2]})
                line="cannot remove file '<run-owned staging>/work/normalize/"+match[2]+"', reason 'No such file or directory'"
            lines.append(line)
        normalized.append(lines);observed.append(paths)
    if normalized[0]!=normalized[1]:raise ValueError('Warnings differ beyond run-owned cleanup paths')
    return {'historical_prefix':str(historical_prefix),'current_prefix':str(current_prefix),
            'historical_cleanup':observed[0],'current_cleanup':observed[1]}


def model_comparison(old,new,request):
    # No fitting. The R validator compares exact numerical/model state, including
    # closure bindings, and enumerates narrowly checked nondeterministic metadata.
    proc=subprocess.run([request['rscript'],'--vanilla',str(inv.HERE/'compare_models.R'),str(old),str(new),request['library']],
        env=ex.environment(),capture_output=True,text=True)
    if proc.returncode:raise ValueError('RDS semantic comparison failed: '+proc.stderr)
    import json
    value=json.loads(proc.stdout)
    if value.get('status')!='passed':raise ValueError('Incomplete RDS comparison')
    return {**value,'checker':io.record(inv.HERE/'compare_models.R'),'stderr':proc.stderr}


def compare(run,bundle,source):
    import numpy as np
    run,bundle,source=(Path(p).resolve(strict=True) for p in (run,bundle,source))
    contract=ex.verify_run(run,bundle);spec,files=ex.verify_bundle(bundle)
    records={r['source']:r for r in inv.verify_local()['files']}
    olddir=Path(spec['result_dir']);result=run/ex.result_path(spec)
    names=ex.TECH_FILES if spec['kind']=='technical' else (*ex.STORY_FILES,'values.npy','COMPLETE.json')
    def original(name):
        relative=(olddir/name).as_posix();path=io.within(source,relative);io.check(path,records[relative]);return path
    payloads={n:{'historical':io.record(original(n)),'current':io.record(result/n),
        'bytes_equal':original(n).read_bytes()==(result/n).read_bytes()} for n in names}
    exceptions={'model.rds','warnings.txt','resolved_config.json','sessionInfo.txt','COMPLETE.json'} if spec['kind']=='technical' else {'model.rds','R_COMPLETE.json','COMPLETE.json','sessionInfo.txt'}
    core=set(names)-exceptions
    if any(not payloads[n]['bytes_equal'] for n in core):
        raise ValueError('Non-identical core '+spec['case']+' result: '+', '.join(n for n in core if not payloads[n]['bytes_equal']))
    report={'case':spec['case'],'contract':contract,'payloads':payloads,'core_byte_identical':len(core),
        'total_files':len(names),'byte_identical_files':sum(r['bytes_equal'] for r in payloads.values()),
        'session':sessions(original('sessionInfo.txt').read_text(),(result/'sessionInfo.txt').read_text())}
    report['model_semantic_comparison']=model_comparison(original('model.rds'),result/'model.rds',io.read_json(run/'REQUEST.json'))
    if spec['kind']=='technical':
        oldcfg=io.read_json(original('resolved_config.json'));newcfg=io.read_json(result/'resolved_config.json')
        report['config_relocation']={'historical':oldcfg['preprocessing_file'],'current':newcfg['preprocessing_file']}
        oldprefix=Path(oldcfg['preprocessing_file']).parent.parent/'alignment'
        report['warnings_relocation']=warnings_relocation(original('warnings.txt').read_text(),(result/'warnings.txt').read_text(),oldprefix,result)
        oldcfg['preprocessing_file']=newcfg['preprocessing_file']
        if oldcfg!=newcfg:raise ValueError('Resolved configuration differs beyond record path')
        a=io.read_json(original('COMPLETE.json'));b=io.read_json(result/'COMPLETE.json')
        for value,folder in ((a,source/olddir),(b,result)):
            if value.get('status')!='COMPLETE' or set(value)!= {'status','outputs'} or set(value['outputs'])!=set(names)-{'COMPLETE.json'}:raise ValueError('Technical completion differs')
            for n,digest in value['outputs'].items():
                if io.sha(folder/n)!=digest:raise ValueError('Technical completion hash differs')
            for n in ('model.rds','warnings.txt','resolved_config.json','sessionInfo.txt'):value['outputs'][n]='<independently checked semantic or metadata comparison>'
        if a!=b:raise ValueError('Unexpected technical completion difference')
    else:
        report['elapsed']={n:elapsed(io.read_json(original(n)),io.read_json(result/n),n=='COMPLETE.json') for n in ('R_COMPLETE.json','COMPLETE.json')}
        a,b=(np.load(p,allow_pickle=False) for p in (original('values.npy'),result/'values.npy'))
        if a.dtype!=b.dtype or a.shape!=b.shape or not np.array_equal(a,b):raise ValueError('Historical NPY differs')
        report['npy_maximum_absolute_difference']=float(np.max(np.abs(a-b)))
    a,b=(io.table(p).iloc[:,1:].to_numpy(float) for p in (original('corrected_expression.csv.gz'),result/'corrected_expression.csv.gz'))
    if not np.array_equal(a,b):raise ValueError('Historical corrected values differ')
    report['csv_maximum_absolute_difference']=float(np.max(np.abs(a-b)))
    report['differing_payload_hash_pairs']={n:{'historical_sha256':r['historical']['sha256'],'current_sha256':r['current']['sha256']}
        for n,r in payloads.items() if not r['bytes_equal']}
    inp=io.table(bundle/'inputs'/spec['input']);canonical=io.table(source/spec['canonical']['evaluation_metadata'])
    if not inp[['cell_id','dataset','unit_key']].equals(canonical[['cell_id','dataset','unit_key']]):raise ValueError('Canonical evaluation identity differs')
    report['dataset_counts']=inp.groupby('dataset',sort=False).agg(cells=('cell_id','size'),units=('unit_key','nunique')).to_dict()
    report['runtime_copies']=[{'path':str(ex.runtime_path(r,spec)),**{k:r[k] for k in ('bytes','sha256')}} for r in files]
    report['bundle_manifest']=io.record(bundle/'MANIFEST.json');report['run_manifest']=io.record(run/'RUN_MANIFEST.json')
    return report
