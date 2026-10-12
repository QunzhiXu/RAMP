"""Read-only, strict historical comparison. Fitting never imports this module."""
import copy
import math
from pathlib import Path

import execution as ex
import fastmnn_io as io
import inventory as inv
import study


def sessions(old, new):
    normalized, observed = [], []
    for text in (old,new):
        lines = text.splitlines(); indices = [i for i,line in enumerate(lines) if line.startswith('Running under: ')]
        if len(indices)!=1 or not lines[indices[0]].startswith('Running under: macOS '): raise ValueError('Unexpected OS line')
        observed.append(lines[indices[0]]); lines[indices[0]] = '<recorded macOS release>'; normalized.append(lines)
    if normalized[0]!=normalized[1]: raise ValueError('Session differs beyond macOS release')
    return {'historical':observed[0],'current':observed[1]}


def elapsed(old, new, nested=False):
    copies = [copy.deepcopy(old),copy.deepcopy(new)]; times = []
    for value in copies:
        target = value['runtime'] if nested else value; time = target['elapsed_seconds']
        if isinstance(time,bool) or not isinstance(time,(int,float)) or not math.isfinite(time) or time<0:
            raise ValueError('Invalid elapsed time')
        times.append(time); target['elapsed_seconds'] = '<recorded elapsed seconds>'
    if copies[0]!=copies[1]: raise ValueError('Completion differs beyond elapsed time')
    return {'historical_seconds':times[0],'current_seconds':times[1]}


def resolved_config(old, new):
    a,b = copy.deepcopy(old),copy.deepcopy(new)
    paths = [v['input_scale']['preprocessing_record_file'] for v in (a,b)]
    for v in (a,b): v['input_scale']['preprocessing_record_file'] = '<authenticated relocated record>'
    if a!=b: raise ValueError('Resolved configuration differs beyond record path')
    return {'historical_record':paths[0],'current_record':paths[1]}


def technical_manifest(old, new):
    # Every payload is independently compared before this is called. Normalize
    # hashes of only the two metadata payloads explicitly allowed to differ.
    a,b = copy.deepcopy(old),copy.deepcopy(new)
    for m in (a,b):
        m['created_at'] = '<recorded time>'
        for key in ('path','marker_file','preprocessing_record'): m['input'][key] = '<relocated path>'
        m['config_file']['path'] = '<relocated path>'
        for row in m['implementation'].values(): row['path'] = '<relocated path>'
        for row in m['output_files']:
            if row['file'] in ('resolved_config.json','sessionInfo.txt'): row['sha256'] = '<independently checked metadata>'
    if a!=b: raise ValueError('Manifest differs beyond enumerated provenance fields')


def compare(run, bundle, source):
    import numpy as np
    run,bundle,source = (Path(p).resolve(strict=True) for p in (run,bundle,source))
    contract = ex.verify_run(run,bundle); spec,files = study.plan(io.read_json(bundle/'MANIFEST.json')['case'])
    records = {r['source']:r for r in inv.verify_local()['files']}
    olddir = Path(spec['output']).parent; result = run/ex.result_path(spec)
    names = (*ex.TECH_FILES,'manifest.json','COMPLETE.json') if spec['kind']=='technical' else (*ex.STORY_FILES,'values.npy','COMPLETE.json')

    def original(name):
        relative = (olddir/name).as_posix(); path = io.within(source,relative); io.check(path,records[relative]); return path

    payloads = {n:{'historical':io.record(original(n)),'current':io.record(result/n),
                  'bytes_equal':original(n).read_bytes()==(result/n).read_bytes()} for n in names}
    exceptions = {'resolved_config.json','sessionInfo.txt','manifest.json','COMPLETE.json'} if spec['kind']=='technical' else {'R_COMPLETE.json','COMPLETE.json','sessionInfo.txt'}
    core = set(names)-exceptions
    if any(not payloads[n]['bytes_equal'] for n in core):
        raise ValueError('Non-identical core '+spec['case']+' result: '+', '.join(n for n in core if not payloads[n]['bytes_equal']))
    report = {'case':spec['case'],'contract':contract,'payloads':payloads,'core_byte_identical':len(core),
              'total_files':len(names),'byte_identical_files':sum(r['bytes_equal'] for r in payloads.values()),
              'session':sessions(original('sessionInfo.txt').read_text(),(result/'sessionInfo.txt').read_text())}
    if spec['kind']=='technical':
        report['config_relocation'] = resolved_config(io.read_json(original('resolved_config.json')),io.read_json(result/'resolved_config.json'))
        technical_manifest(io.read_json(original('manifest.json')),io.read_json(result/'manifest.json'))
        for directory in (olddir,ex.result_path(spec)):
            root = source if directory==olddir else run
            done = io.read_json(root/directory/'COMPLETE.json')
            if done!={'status':'complete','schema_version':'final_xu.fastmnn_repro.v1',
                      'manifest_sha256':io.sha(root/directory/'manifest.json')}:
                raise ValueError('Technical completion contract differs')
    else:
        report['elapsed'] = {n:elapsed(io.read_json(original(n)),io.read_json(result/n),n=='COMPLETE.json')
                             for n in ('R_COMPLETE.json','COMPLETE.json')}
        a,b = (np.load(p,allow_pickle=False) for p in (original('values.npy'),result/'values.npy'))
        if a.dtype!=b.dtype or a.shape!=b.shape or not np.array_equal(a,b): raise ValueError('Historical NPY differs')
        report['npy_maximum_absolute_difference'] = float(np.max(np.abs(a-b)))
    filename = Path(spec['output']).name
    a,b = (io.table(p).iloc[:,1:].to_numpy(float) for p in (original(filename),result/filename))
    if not np.array_equal(a,b): raise ValueError('Historical CSV coordinates differ')
    report['csv_maximum_absolute_difference'] = float(np.max(np.abs(a-b)))
    report['differing_payload_hash_pairs'] = {n:{'historical_sha256':r['historical']['sha256'],'current_sha256':r['current']['sha256']}
                                            for n,r in payloads.items() if not r['bytes_equal']}
    inp = io.table(bundle/'inputs'/spec['input'])
    canonical = io.table(source/spec['canonical']['evaluation_metadata'])
    if not inp[['cell_id','dataset','unit_key']].equals(canonical[['cell_id','dataset','unit_key']]):
        raise ValueError('Canonical evaluation identity differs')
    report['dataset_counts'] = inp.groupby('dataset',sort=False).agg(cells=('cell_id','size'),units=('unit_key','nunique')).to_dict()
    report['runtime_copies'] = [{'path':str(ex.runtime_path(r,spec)),**{k:r[k] for k in ('bytes','sha256')}} for r in files]
    report['bundle_manifest'] = io.record(bundle/'MANIFEST.json'); report['run_manifest'] = io.record(run/'RUN_MANIFEST.json')
    return report
