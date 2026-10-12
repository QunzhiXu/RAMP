"""Create-only CytoNorm correction from accepted bundles; no preprocessing.

The frozen R runners/helpers are copied byte-for-byte. Technical configuration
changes only preprocessing_file, with an explicit hash-bound relocation record.
No historical fitted result is opened by fit or batch.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import cytonorm_io as io
import inventory as inv
import study
from probe_runtime import environment, runtime_contract

HERE = Path(__file__).resolve().parent
LOCALE = {'LC_ALL': 'C.UTF-8', 'LANG': 'C.UTF-8'}
PHASE1_SHA = 'c91c922cf0c4e08b950ca9f1bf84cd730190b3fd6db8744099c17cb5db591a36'
CORE_FILES = ('aggregate_cells.csv.gz', 'cell_clusters.csv.gz', 'cluster_batch_coverage.csv',
              'cluster_cv.csv', 'corrected_expression.csv.gz', 'model.rds',
              'row_metadata.csv.gz', 'spline_diagnostics.csv')
TECH_FILES = (*CORE_FILES, 'integrity.json', 'markers.txt', 'resolved_config.json',
              'sessionInfo.txt', 'warnings.txt', 'COMPLETE.json')
STORY_FILES = (*CORE_FILES, 'sessionInfo.txt', 'R_COMPLETE.json')


def code_records():
    return {n: io.record(HERE/n) for n in ('execution.py', 'check_environment.R', 'probe_runtime.py',
                                         'cytonorm_io.py', 'inventory.py', 'study.py')}


def accepted_bundle_manifests():
    report = HERE/'validation/handoff_20261011_r2/report.json'
    if io.sha(report)!=PHASE1_SHA: raise ValueError('Accepted phase-1 report changed')
    value=io.read_json(report)
    if value.get('status')!='passed': raise ValueError('Unaccepted phase-1 report')
    return {r['case']:r['manifest_sha256'] for r in value['bundles']}


def verify_bundle(bundle):
    # Bind provenance fields (including the protected source root), not merely
    # input payloads. A self-rehashed MANIFEST/COMPLETE is not a trust anchor.
    manifest=Path(bundle)/'MANIFEST.json';case=io.read_json(manifest)['case']
    if accepted_bundle_manifests().get(case)!=io.sha(manifest):
        raise ValueError('Input bundle is not the audited phase-1 manifest')
    return study.verify_bundle(bundle)


def runtime_path(row, spec):
    name = row['path']
    if name.startswith('adapter/'):
        return Path('runtime/tree/algorithm/CytoNorm')/Path(name).relative_to('adapter')
    if name.startswith('inputs/'):
        return Path('runtime/tree')/row['source']
    if name == 'environment/runtime_lock.csv':
        return Path('runtime/environment/runtime_lock.csv')
    if name in ('scripts/run_r.R', 'scripts/cytonorm_repro.R') and spec['kind']=='story':
        return Path('runtime/tree')/spec['base']/name
    raise ValueError('Unexpected runtime payload')


def result_path(spec):
    if spec['kind']=='technical':
        return Path('runtime/tree/algorithm_test/cytonorm_execution')/spec['case']/'result'
    return Path('runtime/tree')/spec['base']/'methods/cytonorm'


def relocated_config(run, bundle, spec):
    cfg = io.read_json(Path(bundle)/'inputs'/spec['config'])
    cfg['preprocessing_file'] = str((Path(run)/'runtime/tree'/spec['input_manifest']).resolve())
    return cfg


def check_relocation(run, bundle, spec):
    run, bundle = Path(run), Path(bundle)
    original = io.read_json(bundle/'inputs'/spec['config'])
    if io.read_json(run/'runtime_config.json') != relocated_config(run,bundle,spec):
        raise ValueError('Configuration differs beyond preprocessing path relocation')
    expected = {'field':'preprocessing_file', 'historical':original['preprocessing_file'],
        'runtime':str((run/'runtime/tree'/spec['input_manifest']).resolve()),
        'preprocessing_sha256':io.sha(bundle/'inputs'/spec['input_manifest']),
        'original_config':io.record(bundle/'inputs'/spec['config']),
        'runtime_config':io.record(run/'runtime_config.json')}
    if io.read_json(run/'CONFIG_RELOCATION.json') != expected:
        raise ValueError('Configuration relocation record differs')
    if original['preprocessing_sha256'] != expected['preprocessing_sha256']:
        raise ValueError('Relocated preprocessing record has different content')


def command_plan(run, spec, rscript, controller_root=None):
    run = Path(run); adapter = run/'runtime/tree/algorithm/CytoNorm'
    controller_root = HERE if controller_root is None else Path(controller_root)
    commands = {'environment':[rscript,'--vanilla',controller_root/'check_environment.R',
                               adapter,run/'runtime/environment/runtime_lock.csv']}
    if spec['kind']=='technical':
        commands['correction'] = [rscript,'--vanilla',adapter/'run_cytonorm.R',
            '--input',run/'runtime/tree'/spec['input'],'--markers',run/'runtime/tree'/spec['marker_file'],
            '--config',run/'runtime_config.json','--output',run/result_path(spec)]
    else:
        here = run/'runtime/tree'/spec['base']
        commands['correction'] = [rscript,'--vanilla',here/'scripts/run_r.R',here,'cytonorm']
    return {k:list(map(str,v)) for k,v in commands.items()}


def run_command(argv, root, name, env):
    with (root/(name+'.stdout.log')).open('x') as out, (root/(name+'.stderr.log')).open('x') as err:
        proc = subprocess.run(argv,cwd=root,env=env,stdout=out,stderr=err)
    if proc.returncode:
        raise RuntimeError(f'{name} exited {proc.returncode}; see preserved logs')


def check_environment_log(path):
    value = io.read_json(path); contract = runtime_contract()
    locale = {k:'C.UTF-8' for k in ('LC_COLLATE','LC_CTYPE','LC_MONETARY','LC_NUMERIC','LC_TIME','LC_MESSAGES')}
    locale['LC_NUMERIC'] = 'C'
    expected = {'R':'4.3.2','locale':locale,'environment':LOCALE,'runtime_versions':contract['versions'],
        'runtime_lock_sha256':contract['lock']['sha256'],
        'upstream_lock_sha256':io.sha(inv.SNAPSHOT/'adapter/upstream_lock.json')}
    if value != expected: raise ValueError('Runtime environment/lock differs')
    return value


def check_output(run, bundle, spec, converted=False):
    import numpy as np
    import pandas as pd
    run, bundle = Path(run), Path(bundle); result = run/result_path(spec)
    inp = io.table(bundle/'inputs'/spec['input']); out = io.table(result/'corrected_expression.csv.gz')
    identity = inp[['cell_id','dataset','unit_key']]
    if list(out.columns)!=['cell_id',*spec['markers']] or not np.array_equal(out.cell_id,inp.cell_id):
        raise ValueError('Corrected cell/marker order differs')
    if not np.isfinite(out.iloc[:,1:].to_numpy(float)).all(): raise ValueError('Non-finite corrected values')
    if not io.table(result/'row_metadata.csv.gz').equals(identity): raise ValueError('Output identity differs')
    clusters = io.table(result/'cell_clusters.csv.gz')
    if (list(clusters.columns)!=['cell_id','dataset','unit_key','cluster'] or
        not clusters.iloc[:,:3].equals(identity) or not clusters.cluster.isin(range(1,11)).all()):
        raise ValueError('Cluster identity or range differs')
    controls = io.table(result/'aggregate_cells.csv.gz')
    if list(controls.columns)!=list(identity.columns) or controls.cell_id.duplicated().any():
        raise ValueError('Invalid aggregate controls')
    lookup = identity.set_index('cell_id',drop=False)
    if not controls.cell_id.isin(lookup.index).all() or not controls.equals(lookup.loc[controls.cell_id].reset_index(drop=True)):
        raise ValueError('Control identity differs')
    counts = identity.groupby('unit_key').size().clip(upper=2000)
    if not controls.groupby('unit_key').size().equals(counts): raise ValueError('Per-unit control cap differs')
    if spec['kind']=='technical':
        required = set(TECH_FILES); done = io.read_json(result/'COMPLETE.json')
        if done.get('status')!='COMPLETE' or set(done)!= {'status','outputs'} or set(done['outputs'])!=required-{'COMPLETE.json'}:
            raise ValueError('Technical nested completion differs')
        for name,digest in done['outputs'].items():
            if io.sha(io.within(result,name))!=digest: raise ValueError('R payload hash differs')
        integrity = io.read_json(result/'integrity.json')
        check_relocation(run,bundle,spec)
        if io.read_json(result/'resolved_config.json')!=relocated_config(run,bundle,spec):
            raise ValueError('R resolved configuration differs')
        if io.sha(result/'markers.txt')!=io.sha(bundle/'inputs'/spec['marker_file']): raise ValueError('R marker order differs')
    else:
        required = set(STORY_FILES); rc = io.read_json(result/'R_COMPLETE.json')
        expected_keys = {'method','input_sha256','helper_sha256','parameters','result','elapsed_seconds','output_sha256'}
        if (set(rc)!=expected_keys or rc['method']!='cytonorm' or rc['parameters']!=inv.PARAMETERS or
            rc['input_sha256']!=io.sha(bundle/'inputs'/spec['input']) or
            rc['helper_sha256']!=io.sha(bundle/'scripts/cytonorm_repro.R') or
            rc['output_sha256']!=io.sha(result/'corrected_expression.csv.gz')):
            raise ValueError('Story R contract differs')
        if isinstance(rc['elapsed_seconds'],bool) or not isinstance(rc['elapsed_seconds'],(int,float)) or not np.isfinite(rc['elapsed_seconds']) or rc['elapsed_seconds']<0:
            raise ValueError('Invalid elapsed time')
        integrity = rc['result']
        if converted:
            required |= {'values.npy','COMPLETE.json'}
            values = np.load(result/'values.npy',allow_pickle=False)
            parsed = pd.read_csv(result/'corrected_expression.csv.gz',low_memory=False).iloc[:,1:].to_numpy(float)
            if values.dtype!=np.float64 or not np.array_equal(values,parsed): raise ValueError('NPY serialization differs')
            expected = {'method':'cytonorm','cells':spec['cells'],'dimensions':len(spec['markers']),'runtime':rc,
                'values_sha256':io.sha(result/'values.npy'),'cell_ids_sha256':io.sha(bundle/'inputs'/spec['ids']),
                'input_manifest_sha256':io.sha(bundle/'inputs'/spec['input_manifest'])}
            if io.read_json(result/'COMPLETE.json')!=expected: raise ValueError('Story completion differs')
    # Integrity is a frozen scientific contract, not used to alter the fit.
    if integrity != spec['integrity']: raise ValueError('Scientific integrity contract differs')
    if {p.name for p in result.iterdir()}!=required: raise ValueError('Extra/missing result files')
    return {'cells':len(inp),'markers':len(spec['markers']),'units':int(inp.unit_key.nunique()),
        'aggregate_cells':len(controls),'ordered_identity_matches':True,'output_type':'corrected_marker_values',
        'spline_replay_max_absolute_error':integrity['spline_replay_max_absolute_error']}


def finish_story(run, bundle, spec):
    import numpy as np
    import pandas as pd
    check_output(run,bundle,spec)
    result = run/result_path(spec); parsed = pd.read_csv(result/'corrected_expression.csv.gz',low_memory=False)
    if not np.array_equal(parsed.cell_id.astype(str),np.load(bundle/'inputs'/spec['ids'],allow_pickle=False)):
        raise ValueError('Default CSV parser changed IDs')
    values = parsed.iloc[:,1:].to_numpy(float)
    if not np.isfinite(values).all(): raise ValueError('Non-finite NPY')
    with (result/'values.npy').open('xb') as stream: np.save(stream,values)
    io.create_json(result/'COMPLETE.json',{'method':'cytonorm','cells':len(values),'dimensions':values.shape[1],
        'values_sha256':io.sha(result/'values.npy'),'cell_ids_sha256':io.sha(bundle/'inputs'/spec['ids']),
        'input_manifest_sha256':io.sha(bundle/'inputs'/spec['input_manifest']),
        'runtime':io.read_json(result/'R_COMPLETE.json')})


def fit(bundle, library, rscript, output, scratch_parent, expected_codes=None):
    bundle=Path(bundle).resolve(strict=True);library=Path(library).resolve(strict=True)
    rscript=Path(rscript).absolute();scratch=Path(scratch_parent).resolve(strict=True)
    if not library.is_dir() or not scratch.is_dir() or any(c.isspace() for c in str(scratch)):
        raise ValueError('Use an existing library and space-free R scratch directory')
    spec,files=verify_bundle(bundle);codes=code_records();contract=runtime_contract(library)
    if expected_codes is not None and codes!=expected_codes: raise ValueError('Controller changed during batch')
    executable=io.record(rscript);bm=io.record(bundle/'MANIFEST.json')
    source=io.read_json(bundle/'MANIFEST.json')['source_root_at_preparation']
    root=io.reserve(output,(bundle,library.parent,rscript,inv.SNAPSHOT,source,inv.LEGACY_SOURCE,*(HERE/n for n in codes)))
    try:
        io.create_json(root/'REQUEST.json',{'case':spec['case'],'bundle':str(bundle),'bundle_manifest':bm,
            'library':str(library),'rscript':str(rscript),'rscript_record':executable,'code_records':codes,
            'catalog_sha256':inv.CATALOG_SHA,'r_locale_environment':LOCALE,'runtime_contract':contract,
            'history_used_by_fit':False,'controller_root':str(HERE)})
        for row in files: io.copy_new(bundle/row['path'],root/runtime_path(row,spec),row)
        adapter=root/'runtime/tree/algorithm/CytoNorm';(adapter/'.Rlib').symlink_to(library,target_is_directory=True)
        if spec['kind']=='technical':
            cfg=relocated_config(root,bundle,spec);io.create_json(root/'runtime_config.json',cfg)
            original=io.read_json(bundle/'inputs'/spec['config'])
            io.create_json(root/'CONFIG_RELOCATION.json',{'field':'preprocessing_file',
                'historical':original['preprocessing_file'],'runtime':cfg['preprocessing_file'],
                'preprocessing_sha256':io.sha(bundle/'inputs'/spec['input_manifest']),
                'original_config':io.record(bundle/'inputs'/spec['config']),
                'runtime_config':io.record(root/'runtime_config.json')})
            check_relocation(root,bundle,spec)
            (root/result_path(spec)).parent.mkdir(parents=True,exist_ok=False)
        commands=command_plan(root,spec,rscript);io.create_json(root/'COMMANDS.json',commands)
        with tempfile.TemporaryDirectory(prefix='cytonorm_fit_',dir=scratch) as tmp:
            env={**environment(),'TMPDIR':str(Path(tmp).resolve())}
            run_command(commands['environment'],root,'environment_before',env)
            check_environment_log(root/'environment_before.stdout.log')
            run_command(commands['correction'],root,'correction',env)
            run_command(commands['environment'],root,'environment_after',env)
        check_environment_log(root/'environment_after.stdout.log')
        if (root/'environment_before.stdout.log').read_bytes()!=(root/'environment_after.stdout.log').read_bytes():
            raise ValueError('Environment changed during fitting')
        if spec['kind']=='story': finish_story(root,bundle,spec)
        check_output(root,bundle,spec,converted=spec['kind']=='story')
        verify_bundle(bundle);io.check(bundle/'MANIFEST.json',bm);io.check(rscript,executable)
        if runtime_contract(library)!=contract: raise ValueError('Runtime lock changed during fitting')
        for n,row in codes.items(): io.check(HERE/n,row)
        for row in files: io.check(root/runtime_path(row,spec),row)
        records={p.relative_to(root).as_posix():io.record(p) for p in sorted(root.rglob('*')) if p.is_file() and not p.is_symlink()}
        links={'runtime/tree/algorithm/CytoNorm/.Rlib':str(library)}
        io.payload(root,records,allowed_links=links)
        io.create_json(root/'RUN_MANIFEST.json',{'schema':1,'status':'complete','case':spec['case'],'files':records,
            'links':links,'history_used_by_fit':False,'scope':'Frozen-input CytoNorm correction only'})
        io.create_json(root/'COMPLETE.json',{'status':'complete','manifest_sha256':io.sha(root/'RUN_MANIFEST.json')})
    except BaseException as error: io.failure(root,error);raise
    return root


def verify_run(run, bundle):
    run,bundle=Path(run).resolve(strict=True),Path(bundle).resolve(strict=True)
    spec,files=verify_bundle(bundle);m=io.complete(run,'RUN_MANIFEST.json');req=io.read_json(run/'REQUEST.json')
    if (m.get('status')!='complete' or m['case']!=spec['case'] or req['case']!=spec['case'] or
        req['catalog_sha256']!=inv.CATALOG_SHA or req['r_locale_environment']!=LOCALE or
        req['history_used_by_fit'] is not False or m['history_used_by_fit'] is not False or
        req['runtime_contract']!=runtime_contract() or req['code_records']!=code_records()):
        raise ValueError('Run/code/lock contract differs')
    io.check(bundle/'MANIFEST.json',req['bundle_manifest'])
    links={'runtime/tree/algorithm/CytoNorm/.Rlib':req['library']}
    if m['links']!=links: raise ValueError('Library binding differs')
    io.payload(run,m['files'],('RUN_MANIFEST.json','COMPLETE.json'),links)
    for row in files: io.check(run/runtime_path(row,spec),row)
    if not Path(req['controller_root']).is_absolute(): raise ValueError('Invalid recorded controller root')
    if io.read_json(run/'COMMANDS.json')!=command_plan(run,spec,req['rscript'],req['controller_root']):
        raise ValueError('Unexpected fit commands')
    before=check_environment_log(run/'environment_before.stdout.log')
    if before!=check_environment_log(run/'environment_after.stdout.log'): raise ValueError('Environment changed')
    return check_output(run,bundle,spec,converted=spec['kind']=='story')


def batch(bundles, library, rscript, output, scratch):
    bundles=Path(bundles).resolve(strict=True);specs=inv.verify_local()['branches'];codes=code_records()
    sources=[];paths=[]
    for spec in specs:
        bundle=bundles/(spec['case']+'_inputs_20261011');observed,_=verify_bundle(bundle)
        if observed!=spec: raise ValueError('Case-named bundle contains a different cohort')
        paths.append(bundle);sources.append(io.read_json(bundle/'MANIFEST.json')['source_root_at_preparation'])
    # Protect all authenticated source roots before even creating the batch parent.
    root=io.reserve(output,(*paths,*sources,Path(library).resolve().parent,rscript,inv.SNAPSHOT,inv.LEGACY_SOURCE,*(HERE/n for n in codes)))
    try:
        io.create_json(root/'BATCH_REQUEST.json',{'code_records':codes,'cases':[s['case'] for s in specs],
            'bundle_manifests':{s['case']:io.record(p/'MANIFEST.json') for s,p in zip(specs,paths)}})
        rows=[]
        for spec,bundle in zip(specs,paths):
            case=spec['case'];print('START '+case,flush=True)
            run=fit(bundle,library,rscript,root/case,scratch,expected_codes=codes)
            result=verify_run(run,bundle)
            rows.append({'case':case,'bundle':str(bundle),'run':case,'manifest':io.record(run/'RUN_MANIFEST.json'),'result':result})
            print('PASS '+case,flush=True)
        # Recheck every run and controller after the last fit, before batch completion.
        for row,bundle in zip(rows,paths):
            verify_run(root/row['run'],bundle);io.check(root/row['run']/'RUN_MANIFEST.json',row['manifest'])
            if io.read_json(root/row['run']/'REQUEST.json')['code_records']!=codes: raise ValueError('Mixed batch controllers')
        if code_records()!=codes: raise ValueError('Controller changed during batch')
        io.create_json(root/'BATCH_MANIFEST.json',{'status':'complete','cases':rows,'code_records':codes,'history_used_by_fit':False})
        io.create_json(root/'COMPLETE.json',{'status':'complete','manifest_sha256':io.sha(root/'BATCH_MANIFEST.json')})
    except BaseException as error: io.failure(root,error);raise
    return root


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('fit','batch','verify-run'))
    for n in ('bundle','bundles','library','rscript','output','run','scratch-parent'): p.add_argument('--'+n,type=Path)
    a=p.parse_args()
    if a.command=='verify-run':
        if not all((a.run,a.bundle)): p.error('verify-run requires run and bundle')
        print(json.dumps(verify_run(a.run,a.bundle)));return
    if not all((a.library,a.rscript,a.output,a.scratch_parent)): p.error('fit/batch require library, rscript, output, scratch-parent')
    if a.command=='batch':
        if a.bundles is None: p.error('batch requires bundles')
        print(batch(a.bundles,a.library,a.rscript,a.output,a.scratch_parent))
    else:
        if a.bundle is None: p.error('fit requires bundle')
        print(fit(a.bundle,a.library,a.rscript,a.output,a.scratch_parent))


if __name__=='__main__': main()
