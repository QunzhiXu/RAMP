"""Create-only real fastMNN execution from phase-1 bundles; no preprocessing.

Technical runs use the unchanged CLI/config. Story runs use the unchanged
run_r.R. This controller is separate so every accepted phase-1 file stays intact.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import fastmnn_io as io
import inventory as inv
import study
from probe_runtime import environment

HERE = Path(__file__).resolve().parent
LOCALE = {'LC_ALL': 'C.UTF-8', 'LANG': 'C.UTF-8'}
TECH_FILES = (
    'authenticated_preprocessing_manifest.json', 'cell_metadata_label_free.csv.gz',
    'fastmnn_corrected_embedding.csv.gz', 'fastmnn_corrected_embedding.rds',
    'fastmnn_lost_variance.csv', 'fastmnn_merge_diagnostics.csv', 'fastmnn_merge_info.rds',
    'fastmnn_pca_centers.csv', 'fastmnn_pca_rotation.csv.gz', 'fastmnn_pca_variance.csv',
    'fastmnn_runtime_messages.txt', 'input_cell_l2_statistics.csv', 'input_column_audit.json',
    'input_marker_batch_statistics.csv', 'method_semantics.json', 'resolved_config.json', 'sessionInfo.txt')
STORY_FILES = ('corrected_expression.csv.gz', 'fit.rds', 'cell_metadata.csv.gz',
               'marker_batch_statistics.csv.gz', 'cell_l2_statistics.csv.gz', 'sessionInfo.txt', 'R_COMPLETE.json')


def code_records():
    return {n: io.record(HERE/n) for n in ('execution.py', 'check_environment.R', 'probe_runtime.py',
                                         'fastmnn_io.py', 'inventory.py', 'study.py')}


def runtime_path(row, spec):
    if row['path'].startswith('adapter/'):
        return Path('runtime/tree/algorithm/fastmnn')/Path(row['path']).relative_to('adapter')
    if row['path'].startswith('inputs/'):
        return Path('runtime/tree')/row['source']
    if row['path']=='scripts/run_r.R' and spec['kind']=='story':
        return Path('runtime/tree')/spec['base']/'scripts/run_r.R'
    raise ValueError('Unexpected runtime payload')


def result_path(spec):
    return Path('result') if spec['kind']=='technical' else Path('runtime/tree')/spec['base']/'methods/fastmnn'


def command_plan(run, spec, rscript, controller_root=None):
    run = Path(run); adapter = run/'runtime/tree/algorithm/fastmnn'
    controller_root = HERE if controller_root is None else Path(controller_root)
    commands = {'environment': [rscript, '--vanilla', controller_root/'check_environment.R', adapter, spec['kind']]}
    if spec['kind']=='technical':
        commands['correction'] = [rscript, '--vanilla', adapter/'run_fastmnn.R',
            '--input', run/'runtime/tree'/spec['input'], '--markers', run/'runtime/tree'/spec['marker_file'],
            '--config', run/'runtime/tree'/spec['config'], '--output', run/result_path(spec)]
    else:
        here = run/'runtime/tree'/spec['base']
        commands['correction'] = [rscript, '--vanilla', here/'scripts/run_r.R', here, 'fastmnn']
    return {k: list(map(str, v)) for k, v in commands.items()}


def run_command(argv, output, name, env):
    with (output/(name+'.stdout.log')).open('x') as out, (output/(name+'.stderr.log')).open('x') as err:
        proc = subprocess.run(argv, cwd=output, env=env, stdout=out, stderr=err)
    if proc.returncode: raise RuntimeError(f'{name} exited {proc.returncode}; see preserved logs')


def check_environment_log(path, kind):
    value = json.loads(Path(path).read_text())
    expected = {k: 'C.UTF-8' for k in ('LC_COLLATE','LC_CTYPE','LC_MONETARY','LC_NUMERIC','LC_TIME','LC_MESSAGES')}
    expected['LC_NUMERIC'] = 'C'
    lock = io.table(inv.SNAPSHOT/'adapter/runtime_lock.csv').to_dict('records')
    if value['environment']!=LOCALE or value['locale']!=expected or value['runtime_lock']!=lock:
        raise ValueError('Runtime locale/dependency lock differs')
    if value['extra_serialization'] != ({'data.table':'1.17.8'} if kind=='story' else None):
        raise ValueError('Story serialization environment differs')
    return value


def check_output(run, bundle, spec, converted=False):
    import numpy as np
    import pandas as pd
    run, bundle = Path(run), Path(bundle); result = run/result_path(spec)
    inp = io.table(bundle/'inputs'/spec['input'])
    name = 'fastmnn_corrected_embedding.csv.gz' if spec['kind']=='technical' else 'corrected_expression.csv.gz'
    out = io.table(result/name)
    if list(out.columns)!=['cell_id', *spec['embedding_columns']] or not np.array_equal(out.cell_id, inp.cell_id):
        raise ValueError('Embedding cell/PC order differs')
    if not np.isfinite(out.iloc[:,1:].to_numpy(float)).all(): raise ValueError('Non-finite embedding')
    meta = io.table(result/('cell_metadata_label_free.csv.gz' if spec['kind']=='technical' else 'cell_metadata.csv.gz'))
    if (list(meta.columns)!=['cell_id','batch','sample'] or not np.array_equal(meta.cell_id, inp.cell_id)
        or not np.array_equal(meta.batch, inp.dataset) or not np.array_equal(meta['sample'], inp.unit_key)):
        raise ValueError('Output dataset/unit provenance differs')
    if spec['kind']=='technical':
        m = io.complete(result,'manifest.json'); alg = m['algorithm']; mi = m['input']
        if (m.get('status')!='complete' or alg['resolved_parameters']!=inv.PARAMETERS or
            (alg['requested_d'],alg['resolved_d'],alg['actual_d'])!=(50,spec['dimensions'],spec['dimensions']) or
            alg['reconstructed_assay_exported'] is not False or mi['markers']!=spec['markers'] or
            mi['n_cells']!=spec['cells'] or mi['n_markers']!=len(spec['markers']) or
            mi['batch_levels_in_progressive_merge_order']!=spec['datasets']):
            raise ValueError('Technical scientific contract differs')
        for key, target in [('sha256',spec['input']), ('marker_file_sha256',spec['marker_file']),
                            ('preprocessing_record_sha256',spec['record'])]:
            if mi[key]!=io.sha(bundle/'inputs'/target): raise ValueError('R input hash differs')
        for key, target in [('path',spec['input']),('marker_file',spec['marker_file']),('preprocessing_record',spec['record'])]:
            if mi[key]!=str((run/'runtime/tree'/target).resolve()): raise ValueError('R read an unexpected input path')
        for key, name in [('wrapper_file','fastmnn_repro.R'),('runner_file','run_fastmnn.R'),
                          ('upstream_lock','upstream_lock.json'),('runtime_lock','runtime_lock.csv')]:
            if m['implementation'][key] != {'path':str(run/'runtime/tree/algorithm/fastmnn'/name),
                                           'sha256':io.sha(bundle/'adapter'/name)}:
                raise ValueError('R implementation provenance differs')
        if m['config_file']!={'path':str(run/'runtime/tree'/spec['config']), 'sha256':io.sha(bundle/'inputs'/spec['config'])}:
            raise ValueError('R config differs')
        cfg = io.read_json(bundle/'inputs'/spec['config'])
        cfg['input_scale']['preprocessing_record_file'] = str((run/'runtime/tree'/spec['record']).resolve())
        if io.read_json(result/'resolved_config.json')!=cfg: raise ValueError('Resolved config differs')
        if io.sha(result/'authenticated_preprocessing_manifest.json')!=io.sha(bundle/'inputs'/spec['record']):
            raise ValueError('Authenticated record differs')
        records = {r['file']:r['sha256'] for r in m['output_files']}
        if set(records)!=set(TECH_FILES) or len(m['output_files'])!=len(TECH_FILES): raise ValueError('Wrong technical payload set')
        for name, digest in records.items():
            if io.sha(io.within(result,name))!=digest: raise ValueError('R payload hash differs')
        required = set(TECH_FILES)|{'manifest.json','COMPLETE.json'}
    else:
        rc = io.read_json(result/'R_COMPLETE.json')
        if (rc['method']!='fastmnn' or rc['parameters']!=inv.PARAMETERS or rc['result']!=
            {'cells':spec['cells'],'dimensions':spec['dimensions'],'row_identity_verified':True,'output_type':'embedding'} or
            rc['input_sha256']!=io.sha(bundle/'inputs'/spec['input']) or
            rc['helper_sha256']!=io.sha(bundle/'adapter/fastmnn_repro.R') or rc['output_sha256']!=io.sha(result/name)):
            raise ValueError('Story R output contract differs')
        required = set(STORY_FILES)
        if converted:
            required |= {'values.npy','COMPLETE.json'}
            values = np.load(result/'values.npy',allow_pickle=False)
            parsed = pd.read_csv(result/name,low_memory=False).iloc[:,1:].to_numpy(float)
            if values.dtype!=np.float64 or not np.array_equal(values,parsed): raise ValueError('NPY serialization differs')
            done = io.read_json(result/'COMPLETE.json')
            expected = {'method':'fastmnn','cells':spec['cells'],'dimensions':spec['dimensions'],'runtime':rc,
                'values_sha256':io.sha(result/'values.npy'),'cell_ids_sha256':io.sha(bundle/'inputs'/spec['ids']),
                'input_manifest_sha256':io.sha(bundle/'inputs'/spec['input_manifest'])}
            if done!=expected: raise ValueError('Story completion differs')
    if {p.name for p in result.iterdir()}!=required: raise ValueError('Extra/missing result files')
    return {'cells':len(inp),'dimensions':spec['dimensions'],'markers':len(spec['markers']),
            'units':int(inp.unit_key.nunique()),'ordered_identity_matches':True,'output_type':'corrected_latent_embedding'}


def finish_story(run, bundle, spec):
    import numpy as np
    import pandas as pd
    check_output(run,bundle,spec)
    result = run/result_path(spec)
    parsed = pd.read_csv(result/'corrected_expression.csv.gz',low_memory=False)
    if not np.array_equal(parsed.cell_id.astype(str),np.load(bundle/'inputs'/spec['ids'],allow_pickle=False)):
        raise ValueError('Default CSV parser changed IDs')
    values = parsed.iloc[:,1:].to_numpy(float)
    if not np.isfinite(values).all(): raise ValueError('Non-finite NPY')
    with (result/'values.npy').open('xb') as stream: np.save(stream,values)
    io.create_json(result/'COMPLETE.json',{'method':'fastmnn','cells':len(values),'dimensions':values.shape[1],
        'values_sha256':io.sha(result/'values.npy'),'cell_ids_sha256':io.sha(bundle/'inputs'/spec['ids']),
        'input_manifest_sha256':io.sha(bundle/'inputs'/spec['input_manifest']),
        'runtime':io.read_json(result/'R_COMPLETE.json')})


def fit(bundle, library, rscript, output, scratch_parent):
    bundle = Path(bundle).resolve(strict=True); library = Path(library).resolve(strict=True)
    rscript = Path(rscript).absolute(); scratch = Path(scratch_parent).resolve(strict=True)
    if not library.is_dir() or not scratch.is_dir() or any(c.isspace() for c in str(scratch)):
        raise ValueError('Use an existing library and space-free R scratch directory')
    spec, files = study.verify_bundle(bundle)
    codes = code_records(); executable = io.record(rscript); bm = io.record(bundle/'MANIFEST.json')
    source = io.read_json(bundle/'MANIFEST.json')['source_root_at_preparation']
    root = io.reserve(output,(bundle,library.parent,rscript,inv.SNAPSHOT,source,*(HERE/n for n in codes)))
    try:
        io.create_json(root/'REQUEST.json',{'case':spec['case'],'bundle':str(bundle),'bundle_manifest':bm,
            'library':str(library),'rscript':str(rscript),'rscript_record':executable,'code_records':codes,
            'catalog_sha256':inv.CATALOG_SHA,'r_locale_environment':LOCALE,'history_used_by_fit':False,
            'controller_root':str(HERE)})
        for row in files: io.copy_new(bundle/row['path'],root/runtime_path(row,spec),row)
        adapter = root/'runtime/tree/algorithm/fastmnn'
        (adapter/'.Rlib').symlink_to(library,target_is_directory=True)
        if spec['kind']=='technical':
            cfg_path = root/'runtime/tree'/spec['config']; cfg = io.read_json(cfg_path)
            if (cfg_path.parent/cfg['input_scale']['preprocessing_record_file']).resolve()!=root/'runtime/tree'/spec['record']:
                raise ValueError('Technical relative preprocessing path is not mirrored')
        commands = command_plan(root,spec,rscript); io.create_json(root/'COMMANDS.json',commands)
        with tempfile.TemporaryDirectory(prefix='fastmnn_fit_',dir=scratch) as tmp:
            env = {**environment(),'TMPDIR':str(Path(tmp).resolve())}
            run_command(commands['environment'],root,'environment_before',env)
            check_environment_log(root/'environment_before.stdout.log',spec['kind'])
            run_command(commands['correction'],root,'correction',env)
            run_command(commands['environment'],root,'environment_after',env)
        check_environment_log(root/'environment_after.stdout.log',spec['kind'])
        if (root/'environment_before.stdout.log').read_bytes()!=(root/'environment_after.stdout.log').read_bytes():
            raise ValueError('Environment changed during fitting')
        if spec['kind']=='story': finish_story(root,bundle,spec)
        check_output(root,bundle,spec,converted=spec['kind']=='story')
        study.verify_bundle(bundle); io.check(bundle/'MANIFEST.json',bm); io.check(rscript,executable)
        for n,row in codes.items(): io.check(HERE/n,row)
        for row in files: io.check(root/runtime_path(row,spec),row)
        records = {p.relative_to(root).as_posix():io.record(p) for p in sorted(root.rglob('*')) if p.is_file() and not p.is_symlink()}
        links = {'runtime/tree/algorithm/fastmnn/.Rlib':str(library)}
        io.payload(root,records,allowed_links=links)
        io.create_json(root/'RUN_MANIFEST.json',{'schema':1,'status':'complete','case':spec['case'],'files':records,
            'links':links,'history_used_by_fit':False,'scope':'Frozen-input fastMNN corrected embedding only'})
        io.create_json(root/'COMPLETE.json',{'status':'complete','manifest_sha256':io.sha(root/'RUN_MANIFEST.json')})
    except BaseException as error: io.failure(root,error); raise
    return root


def verify_run(run, bundle):
    run, bundle = Path(run).resolve(strict=True), Path(bundle).resolve(strict=True)
    spec, files = study.verify_bundle(bundle); m = io.complete(run,'RUN_MANIFEST.json'); req = io.read_json(run/'REQUEST.json')
    if (m.get('status')!='complete' or m['case']!=spec['case'] or req['case']!=spec['case'] or
        req['catalog_sha256']!=inv.CATALOG_SHA or req['r_locale_environment']!=LOCALE or
        req['history_used_by_fit'] is not False or m['history_used_by_fit'] is not False): raise ValueError('Run contract differs')
    io.check(bundle/'MANIFEST.json',req['bundle_manifest'])
    if req['code_records']!=code_records(): raise ValueError('Execution code changed since fit')
    links = {'runtime/tree/algorithm/fastmnn/.Rlib':req['library']}
    if m['links']!=links: raise ValueError('Library binding differs')
    io.payload(run,m['files'],('RUN_MANIFEST.json','COMPLETE.json'),links)
    for row in files: io.check(run/runtime_path(row,spec),row)
    # Authenticate code by its hash, not by the verifier's current filesystem
    # location. The command log keeps the controller's recorded launch path.
    if not Path(req['controller_root']).is_absolute(): raise ValueError('Invalid recorded controller root')
    if io.read_json(run/'COMMANDS.json')!=command_plan(run,spec,req['rscript'],req['controller_root']):
        raise ValueError('Unexpected fit commands')
    before = check_environment_log(run/'environment_before.stdout.log',spec['kind'])
    if before!=check_environment_log(run/'environment_after.stdout.log',spec['kind']): raise ValueError('Environment changed')
    return check_output(run,bundle,spec,converted=spec['kind']=='story')


def batch(bundles, library, rscript, output, scratch):
    bundles = Path(bundles).resolve(strict=True); specs = inv.verify_local()['branches']
    for spec in specs: study.verify_bundle(bundles/(spec['case']+'_inputs_20261010'))
    protected_bundles = tuple(bundles/(s['case']+'_inputs_20261010') for s in specs)
    root = io.reserve(output,(*protected_bundles,Path(library).resolve().parent,rscript,inv.SNAPSHOT))
    try:
        rows = []
        for spec in specs:
            case = spec['case']; bundle = bundles/(case+'_inputs_20261010')
            print('START '+case,flush=True)
            run = fit(bundle,library,rscript,root/case,scratch)
            result = verify_run(run,bundle)
            rows.append({'case':case,'bundle':str(bundle),'run':case,'manifest':io.record(run/'RUN_MANIFEST.json'),'result':result})
            print('PASS '+case,flush=True)
        io.create_json(root/'BATCH_MANIFEST.json',{'status':'complete','cases':rows,'history_used_by_fit':False})
        io.create_json(root/'COMPLETE.json',{'status':'complete','manifest_sha256':io.sha(root/'BATCH_MANIFEST.json')})
    except BaseException as error: io.failure(root,error); raise
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=('fit','batch','verify-run'))
    for n in ('bundle','bundles','library','rscript','output','run','scratch-parent'): p.add_argument('--'+n,type=Path)
    a = p.parse_args()
    if a.command=='verify-run':
        if not all((a.run,a.bundle)): p.error('verify-run requires run and bundle')
        print(json.dumps(verify_run(a.run,a.bundle))); return
    if not all((a.library,a.rscript,a.output,a.scratch_parent)): p.error('fit/batch require library, rscript, output, scratch-parent')
    if a.command=='batch':
        if a.bundles is None: p.error('batch requires bundles')
        print(batch(a.bundles,a.library,a.rscript,a.output,a.scratch_parent))
    else:
        if a.bundle is None: p.error('fit requires bundle')
        print(fit(a.bundle,a.library,a.rscript,a.output,a.scratch_parent))


if __name__=='__main__': main()
