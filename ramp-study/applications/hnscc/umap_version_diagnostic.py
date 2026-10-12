"""One authorized 0.5.11 UMAP rerun; no changes to the accepted nine-stage run."""
import argparse
import gzip
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import runpy
import shutil
import subprocess
import sys
import time
from common import (HERE, STUDY, THREADS, read, write, sha, record, files, links,
                    reserve, check_resources, environment, write_guard)

SCRIPT = 'figure_HNSCC/alignment/03_plot_umap.py'
COORD = 'full_94488_umap_coordinates.csv.gz'
OUTPUTS = {COORD, 'display_cell_counts.csv', 'umap_manifest.json',
           'hnscc_pt74_ramp_ver2_full_94488_umap.png', 'hnscc_pt74_ramp_ver2_full_94488_umap.pdf'}
LOCK_SHA = 'b07d7afa1569dd627912ed447446cc36f5fb5aa9463a502391bdbe5321d989f3'
PRIOR_MANIFEST_SHA = '19bbf86f84306b14d3c5b7b3e97a67e988e3d10cf06a0ebbd41199b22c55a5ee'
PRIOR_REPORT_SHA = 'f23b16fb40aa73dacfff778f57f315cea897e568dac5715ad1f9ef0d58b229c2'
VERSIONS = {'numpy': '2.4.0', 'pandas': '2.3.3', 'scipy': '1.16.3',
            'scikit-learn': '1.8.0', 'umap-learn': '0.5.11', 'numba': '0.64.0',
            'pynndescent': '0.6.0', 'llvmlite': '0.46.0', 'matplotlib': '3.10.8',
            'threadpoolctl': '3.6.0'}
PARAMETERS = {'n_neighbors': 30, 'min_dist': 0.25, 'metric': 'euclidean', 'random_state': 20260824}


def controller():
    return {name: record(HERE / name) for name in ('umap_version_diagnostic.py', 'common.py')}


def bound(path):
    path = Path(path).resolve()
    return {'path': str(path), **record(path)}


def check_bindings(bindings):
    for name, item in bindings.items():
        if record(item['path']) != {k: item[k] for k in ('bytes', 'sha256')}:
            raise ValueError('Bound source changed: ' + name)


def probe(expected):
    import numpy, scipy, sklearn
    from threadpoolctl import threadpool_info
    versions = {name: metadata.version(name) for name in VERSIONS}
    required = {**VERSIONS, 'umap-learn': expected}
    if versions != required or platform.python_version() != '3.14.1':
        raise ValueError('Diagnostic dependency versions differ')
    threads = {k: os.environ.get(k) for k in THREADS}
    pools = [{k: p[k] for k in ('user_api', 'internal_api', 'num_threads', 'version')}
             for p in threadpool_info()]
    if threads != {k: '1' for k in THREADS} or not pools or any(p['num_threads'] != 1 for p in pools):
        raise ValueError('Diagnostic thread policy differs')
    modules = {'umap-learn': 'umap/umap_.py', 'numba': 'numba/__init__.py',
               'pynndescent': 'pynndescent/pynndescent_.py', 'llvmlite': 'llvmlite/__init__.py',
               'numpy': 'numpy/__init__.py', 'pandas': 'pandas/__init__.py',
               'scipy': 'scipy/__init__.py', 'scikit-learn': 'sklearn/preprocessing/_data.py'}
    return {'python': platform.python_version(), 'executable': sys.executable,
            'versions': versions, 'threads': threads, 'pools': pools,
            'implementation_files': {k: bound(metadata.distribution(k).locate_file(v)) for k,v in modules.items()}}


def comparison(new, old):
    import numpy as np
    import pandas as pd
    a = pd.read_csv(new, low_memory=False, float_precision='round_trip')
    b = pd.read_csv(old, low_memory=False, float_precision='round_trip')
    coord = ['UMAP1', 'UMAP2']
    if list(a.columns) != list(b.columns) or len(a) != 94488 or len(b) != 94488:
        raise ValueError('Coordinate schema or cohort differs')
    if a.cell_id.duplicated().any() or not a.drop(columns=coord).equals(b.drop(columns=coord)):
        raise ValueError('Coordinate identity, metadata or row order differs')
    if not np.isfinite(a[coord].to_numpy()).all() or not np.isfinite(b[coord].to_numpy()).all():
        raise ValueError('Nonfinite coordinates')
    columns = {c: {'exact': bool(np.array_equal(a[c].to_numpy(), b[c].to_numpy())),
                   'changed_values': int(np.count_nonzero(a[c].to_numpy() != b[c].to_numpy())),
                   'max_abs_difference': float(np.max(np.abs(a[c].to_numpy() - b[c].to_numpy())))} for c in coord}
    return {'cells': len(a), 'metadata_and_row_order_exact': True, 'columns': columns,
            'coordinates_exact': all(v['exact'] for v in columns.values()),
            'gzip_decompressed_equal': gzip.open(new, 'rb').read() == gzip.open(old, 'rb').read(),
            'byte_identical': sha(new) == sha(old)}


def worker(run):
    run = Path(run).resolve()
    if (run/'COMPLETE.json').exists() or (run/'FAILED.json').exists():
        raise ValueError('Sealed diagnostic')
    req = read(run/'REQUEST.json')
    if req['controller'] != controller():
        raise ValueError('Diagnostic controller changed')
    tree = run/'tree/figure_HNSCC/alignment'
    immutable = [run/'controller'/p for p in req['controller']] + [tree/'03_plot_umap.py']
    write_guard(run, immutable)
    write(run/'FIT_STARTED.json', {'pid': os.getpid(), 'time_unix': time.time(),
                                'empty_output': not (tree/'umap').exists(),
                                'no_historical_coordinates_in_fit_tree': True})
    if (tree/'umap').exists() or (tree/'.cache').exists():
        raise ValueError('Diagnostic fit output or cache already exists')
    before = probe('0.5.11'); write(run/'environment_before.json', before)
    started = time.monotonic()
    # Execute the complete, byte-identical historical script, including its plotting.
    # The fit sees only the accepted matrix and input metadata, never old coordinates.
    runpy.run_path(str(tree/'03_plot_umap.py'), run_name='__main__')
    after = probe('0.5.11'); write(run/'environment_after.json', after)
    if before != after:
        raise ValueError('Diagnostic environment changed')
    write(run/'FIT_DONE.json', {'elapsed_seconds': time.monotonic()-started,
                             'controller': controller(), 'fit_calls': 1})


def check_payload(run):
    req = read(run/'REQUEST.json')
    if req['controller'] != controller() or req['parameters'] != PARAMETERS:
        raise ValueError('Unrecognized diagnostic controller or parameters')
    check_bindings(req['bindings'])
    binding = req['bindings']
    if binding['lock']['sha256'] != LOCK_SHA or binding['prior_manifest']['sha256'] != PRIOR_MANIFEST_SHA or binding['prior_comparison']['sha256'] != PRIOR_REPORT_SHA:
        raise ValueError('Unknown accepted-run anchors')
    expected_links = {f'tree/figure_HNSCC/alignment/{rel}': binding[key]['path'] for key,rel in [
        ('matrix','output/aligned_ramp_ver2.float32.npy'), ('metadata','input/cell_metadata.csv.gz')]}
    if links(run) != expected_links:
        raise ValueError('Diagnostic input link grid differs')
    for rel,path in expected_links.items():
        if record(Path(path)) != {k: binding['matrix' if '/output/' in rel else 'metadata'][k] for k in ('bytes','sha256')}:
            raise ValueError('Input changed')
    if record(run/'tree'/SCRIPT) != {k: binding['scientific_script'][k] for k in ('bytes','sha256')}:
        raise ValueError('Scientific script differs')
    if files(run/'controller') != req['controller']:
        raise ValueError('Archived controller differs')
    expected_command = [req['python'], '-B', str(Path(req['original_run'])/'controller/umap_version_diagnostic.py'), 'worker', '--run', req['original_run']]
    if read(run/'COMMAND.json') != expected_command:
        raise ValueError('Diagnostic command differs')
    before = read(run/'environment_before.json'); after = read(run/'environment_after.json')
    control = read(run/'environment_0512_current_probe.json')
    for env,version in ((before,'0.5.11'),(after,'0.5.11'),(control,'0.5.12')):
        if env['versions'] != {**VERSIONS,'umap-learn':version} or env['python'] != '3.14.1' or env['threads'] != {k:'1' for k in THREADS}:
            raise ValueError('Runtime record differs')
        if not env['pools'] or any(p['num_threads'] != 1 for p in env['pools']):
            raise ValueError('Runtime pools differ')
    if before != after or before['executable'] != req['python']:
        raise ValueError('Runtime changed')
    for k,v in before['implementation_files'].items():
        if k != 'umap-learn' and v != control['implementation_files'][k]:
            raise ValueError('Additional implementation difference: '+k)
    started = read(run/'FIT_STARTED.json'); done = read(run/'FIT_DONE.json')
    if started['empty_output'] is not True or started['no_historical_coordinates_in_fit_tree'] is not True or done['fit_calls'] != 1 or done['controller'] != req['controller']:
        raise ValueError('Fit record differs')
    if type(done['elapsed_seconds']) not in (int,float) or not 0 < done['elapsed_seconds'] < float('inf'):
        raise ValueError('Invalid fit duration')
    out = run/'tree/figure_HNSCC/alignment/umap'
    if {p.name for p in out.iterdir()} != OUTPUTS:
        raise ValueError('UMAP output grid differs')
    manifest = read(out/'umap_manifest.json')
    if manifest['umap'] != PARAMETERS or manifest['cells'] != 94488 or manifest['labels_used_for_embedding'] is not False:
        raise ValueError('Scientific parameter record differs')
    hist_manifest = read(binding['historical_manifest']['path'])
    if {k:v for k,v in manifest.items() if k!='outputs'} != {k:v for k,v in hist_manifest.items() if k!='outputs'}:
        raise ValueError('UMAP historical manifest nonpath fields differ')
    if sha(out/'display_cell_counts.csv') != binding['historical_counts']['sha256']:
        raise ValueError('Display cell counts differ')
    history = comparison(out/COORD, binding['historical_coordinates']['path'])
    previous = comparison(out/COORD, binding['prior_coordinates']['path'])
    for rel,rec in read(run/'PRIOR_FILES.json').items():
        if record(Path(req['module_root'])/rel) != rec:
            raise ValueError('Existing module file changed: '+rel)
    return {'status': 'exact_coordinates' if history['coordinates_exact'] else 'completed_with_differences',
            'historical_comparison': history, 'previous_0512_comparison': previous,
            'counts_byte_identical': True, 'manifest_equal_except_output_paths': True,
            'prior_module_files_unchanged': len(read(run/'PRIOR_FILES.json')),
            'fit_elapsed_seconds': done['elapsed_seconds'], 'fit_calls': 1,
            'comparison_scope': 'coordinates, full metadata, counts and manifest; preview images not visually audited',
            'historical_environment_unrecorded': True,
            'only_version_difference_among_ten_recorded_distributions': 'umap-learn'}


def verify(run):
    run = Path(run).resolve()
    if (run/'FAILED.json').exists() or not (run/'COMPLETE.json').is_file():
        raise ValueError('Diagnostic incomplete or failed')
    if read(run/'COMPLETE.json') != {'status':'complete','manifest_sha256':sha(run/'MANIFEST.json')}:
        raise ValueError('Diagnostic seal differs')
    actual = files(run); actual.pop('MANIFEST.json'); actual.pop('COMPLETE.json')
    if read(run/'MANIFEST.json') != {'files': actual, 'links': links(run)}:
        raise ValueError('Diagnostic payload differs')
    result = check_payload(run)
    if result != read(run/'report.json'):
        raise ValueError('Comparison report differs from actual outputs')
    return result


def fit(lockdir, previous, python, output):
    from prepare import verify as verify_lock
    from study import verify_run
    from baseline import check as check_prior
    lockdir, previous = Path(lockdir).resolve(), Path(previous).resolve()
    # Do not resolve the interpreter symlink: its venv selection is essential.
    python = str(Path(python).absolute())
    lock = verify_lock(lockdir, source=True)
    verify_run(lockdir, previous)
    source = Path(lock['source_root'])
    paths = {'lock':lockdir/'LOCK.json', 'scientific_script':lockdir/'code'/SCRIPT,
             'original_script':source/SCRIPT,
             'matrix':Path(lock['input_paths']['figure_HNSCC/alignment/output/aligned_ramp_ver2.float32.npy']),
             'metadata':Path(lock['input_paths']['figure_HNSCC/alignment/input/cell_metadata.csv.gz']),
             'historical_coordinates':source/'figure_HNSCC/alignment/umap'/COORD,
             'historical_counts':source/'figure_HNSCC/alignment/umap/display_cell_counts.csv',
             'historical_manifest':source/'figure_HNSCC/alignment/umap/umap_manifest.json',
             'prior_manifest':previous/'RUN_MANIFEST.json', 'prior_complete':previous/'COMPLETE.json',
             'prior_coordinates':previous/'tree/figure_HNSCC/alignment/umap'/COORD,
             'prior_comparison':HERE/'validation/comparison_20261011/report.json',
             'prior_handoff':HERE/'validation/handoff_20261011/report.json',
             'python_binary':Path(python).resolve(), 'venv_configuration':Path(python).parent.parent/'pyvenv.cfg'}
    bindings = {k:bound(p) for k,p in paths.items()}
    if bindings['prior_manifest']['sha256'] != PRIOR_MANIFEST_SHA or bindings['prior_comparison']['sha256'] != PRIOR_REPORT_SHA:
        raise ValueError('Unknown prior evidence')
    prior = files(HERE)
    resources = check_resources()
    run = reserve(output, [source, lockdir, previous, Path(bindings['matrix']['path']).parent, HERE/'tests', STUDY.parent/'ramp-proteomics'])
    try:
        write(run/'PRIOR_FILES.json', prior)
        write(run/'REQUEST.json', {'controller':controller(), 'bindings':bindings, 'parameters':PARAMETERS,
              'python':python, 'original_run':str(run), 'module_root':str(HERE), 'resources_before':resources,
              'scope':'one additional 0.5.11 fit of unchanged historical script; no parameter or seed search'})
        (run/'controller').mkdir()
        for name in controller(): shutil.copyfile(HERE/name,run/'controller'/name)
        tree = run/'tree/figure_HNSCC/alignment'
        for name in ('input','output'): (tree/name).mkdir(parents=True)
        shutil.copyfile(paths['scientific_script'], tree/'03_plot_umap.py')
        (tree/'input/cell_metadata.csv.gz').symlink_to(bindings['metadata']['path'])
        (tree/'output/aligned_ramp_ver2.float32.npy').symlink_to(bindings['matrix']['path'])
        env = environment(run)
        for key in ('TMPDIR','MPLCONFIGDIR','NUMBA_CACHE_DIR','XDG_CACHE_HOME','JOBLIB_TEMP_FOLDER'):
            Path(env[key]).mkdir(parents=True,exist_ok=True)
        probe_cmd = [sys.executable,'-B',str(run/'controller/umap_version_diagnostic.py'),'probe']
        write(run/'PROBE_COMMAND.json',probe_cmd)
        proc = subprocess.run(probe_cmd,env=env,cwd=run,text=True,capture_output=True)
        with (run/'probe.stderr').open('x') as stream: stream.write(proc.stderr)
        if proc.returncode: raise RuntimeError('0.5.12 control probe failed')
        write(run/'environment_0512_current_probe.json',json.loads(proc.stdout))
        command = [python,'-B',str(run/'controller/umap_version_diagnostic.py'),'worker','--run',str(run)]
        write(run/'COMMAND.json',command)
        print('ONE_0511_FIT_START',flush=True)
        with (run/'stdout.log').open('x') as stdout, (run/'stderr.log').open('x') as stderr:
            proc = subprocess.run(command,env=env,cwd=run,stdout=stdout,stderr=stderr)
        if proc.returncode: raise RuntimeError('0.5.11 diagnostic failed; retained logs')
        print('ONE_0511_FIT_FINISHED',flush=True)
        verify_lock(lockdir,source=True); verify_run(lockdir,previous)
        write(run/'external_baseline.json',check_prior(HERE/'validation/preflight_20261011'))
        write(run/'resources_after.json',check_resources())
        result = check_payload(run)
        write(run/'report.json', result)
        write(run/'MANIFEST.json', {'files':files(run),'links':links(run)})
        write(run/'COMPLETE.json', {'status':'complete','manifest_sha256':sha(run/'MANIFEST.json')})
        return result
    except BaseException as exc:
        write(run/'FAILED.json',{'error':repr(exc)}); raise


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['fit','worker','probe','verify'])
    p.add_argument('--run'); p.add_argument('--out'); p.add_argument('--lockdir'); p.add_argument('--previous-run'); p.add_argument('--python')
    a=p.parse_args()
    if a.mode=='probe': result=probe('0.5.12')
    elif a.mode=='worker': result=worker(a.run)
    elif a.mode=='verify': result=verify(a.run)
    else: result=fit(a.lockdir,a.previous_run,a.python,a.out)
    print(json.dumps(result,indent=2))
