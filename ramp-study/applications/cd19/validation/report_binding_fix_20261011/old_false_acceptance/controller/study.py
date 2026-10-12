"""Sequential, bounded CD19 batch; references are never placed in the fit tree."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from common import (HERE, EXP, codes, read, write, record, sha, reserve, check_resources,
                    environment, files, links, stage_root)
from specs import CODES, INPUTS, STAGES, action
from prepare import verify as verify_lock


def mirror(run, lockdir, lock):
    for rel in CODES:
        destination = run / 'tree' / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as out, (lockdir / 'code' / rel).open('rb') as inp:
            shutil.copyfileobj(inp, out)
    for rel in INPUTS:
        destination = run / 'tree' / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(Path(lock['source_root']) / rel)
    for rel in ('tmp', 'cache/matplotlib', 'cache/numba', 'cache/xdg', 'cache/joblib',
                'real_spatial0913/results'):
        (run / 'tree' / EXP / rel).mkdir(parents=True, exist_ok=True)


def seal_stage(run, stage):
    folder = stage_root(run) / stage
    done = read(folder / 'WORKER_DONE.json')
    if done['stage'] != stage or done['code_records'] != codes() or done['input_hashes_verified'] != CODES + INPUTS:
        raise ValueError('Invalid worker completion')
    write(folder / 'MANIFEST.json', files(folder))
    write(folder / 'COMPLETE.json', {'status': 'complete', 'manifest_sha256': sha(folder / 'MANIFEST.json')})


def fit(lockdir, output):
    lockdir = Path(lockdir).resolve()
    lock = verify_lock(lockdir, source=True)
    resources = check_resources()
    run = reserve(output, [Path(lock['source_root']), lockdir, HERE / 'tests'])
    try:
        write(run / 'REQUEST.json', {'stages': STAGES, 'lock_sha256': sha(lockdir / 'LOCK.json'),
              'code_records': codes(), 'resources': resources, 'max_study_bytes': 60_000_000_000,
              'min_free_bytes': 20_000_000_000, 'estimated_new_bytes_under': 1_000_000_000,
              'original_run': str(run), 'python': sys.executable, 'worker': str(HERE / 'worker.py'),
              'source_inputs': 'read-only file symlinks; historical outputs are not execution inputs'})
        with (run / 'LOCK.json').open('xb') as out:
            out.write((lockdir / 'LOCK.json').read_bytes())
        mirror(run, lockdir, lock)
        stage_root(run).mkdir(parents=True)
        for stage in STAGES:
            state = check_resources()
            folder = stage_root(run) / stage
            folder.mkdir(exist_ok=False)
            write(folder / 'REQUEST.json', {'stage': stage, 'code_records': codes(), 'lock_sha256': sha(run / 'LOCK.json')})
            command = [sys.executable, '-B', str(HERE / 'worker.py'), '--run', str(run), '--stage', stage]
            write(folder / 'COMMAND.json', command)
            print(f'STAGE_START {stage}', flush=True)
            with (folder / 'stdout.log').open('x') as stdout, (folder / 'stderr.log').open('x') as stderr:
                process = subprocess.Popen(command, env=environment(run), stdout=stdout, stderr=stderr, cwd=run)
                try:
                    while True:
                        try:
                            result = process.wait(timeout=20)
                            break
                        except subprocess.TimeoutExpired:
                            check_resources()
                    if result:
                        raise RuntimeError(f'Stage {stage} failed: {result}; inspect retained stderr')
                except BaseException:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill(); process.wait()
                    raise
            seal_stage(run, stage)
            print(f'STAGE_COMPLETE {stage}', flush=True)
        verify_lock(lockdir, source=True, expected_sha=read(run / 'REQUEST.json')['lock_sha256'])
        if codes() != read(run / 'REQUEST.json')['code_records']:
            raise ValueError('Controller changed')
        write(run / 'RESOURCES_AFTER.json', check_resources())
        write(run / 'RUN_MANIFEST.json', {'files': files(run), 'links': links(run)})
        verify_run(lockdir, run, require_complete=False)
        write(run / 'COMPLETE.json', {'status': 'complete', 'run_manifest_sha256': sha(run / 'RUN_MANIFEST.json')})
        return {'status': 'complete', 'stages': len(STAGES), 'run': str(run)}
    except BaseException as exc:
        write(run / 'FAILED.json', {'error': repr(exc)})
        raise


def verify_run(lockdir, run, require_complete=True):
    run = Path(run).resolve()
    if (run / 'FAILED.json').exists():
        raise ValueError('Failed run')
    request = read(run / 'REQUEST.json')
    lock = verify_lock(lockdir, expected_sha=request['lock_sha256'])
    if sha(run / 'LOCK.json') != request['lock_sha256'] or request['code_records'] != codes() or request['stages'] != STAGES:
        raise ValueError('Run binding differs')
    if require_complete and read(run / 'COMPLETE.json') != {'status': 'complete', 'run_manifest_sha256': sha(run / 'RUN_MANIFEST.json')}:
        raise ValueError('Run incomplete')
    expected_links = {'tree/' + rel: str(Path(lock['source_root']) / rel) for rel in INPUTS}
    manifest = read(run / 'RUN_MANIFEST.json')
    if links(run) != expected_links or manifest['links'] != expected_links:
        raise ValueError('Input link grid differs')
    actual = files(run)
    actual.pop('RUN_MANIFEST.json'); actual.pop('COMPLETE.json', None)
    if actual != manifest['files']:
        raise ValueError('Run payload differs')
    if {p.name for p in stage_root(run).iterdir()} != set(STAGES):
        raise ValueError('Stage grid differs')
    for rel in CODES + INPUTS:
        if record((run / 'tree' / rel).resolve()) != {k: lock['files'][rel][k] for k in ('bytes', 'sha256')}:
            raise ValueError(f'Runtime input/code changed: {rel}')
    for stage in STAGES:
        folder = stage_root(run) / stage
        if read(folder / 'REQUEST.json') != {'stage': stage, 'code_records': codes(), 'lock_sha256': request['lock_sha256']}:
            raise ValueError('Stage request differs')
        if read(folder / 'COMPLETE.json') != {'status': 'complete', 'manifest_sha256': sha(folder / 'MANIFEST.json')}:
            raise ValueError('Stage incomplete')
        current = files(folder); current.pop('MANIFEST.json'); current.pop('COMPLETE.json')
        if current != read(folder / 'MANIFEST.json'):
            raise ValueError('Stage payload differs')
        done = read(folder / 'WORKER_DONE.json')
        if done['stage'] != stage or done['code_records'] != codes() or done['input_hashes_verified'] != CODES + INPUTS:
            raise ValueError('Worker binding differs')
        expected_command = [request['python'], '-B', request['worker'], '--run', request['original_run'], '--stage', stage]
        if read(folder / 'COMMAND.json') != expected_command or Path(request['worker']).name != 'worker.py':
            raise ValueError('Command differs')
        if not isinstance(done['elapsed_seconds'], (int, float)) or not 0 <= done['elapsed_seconds'] < float('inf'):
            raise ValueError('Invalid stage duration')
        from common import THREADS, VERSIONS
        before = read(folder / 'environment_before.json'); after = read(folder / 'environment_after.json')
        for env in (before, after):
            if env['versions'] != VERSIONS or env['python'] != '3.14.1' or env['threads'] != {k: '1' for k in THREADS}:
                raise ValueError('Environment record differs')
            if not env['pools'] or any(p['num_threads'] != 1 for p in env['pools']):
                raise ValueError('Thread pool record differs')
        if any(p not in after['pools'] for p in before['pools']):
            raise ValueError('Runtime pool changed')
    return {'status': 'verified', 'stages': len(STAGES), 'runtime_code_files': len(CODES), 'input_files': len(INPUTS)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['fit', 'verify-run'])
    parser.add_argument('--lockdir', required=True); parser.add_argument('--out'); parser.add_argument('--run')
    a = parser.parse_args()
    print(json.dumps(fit(a.lockdir, a.out) if a.mode == 'fit' else verify_run(a.lockdir, a.run), indent=2))
