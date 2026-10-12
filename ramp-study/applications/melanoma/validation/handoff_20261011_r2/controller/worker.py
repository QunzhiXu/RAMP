"""Run one unchanged historical numerical entry point inside a guarded mirror."""
import argparse
import importlib.util
from importlib.metadata import version
import os
from pathlib import Path
import platform
import runpy
import sys
import time
from common import read, write, sha, codes, THREADS, VERSIONS, EXP, record, write_guard, stage_root
from specs import action, CODES, INPUTS, STAGES


def probe():
    import numpy, pandas, scipy, sklearn, torch
    from threadpoolctl import threadpool_info
    versions = {key: version(key) for key in VERSIONS}
    if versions != VERSIONS or platform.python_version() != '3.14.1':
        raise ValueError(f'Runtime versions differ: {versions}')
    threads = {key: os.environ.get(key) for key in THREADS}
    if set(threads.values()) != {'1'}:
        raise ValueError('Thread environment differs')
    pools = [{k: p[k] for k in ('user_api', 'internal_api', 'num_threads', 'version')}
             for p in threadpool_info()]
    if any(p['num_threads'] != 1 for p in pools):
        raise ValueError(f'Thread pools differ: {pools}')
    return {'python': platform.python_version(), 'versions': versions, 'threads': threads, 'pools': pools}


def execute(run, stage):
    run = Path(run).resolve()
    if stage not in STAGES or (run / 'FAILED.json').exists() or (run / 'COMPLETE.json').exists():
        raise ValueError('Unknown or sealed run')
    request = read(run / 'REQUEST.json')
    if request['code_records'] != codes():
        raise ValueError('Controller changed')
    lock = read(run / 'LOCK.json')
    if sha(run / 'LOCK.json') != request['lock_sha256']:
        raise ValueError('Run lock changed')
    status = stage_root(run) / stage
    if read(status / 'REQUEST.json') != {'stage': stage, 'code_records': codes(), 'lock_sha256': request['lock_sha256']}:
        raise ValueError('Stage request differs')
    position = STAGES.index(stage)
    if position and read(stage_root(run) / STAGES[position - 1] / 'COMPLETE.json')['status'] != 'complete':
        raise ValueError('Previous stage incomplete')
    write(status / 'STARTED.json', {'stage': stage, 'pid': os.getpid()})
    immutable = [run / 'tree' / rel for rel in CODES + INPUTS]
    write_guard(run, immutable)
    before = probe(); write(status / 'environment_before.json', before)
    started = time.monotonic()
    import observed
    if stage in ('observed_inputs', 'observed_survival', 'fixed_roi'):
        getattr(observed, stage)(run)
    else:
        rel, function, arguments = action(stage)
        script = run / 'tree' / rel
        sys.path.insert(0, str(script.parent))
        sys.argv = [str(script)]
        if function == '__main__':
            runpy.run_path(str(script), run_name='__main__')
        else:
            module = observed.load(script)
            getattr(module, function)(*arguments)
    after = probe(); write(status / 'environment_after.json', after)
    # Libraries imported by a scientific entry point may add pools. Existing pools must
    # not change; every final pool is independently checked to be single-threaded.
    if before['versions'] != after['versions'] or before['threads'] != after['threads']:
        raise ValueError('Runtime changed')
    if any(pool not in after['pools'] for pool in before['pools']):
        raise ValueError('Existing runtime pool changed')
    for rel in CODES + INPUTS:
        p = run / 'tree' / rel
        actual = record(p.resolve())
        if actual != {key: lock['files'][rel][key] for key in ('bytes', 'sha256')}:
            raise ValueError(f'Input/code changed: {rel}')
    if codes() != request['code_records']:
        raise ValueError('Controller changed during computation')
    write(status / 'WORKER_DONE.json', {'stage': stage, 'elapsed_seconds': time.monotonic() - started,
                                      'code_records': codes(), 'input_hashes_verified': CODES + INPUTS})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', required=True); p.add_argument('--stage', choices=STAGES, required=True)
    a = p.parse_args(); execute(a.run, a.stage)
