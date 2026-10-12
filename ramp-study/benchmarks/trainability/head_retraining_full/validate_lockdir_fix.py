"""Create-only evidence for the external-lock portability fix; no fitting or data copies."""
import argparse
import ast
import hashlib
import os
import re
import subprocess
import sys
import tempfile
sys.dont_write_bytecode = True
from full_common import *

PREFLIGHT = HERE/'validation/lockdir_fix_preflight_20261011'
EVIDENCE = HERE/'validation/lockdir_fix_20261011'
REPORTS = ('validation/interruption_20261011/report.json',
           'validation/continuation_preflight_20261011/report.json')


def result_metadata():
    """A read-only mutation sentinel, not a substitute for payload SHA-256 verification."""
    sha = hashlib.sha256(); count = total = 0
    for directory, dirs, names in os.walk(HERE/'runs'):
        dirs.sort()
        for name in sorted(names):
            p = Path(directory)/name; s = p.lstat()
            row = [p.relative_to(HERE).as_posix(), s.st_mode, s.st_size,
                   s.st_mtime_ns, s.st_ctime_ns, s.st_ino]
            sha.update(json.dumps(row, separators=(',', ':')).encode()+b'\n')
            count += 1; total += s.st_size
    return dict(files=count, bytes=total, metadata_sha256=sha.hexdigest())


def existing_records():
    records = {}
    for directory, dirs, names in os.walk(HERE):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in ('runs', '__pycache__')
                         and Path(directory)/d not in (PREFLIGHT, EVIDENCE))
        for name in sorted(names):
            if name == '.DS_Store' or name.endswith('.pyc'): continue
            p = Path(directory)/name
            records[p.relative_to(HERE).as_posix()] = record(p)
    return records


def capture():
    require(not PREFLIGHT.exists() and not EVIDENCE.exists(), 'Fix evidence already exists')
    delivery = read(HERE/'DELIVERY_RECORD_20261011.json')
    for name, expected in delivery['module_python_and_tests'].items():
        require(record(HERE/name) == expected, 'Pre-fix delivery code changed: '+name)
    for name, expected in delivery['docs'].items():
        require(record(HERE/name) == expected, 'Pre-fix delivery document changed: '+name)
    records = existing_records(); state = result_metadata()
    out = reserve(PREFLIGHT)
    copy(HERE/'combined.py', out/'controller/combined.py', records['combined.py'])
    write(out/'report.json', dict(status='captured_before_fix', records=records, results=state))
    seal(out, dict(scope='external_lockdir_fix_preflight'))
    print('PREFLIGHT', len(records), state, flush=True)


def invoke(command, path, success=True, contains=None):
    result = subprocess.run(command, env=child_environment(), capture_output=True, text=True)
    write(path, dict(command=command, returncode=result.returncode,
                     stdout=result.stdout, stderr=result.stderr))
    require((result.returncode == 0) == success, 'Unexpected CLI outcome: '+path.name)
    if contains: require(contains in result.stderr, 'Unexpected rejection reason: '+path.name)
    return result


def copy_controller(destination, old=False):
    destination.mkdir(parents=True)
    for p in HERE.glob('*.py'):
        source = PREFLIGHT/'controller/combined.py' if old and p.name == 'combined.py' else p
        copy(source, destination/p.name, record(source))
    for p in (HERE/'vendor/pilot').glob('*.py'):
        copy(p, destination/'vendor/pilot'/p.name, record(p))
    for n in REPORTS: copy(HERE/n, destination/n, record(HERE/n))
    require(not (destination/'locks').exists(), 'Relocated controller must have no default locks')


def validate():
    verify_seal(PREFLIGHT); before = read(PREFLIGHT/'report.json')
    # Prove the production diff is exactly argument plumbing, not verifier relaxation.
    old = (PREFLIGHT/'controller/combined.py').read_text()
    expected = old.replace('def expected_routes(root,original_run,continuation):',
                           'def expected_routes(lockdir,root,original_run,continuation):')
    expected = expected.replace("resume_run.binding(HERE/'locks/full_20261011',continuation)",
                                'resume_run.binding(lockdir,continuation)')
    expected = expected.replace('expected_routes(root,original_run,continuation)',
                                'expected_routes(lockdir,root,original_run,continuation)')
    require((HERE/'combined.py').read_text() == expected, 'Unexpected production changes')
    for name, item in before['records'].items():
        if name != 'combined.py': require(record(HERE/name) == item, 'Old file changed: '+name)
    require(result_metadata() == before['results'], 'Existing result metadata changed')
    start = existing_records(); out = reserve(EVIDENCE)
    lockdir = HERE/'locks/full_20261011'
    source = STUDY.parent/'final_xu'
    original = HERE/'runs/full_20261011'; continuation = HERE/'runs/continuation_20261011'
    run = HERE/'validation/combined_full_20261011'
    try:
        root = load_lock(lockdir)
        with tempfile.TemporaryDirectory(prefix='head_lockdir_fix_') as t:
            temp = Path(t).resolve()
            # Separate old/new controllers. Neither receives any default lock directory.
            oldmod = temp/'old controller'; newmod = temp/'new controller'
            copy_controller(oldmod, old=True); copy_controller(newmod)
            flags = ['--lockdir', str(lockdir), '--source-root', str(source),
                     '--original-run', str(original), '--continuation', str(continuation), '--run', str(run)]
            for mode in ('check-binding', 'verify-run'):
                invoke([sys.executable, '-B', str(oldmod/'combined.py'), mode, *flags],
                       out/('old_'+mode+'.json'), False, str(oldmod/'locks/full_20261011/LOCK.json'))
            invoke([sys.executable, '-B', str(newmod/'combined.py'), 'check-binding', *flags],
                   out/'new_check_binding.json')
            print('OLD_FAILURE_REPRODUCED_NEW_BINDING_PASSED', flush=True)
            # All 860 groups are read in place, once. No result/lock/input payload is copied.
            verified = invoke([sys.executable, '-B', str(newmod/'combined.py'), 'verify-run', *flags],
                              out/'new_verify_run.json')
            actual = ast.literal_eval(verified.stdout.strip().splitlines()[-1])
            require(actual == read(run/'verification.json'), 'Structural report changed')
            print('ALL_860_GROUPS_VERIFIED', flush=True)
            for name in ('missing', 'invalid'):
                bad = temp/(name+' external locks')
                if name == 'invalid': bad.mkdir(); write(bad/'LOCK.json', {})
                badflags = flags.copy(); badflags[1] = str(bad)
                for mode in ('check-binding', 'verify-run'):
                    invoke([sys.executable, '-B', str(newmod/'combined.py'), mode, *badflags],
                           out/(name+'_'+mode+'.json'), False,
                           'Full lock changed' if name == 'invalid' else str(bad/'LOCK.json'))
            # Existing tests expect default locks. Only this distinct test tree gets a
            # read-only reference to real locks; the CLI relocation trees above never do.
            suite = temp/'suite/ramp-study/benchmarks/trainability/head_retraining_full'
            copy_controller(suite)
            for p in (HERE/'tests').glob('*.py'): copy(p, suite/'tests'/p.name, record(p))
            (suite/'locks').mkdir(); (suite/'locks/full_20261011').symlink_to(lockdir, target_is_directory=True)
            for name, item in root['scientific'].items(): copy(member(source,name), temp/'suite/final_xu'/name, item)
            tests = invoke([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(suite/'tests'), '-v'], out/'tests.json')
            match = re.search(r'Ran (\d+) tests', tests.stderr); require(match is not None, 'Missing test count')
            copied = [p for p in temp.rglob('*') if p.is_file() and not p.is_symlink()]
            copied_bytes = sum(p.stat().st_size for p in copied)
            require(copied_bytes < 5*1024**2, 'Unexpected temporary copy volume')
            require(not(oldmod/'locks').exists() and not(newmod/'locks').exists(), 'Default locks appeared')
        require(existing_records() == start, 'Code, locks or historical evidence changed during validation')
        require(result_metadata() == before['results'], 'Existing results changed during validation')
        report = dict(status='passed_internal_validation_pending_external_audit', tests=int(match[1]),
            production_change='Only actual lockdir argument propagation in combined.py',
            old_cli_rejected_valid_external_locks=True, new_cli_accepted_without_default_locks=True,
            structural=actual, historical_non_run_files_unchanged=len(before['records'])-1,
            result_metadata_unchanged=before['results'], temporary_regular_file_bytes=copied_bytes,
            results_or_lock_payloads_copied=False, training_rerun=False,
            old_reports_rewritten=False, preflight_report=record(PREFLIGHT/'report.json'),
            current_code={n:record(HERE/n) for n in ('combined.py','tests/test_lockdir.py','validate_lockdir_fix.py')},
            limits='Portability and existing structural verification only; no new numerical comparison, aggregation or scientific claim. Run metadata sentinel is not a content hash; accepted payloads were SHA-256 checked by verify-run.')
        write(out/'report.json', report); seal(out, dict(scope='external_lockdir_fix'))
        print(json.dumps(report), flush=True)
    except BaseException as e:
        write(out/'FAILED.json', dict(error=repr(e))); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['capture','validate']); a = p.parse_args()
    capture() if a.mode == 'capture' else validate()
