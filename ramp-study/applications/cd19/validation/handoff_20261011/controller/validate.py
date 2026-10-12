"""Handoff checks with small temporary code copies; never copy real results."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from common import HERE, STUDY, STORY, EXP, MARKERS, record, sha, read, write, reserve, resource_state
from prepare import verify as verify_lock
from study import verify_run
from baseline import check as check_prior


def cohorts(lockdir, run):
    import numpy as np
    import pandas as pd
    lock = verify_lock(lockdir, source=True); root = Path(lock['source_root'])
    datasets = [('BM CODEX', 'bm_codex', 30000, 12), ('Triana AbSeq', 'triana_abseq', 28031, 3), ('BM CyTOF', 'bm_cytof', 30000, 2)]
    identities = []; result = []
    for dataset, slug, cells, units in datasets:
        frame = pd.read_csv(root / STORY / f'sampled_data/{slug}/source_input_15.csv.gz')
        if len(frame) != cells or frame.unit_key.nunique() != units or not frame.dataset.eq(dataset).all() or not frame.cell_id.is_unique:
            raise ValueError('Cohort identity differs')
        x = frame[MARKERS + ['CD19']].to_numpy(float)
        if not np.isfinite(x).all():
            raise ValueError('Cohort contains nonfinite values')
        identities.append(frame[['cell_id', 'dataset', 'donor', 'unit_key']])
        result.append({'dataset': dataset, 'cells': cells, 'units': units, 'markers_including_target': 15})
    identity = pd.concat(identities, ignore_index=True)
    tree = Path(run) / 'tree'
    from specs import marker_dir
    for marker in MARKERS:
        folder = tree / marker_dir(marker)
        md = pd.read_csv(folder / 'codex_prediction_metadata.csv.gz')
        if not md.equals(identities[0]):
            raise ValueError('Holdout prediction identity differs')
        with np.load(folder / f'{marker}_prediction.npz') as z:
            fit = z['fit_rows']
            if not np.array_equal(z['codex_rows'], np.arange(30000)) or len(fit) != 12000 or len(np.unique(fit)) != 12000:
                raise ValueError('Prediction/fit row grid differs')
            if identity.iloc[fit].groupby('dataset').size().to_dict() != {d: 4000 for d, _, _, _ in datasets}:
                raise ValueError('Fit platform count differs')
            for key in ('prediction', 'triana_component', 'cytof_component'):
                if z[key].shape != (30000,) or not np.isfinite(z[key]).all():
                    raise ValueError('Invalid prediction')
            if not np.array_equal(z['prediction'], .5 * (z['triana_component'] + z['cytof_component'])):
                raise ValueError('Source combination differs')
    return {'cohorts': result, 'total_cells': len(identity), 'total_units': identity.unit_key.nunique(), 'holdout_markers': len(MARKERS)}


def original_indexes():
    workspace = STUDY.parent
    answer = {}
    for name in ('source_census.json', 'curated_sources.json'):
        index = read(STUDY / 'inventory_20261010/evidence' / name)
        records = index['files']; checked = 0
        for item in records:
            if item.get('sha256'):
                if sha(workspace / item['path']) != item['sha256']:
                    raise ValueError(f'Original indexed source changed: {item["path"]}')
                checked += 1
        answer[name] = checked
    return answer


def validate(lockdir, run, comparison, figure_metrics, output):
    lockdir = Path(lockdir).resolve(); run = Path(run).resolve()
    output = reserve(output, [Path(read(lockdir / 'LOCK.json')['source_root']), lockdir, run, Path(comparison), Path(figure_metrics)])
    try:
        structure = verify_run(lockdir, run)
        (output / 'controller').mkdir()
        for path in HERE.glob('*.py'):
            shutil.copyfile(path, output / 'controller' / path.name)
        for name, expected in read(run / 'REQUEST.json')['code_records'].items():
            if record(output / 'controller' / name) != expected:
                raise ValueError('Archived fitting controller differs')
        cohort = cohorts(lockdir, run)
        comparison = Path(comparison); figure_metrics = Path(figure_metrics)
        for directory in (comparison, figure_metrics):
            if read(directory / 'COMPLETE.json') != {'status': 'complete', 'report_sha256': sha(directory / 'report.json')}:
                raise ValueError('Unsealed comparison')
        with tempfile.TemporaryDirectory(prefix='cd19_validation_') as tmp:
            copy = Path(tmp).resolve() / 'controller'; copy.mkdir()
            for p in HERE.glob('*.py'):
                shutil.copyfile(p, copy / p.name)
            shutil.copytree(HERE / 'tests', copy / 'tests', ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copytree(lockdir, copy / 'frozen_20261011')
            command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v']
            test = subprocess.run(command, cwd=copy, capture_output=True, text=True)
            (output / 'tests.stdout').write_text(test.stdout); (output / 'tests.stderr').write_text(test.stderr)
            if test.returncode:
                raise ValueError('Temporary-copy tests failed')
            moved = Path(tmp).resolve() / 'moved lock with spaces'
            (copy / 'frozen_20261011').rename(moved)
            cmd = [sys.executable, '-B', str(copy / 'study.py'), 'verify-run', '--lockdir', str(moved), '--run', str(run)]
            checked = subprocess.run(cmd, cwd=copy, capture_output=True, text=True)
            (output / 'moved.stdout').write_text(checked.stdout); (output / 'moved.stderr').write_text(checked.stderr)
            if checked.returncode:
                raise ValueError('Moved-lock verification failed')
            copied_bytes = sum(p.stat().st_size for p in Path(tmp).rglob('*') if p.is_file())
        prior = check_prior(HERE / 'validation/preflight_20261011')
        original = original_indexes()
        report = {'status': 'internally_checked_pending_external_audit', 'structure': structure, 'cohorts': cohort,
                  'comparison': record(comparison / 'report.json'), 'figure_metrics': record(figure_metrics / 'report.json'),
                  'comparison_status': read(comparison / 'report.json')['status'],
                  'figure_metrics_status': read(figure_metrics / 'report.json')['status'],
                  'tests_returncode': test.returncode, 'moved_lock_and_controller_returncode': checked.returncode,
                  'temporary_copied_bytes': copied_bytes, 'prior': prior, 'original_indexes': original,
                  'current_code': {p.name: record(p) for p in sorted(HERE.glob('*.py'))},
                  'resource_state': resource_state(), 'no_raw_image_reextraction_or_final_pdf_render': True}
        write(output / 'report.json', report)
        write(output / 'COMPLETE.json', {'status': 'complete', 'report_sha256': sha(output / 'report.json')})
        return report
    except BaseException as exc:
        write(output / 'FAILED.json', {'error': repr(exc)})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for field in ('lockdir', 'run', 'comparison', 'figure-metrics', 'out'):
        p.add_argument('--' + field, required=True)
    a = p.parse_args(); print(json.dumps(validate(a.lockdir, a.run, a.comparison, a.figure_metrics, a.out), indent=2))
