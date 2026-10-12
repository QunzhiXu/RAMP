"""Exercise the actual new entry on generated data only (240 and 360 cells)."""
import argparse
import os
from pathlib import Path
import shutil
import tempfile

from fixtures import fixture, io, inv, study


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('library', 'rscript', 'scratch-parent', 'output'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); output = io.reserve(a.output, (inv.SNAPSHOT, a.library))
    results = []
    try:
        for kind, platforms in [('technical', 3), ('story', 2)]:
            with tempfile.TemporaryDirectory(prefix='harmony_synthetic_', dir=a.scratch_parent) as temp:
                root = Path(temp).resolve()
                with fixture(root, kind, platforms) as (source, spec):
                    bundle = study.prepare(source, root / 'bundle with spaces', spec['case'])
                    # Move the bundle before fitting; remove its original source.
                    moved = root / 'moved bundle'; bundle.rename(moved); shutil.rmtree(source)
                    run = study.fit(moved, a.library, a.rscript, root / 'run with spaces', a.scratch_parent)
                    verified = study.verify_run(run, moved)
                    command = [a.rscript, '--vanilla', Path(__file__).parent / 'verify_synthetic.R',
                        run / 'runtime/tree/algorithm/harmony', kind, moved / 'inputs' / spec['input'],
                        moved / 'inputs' / spec['marker_file'], run / study.result_path(spec),
                        moved / 'inputs' / spec['config'], output / (kind + '.json')]
                    env = study.r_environment(); env['TMPDIR'] = str(root)
                    study.run_command(command, output, kind, env)
                    results.append({'kind': kind, 'platforms': platforms, 'scope': 'synthetic only',
                        'verified': verified, 'comparison': io.read_json(output / (kind + '.json')),
                        'run_manifest': io.read_json(run / 'RUN_MANIFEST.json'),
                        'request': io.read_json(run / 'REQUEST.json'), 'commands': io.read_json(run / 'COMMANDS.json')})
                    for name in ('environment_before.stdout.log', 'environment_after.stdout.log', 'correction.stdout.log', 'correction.stderr.log'):
                        io.copy_new(run / name, output / (kind + '.' + name), io.record(run / name))
                    # Verification remains valid after moving completed input bundles again.
                    again = root / 'moved again'; moved.rename(again); study.verify_run(run, again)
        io.create_json(output / 'report.json', {'status': 'passed', 'real_cohort_fit': False, 'results': results,
            'parent_locale_environment': {name: os.environ.get(name) for name in ('LC_ALL', 'LANG')},
            'r_locale_environment': dict(study.R_LOCALE_ENV)})
    except BaseException as error:
        io.failure(output, error); raise
    print(output / 'report.json')


if __name__ == '__main__': main()
