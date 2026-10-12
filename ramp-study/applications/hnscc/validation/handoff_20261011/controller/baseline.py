"""Small read-only provenance checks; no prior scientific payload duplication."""
import argparse
from pathlib import Path
from common import HERE, STUDY, read, write, record, sha


def capture(out):
    out = Path(out); out.mkdir(exist_ok=False)
    prior = {}
    for path in sorted(STUDY.rglob('*')):
        rel = path.relative_to(STUDY)
        if path.is_file() and not path.is_symlink() and not path.is_relative_to(HERE) and not any(
                p in ('runs', '.venv', '__pycache__', '.cache', '.git') or p.startswith('.') for p in rel.parts):
            # Exact hashes for code, locks and reports; prior heavy results remain in place.
            if path.stat().st_size <= 20_000_000:
                prior[str(rel)] = record(path)
    write(out / 'PRIOR.json', prior)
    write(out / 'COMPLETE.json', {'prior_sha256': sha(out / 'PRIOR.json'), 'files': len(prior),
                                'excludes': ['this module', 'runs', 'hidden directories', 'files over 20 MB']})
    return {'files': len(prior)}


def check(out):
    out = Path(out)
    if sha(out / 'PRIOR.json') != read(out / 'COMPLETE.json')['prior_sha256']:
        raise ValueError('Baseline record changed')
    prior = read(out / 'PRIOR.json')
    for rel, expected in prior.items():
        if record(STUDY / rel) != expected:
            raise ValueError(f'Prior changed: {rel}')
    return {'files_unchanged': len(prior)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['capture', 'check']); p.add_argument('--out', required=True)
    a = p.parse_args(); print(capture(a.out) if a.mode == 'capture' else check(a.out))

