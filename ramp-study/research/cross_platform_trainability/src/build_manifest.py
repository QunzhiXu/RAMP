"""Record a sha256 for every input file before it is read. Read-only."""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import paths, common

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

man = {'generated': '2026-09-18',
       'design_sha256': sha(paths.ROOT / 'design.json'),
       'source_root': str(paths.SRC),
       'rule': 'read-only; these files are never modified',
       'tasks': {}}

for t in paths.D['tasks']:
    name = t['name']
    entry = {'metadata': {}, 'markers': {}, 'arms': {}}
    for lbl, p in (('metadata', paths.meta_path(name)), ('markers', paths.marker_path(name))):
        entry[lbl] = {'path': str(p.relative_to(paths.SRC)), 'sha256': sha(p), 'bytes': p.stat().st_size}
    for arm in common.ARMS:
        fs = paths.arm_files(name, arm)
        if paths.D['arms'][arm]['kind'] == 'recipe':
            entry['arms'][arm] = {'derived_from': 'Raw', 'files': []}
            continue
        entry['arms'][arm] = {'files': [
            {'path': str(f.relative_to(paths.SRC)), 'sha256': sha(f), 'bytes': f.stat().st_size}
            for f in fs]}
    # locally cached fastMNN rotation for the applications
    loc = paths.ROOT / 'data' / name
    if loc.exists():
        entry['local_cache'] = {f.name: {'sha256': sha(f), 'bytes': f.stat().st_size}
                                for f in sorted(loc.glob('*.csv'))}
    man['tasks'][name] = entry
    print(f'{name:<16} hashed', flush=True)

out = paths.ROOT / 'data' / 'MANIFEST.json'
out.write_text(json.dumps(man, indent=2) + '\n')
n = sum(len(a.get('files', [])) for t in man['tasks'].values() for a in t['arms'].values())
print(f'\n{len(man["tasks"])} tasks, {n} arm files hashed -> {out.relative_to(paths.ROOT)}')
