"""Stage one leave-one-marker-out alignment run.

The R algorithms verify that the inputs they are handed are the ones the
benchmark recorded: each reads a marker file, a preprocessing record and a
config, and refuses to run unless the hashes agree.  Dropping a marker is a
legitimate change to the input, so the affected hashes are recomputed here and
every other recorded hash is left exactly as the benchmark wrote it.  Nothing
is written inside final_xu; references that would escape the staged copy are
rewritten to absolute paths back into the read-only tree.
"""
import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import resolve

PROJECT = Path(__file__).resolve().parents[2]
WORK = PROJECT / 'lomo_work'
ALGO = WORK / 'algorithm'
FINAL_XU = Path('/Users/qxu6/Documents/New project/final_xu')

# Which recorded hashes name the marker file, and which name the preprocessing
# record.  Everything else in these files is left untouched.
SPEC = {
    'fastmnn':  dict(runner=ALGO / 'fastmnn/run_fastmnn.R',
                     marker_keys=['correction_marker_sha256', 'marker_file_sha256'],
                     record_keys=['preprocessing_record_sha256'],
                     record='manifest'),
    'cycombine': dict(runner=ALGO / 'cycombine/run_cycombine.R',
                     marker_keys=['correction_marker_sha256'],
                     record_keys=[], record='manifest'),
    'cytonorm': dict(runner=ALGO / 'CytoNorm/run_cytonorm.R',
                     marker_keys=['markers_sha256'],
                     record_keys=[], record=None),
    'harmony':  dict(runner=ALGO / 'harmony/run_harmony_direct.R',
                     marker_keys=['marker_file_sha256'],
                     record_keys=[], record=None),
}
ENV = dict(os.environ, LC_COLLATE='en_US.UTF-8', LANG='en_US.UTF-8')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def set_keys(path, updates):
    d = json.loads(Path(path).read_text())
    n = 0
    def walk(o):
        nonlocal n
        if isinstance(o, dict):
            for k, v in o.items():
                if k in updates:
                    o[k] = updates[k]; n += 1
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    walk(d)
    Path(path).write_text(json.dumps(d, indent=2) + '\n')
    return n


def absolutise(path, src_dir, src_root):
    """Rewrite only the *_file references that escape the staged method root.

    In-tree relative references must stay relative so they resolve against the
    staged copy; the ones that point outside it are pinned to the original
    read-only files, which are unchanged by this study.
    """
    d = json.loads(Path(path).read_text())
    def walk(o):
        if isinstance(o, dict):
            for k, v in list(o.items()):
                target = (Path(src_dir) / v).resolve() if isinstance(v, str) else None
                if (target is not None and k.endswith('_file') and v.startswith('..')
                        and not target.is_relative_to(src_root)):
                    o[k] = str(target)
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    walk(d)
    Path(path).write_text(json.dumps(d, indent=2) + '\n')


def stage(task, method, drop, workdir):
    f = resolve.find(task, method)
    spec = SPEC[method]
    w = Path(workdir); shutil.rmtree(w, ignore_errors=True); w.mkdir(parents=True)
    src_root = Path(f['root'])

    shutil.copytree(src_root, w / 'method', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('umap', 'evaluation', 'figures', '*.rds'))
    shutil.rmtree(w / 'method' / 'alignment' / 'data', ignore_errors=True)
    def local(p):
        p = Path(p)
        try:
            t = w / 'method' / p.relative_to(src_root)
        except ValueError:                       # shared application input
            t = w / 'shared' / p.name
        t.parent.mkdir(parents=True, exist_ok=True)
        if not t.exists():
            shutil.copy(p, t)
        t.chmod(0o644)
        return t

    mk, cfg = local(f['markers']), local(f['config'])
    inp = local(f['input'])
    man = local(f['manifest']) if spec['record'] else None

    markers = Path(mk).read_text().split()
    if drop not in markers:
        raise ValueError(f'{drop} not among {task} markers')
    Path(mk).write_text('\n'.join(m for m in markers if m != drop) + '\n')

    if method == 'cytonorm':
        # CytoNorm asserts the matrix columns are exactly the three key columns
        # followed by the marker list, so the dropped marker leaves the matrix too.
        import pandas as pd
        frame = pd.read_csv(inp)
        frame.drop(columns=[drop]).to_csv(inp, index=False)
        updates_input = {'input_sha256': sha(inp)}
    else:
        updates_input = {}

    for p, s in ((man, Path(f['manifest']).parent), (cfg, Path(f['config']).parent)):
        if p: absolutise(p, s, src_root)

    updates = {k: sha(mk) for k in spec['marker_keys']}
    updates.update(updates_input)
    if man:
        set_keys(man, updates)
        updates.update({k: sha(man) for k in spec['record_keys']})
    set_keys(cfg, updates)

    return dict(input=inp, markers=mk, config=cfg, out=w / 'out',
                runner=spec['runner'], n_markers=len(markers) - 1)


def run(task, method, drop, workdir, timeout=1800):
    p = stage(task, method, drop, workdir)
    if method == 'cytonorm':                     # runner insists on algorithm_test/
        p['out'] = WORK / 'algorithm_test' / f'lomo_{task}_{drop}'
        shutil.rmtree(p['out'], ignore_errors=True)
        p['out'].parent.mkdir(parents=True, exist_ok=True)
    cmd = ['Rscript', '--vanilla', str(p['runner']),
           '--input', str(p['input']), '--markers', str(p['markers']),
           '--config', str(p['config']), '--output', str(p['out'])]
    r = subprocess.run(cmd, capture_output=True, text=True, env=ENV,
                       cwd=str(Path(workdir)), timeout=timeout)
    return p, r
