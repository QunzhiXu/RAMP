"""Create-only local I/O. No module import performs reads, writes or subprocesses."""
import hashlib
import json
from pathlib import Path
import shutil


def record(path):
    path = Path(path)
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': digest}


def check(path, expected):
    if record(path) != {k: expected[k] for k in ('bytes', 'sha256')}:
        raise ValueError('File changed: ' + str(path))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def reserve(output, protected=()):
    path = Path(output).absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path = path.resolve()
    for source in protected:
        source = Path(source).resolve()
        if path == source or path.is_relative_to(source) or source.is_relative_to(path):
            raise ValueError('Output overlaps protected input: ' + str(source))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    return path


def child(root, name):
    rel = Path(name)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError('Unsafe child: ' + str(name))
    root = Path(root).resolve(strict=True)
    path = root / rel
    for ancestor in (path, *path.parents):
        if ancestor == root: break
        if ancestor.is_symlink(): raise ValueError('Symlink not permitted: ' + str(ancestor))
    if not path.resolve().is_relative_to(root):
        raise ValueError('Escaping child: ' + str(name))
    return path


def copy(source, target, expected):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open('rb') as src, target.open('xb') as dst:
        shutil.copyfileobj(src, dst)
    check(target, expected)


def files(root):
    root = Path(root)
    rows = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink(): raise ValueError('Unexpected symlink: ' + str(p))
        if p.is_file(): rows[p.relative_to(root).as_posix()] = record(p)
    return rows


def payload(root, rows, extras=()):
    actual = files(root)
    if set(actual) != set(rows) | set(extras):
        raise ValueError('Payload file inventory differs')
    for n, r in rows.items(): check(child(root, n), r)


def failed(root, error):
    write(Path(root)/'FAILED.json', {'error': repr(error)})
