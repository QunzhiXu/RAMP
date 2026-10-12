"""Create-only internal audit artifacts; importing this module has no side effects."""
import hashlib
import json
import os
from pathlib import Path
import shutil


def require(ok, message):
    if not ok:
        raise ValueError(message)


def record(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Not a regular file: ' + str(path))
    with path.open('rb') as stream:
        return {'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def relative(name):
    p = Path(name)
    require(not p.is_absolute() and bool(p.parts) and '..' not in p.parts and '.' != str(p),
            'Unsafe relative path: ' + str(name))
    return p


def member(root, name):
    root = Path(root).resolve(strict=True)
    p = root / relative(name)
    require(not any(q.is_symlink() for q in (p, *p.parents) if q != root and q.is_relative_to(root)),
            'Symlink within tree: ' + str(p))
    require(p.resolve().is_relative_to(root), 'Path escapes root')
    return p


def reserve(path, protected=()):
    path = Path(os.path.abspath(path))
    require(not path.exists() and not path.is_symlink(), 'Output already exists: ' + str(path))
    resolved = path.resolve()
    for base in protected:
        base = Path(base).resolve()
        require(not (resolved.is_relative_to(base) or base.is_relative_to(resolved)), 'Output overlaps protected tree')
    # Resolve the parent so normal macOS /var aliases work; final mkdir arbitrates races.
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.mkdir(exist_ok=False)
    return resolved


def copy_checked(source, destination, expected):
    require(record(source) == expected, 'Source changed before copy: ' + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open('rb') as src, destination.open('xb') as dst:
        shutil.copyfileobj(src, dst, 1024 * 1024)
    require(record(destination) == expected, 'Copy differs: ' + str(destination))


def files(root):
    result = set()
    for p in Path(root).rglob('*'):
        require(not p.is_symlink(), 'Unexpected symlink: ' + str(p))
        if p.is_file():
            result.add(p.relative_to(root).as_posix())
    return result
