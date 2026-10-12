"""Create-only I/O; adapted from the accepted cyCombine inventory helpers.

No imports execute commands or write files. Directory ownership uses mkdir,
not a check followed by replacement. Partial failures are retained for diagnosis.
"""
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def create_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def record(path):
    return {'bytes': Path(path).stat().st_size, 'sha256': sha(path)}


def check(path, expected):
    if record(path) != {k: expected[k] for k in ('bytes', 'sha256')}:
        raise ValueError('File changed: ' + str(path))


def within(root, relative):
    root, relative = Path(root).resolve(strict=True), Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe relative path: ' + str(relative))
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Not an internal file: ' + str(path))
    return path


def reserve(output, protected=()):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    target = output.resolve()
    for item in protected:
        item = Path(item).resolve()
        if target == item or target.is_relative_to(item) or item.is_relative_to(target):
            raise ValueError('Output overlaps protected input: ' + str(item))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    return output.resolve(strict=True)


def copy_new(source, target, expected):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open('rb') as src, target.open('xb') as dst:
        shutil.copyfileobj(src, dst)
    check(target, expected)


def payload(root, records, extras=(), allowed_links=None):
    """Reject unlisted files and all symlinks except explicitly bound runtime libraries."""
    root = Path(root).resolve(strict=True)
    allowed_links = allowed_links or {}
    actual, links = set(), {}
    for p in root.rglob('*'):
        name = p.relative_to(root).as_posix()
        if p.is_symlink():
            links[name] = str(p.resolve(strict=True))
        elif p.is_file():
            actual.add(name)
    if links != allowed_links or actual != set(records) | set(extras):
        raise ValueError('Payload inventory differs')
    for name, expected in records.items():
        check(within(root, name), expected)


def complete(root, manifest_name):
    done = read_json(Path(root) / 'COMPLETE.json')
    if done.get('status') != 'complete' or done.get('manifest_sha256') != sha(Path(root) / manifest_name):
        raise ValueError('Incomplete or changed directory: ' + str(root))
    return read_json(Path(root) / manifest_name)


def failure(root, error):
    create_json(Path(root) / 'FAILED.json', {'error': repr(error)})


def table(path):
    import pandas as pd
    return pd.read_csv(path, dtype={n: str for n in ('cell_id', 'dataset', 'unit_key', 'batch', 'sample')},
                       keep_default_na=False, low_memory=False, float_precision='round_trip')
