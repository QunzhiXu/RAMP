"""Pinned, unchanged support from the preceding stage (P2 fix awaits external review)."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SUPPORT = HERE / 'support'
LOCK_SHA = 'e15a17d75e6118df99529b9fcede7e7e91a07e6ec8b4231c0d666f07676db231'


def record(path):
    with Path(path).open('rb') as f: digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return {'bytes': Path(path).stat().st_size, 'sha256': digest}


def check_support():
    if record(SUPPORT / 'LOCK.json')['sha256'] != LOCK_SHA: raise ValueError('Support lock changed')
    lock = json.loads((SUPPORT / 'LOCK.json').read_text())
    for name, expected in lock['files'].items():
        if record(SUPPORT / name) != expected: raise ValueError('Support code changed: ' + name)
    return lock


check_support()
sys.path.insert(0, str(SUPPORT))
import replay as r
import replay_worker as previous_worker

