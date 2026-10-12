"""Create-only phase-2 evidence. No fitting or historical runner imports."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
from safe_io import require, record, read, write, reserve, member
import inventory as phase1

ANCHOR = '94020a5fe7b872def27717ffcb6357ed62680b64de6d9c41fe32d25c4d39f76a'
HANDOFF = '71e07e77424512f0c035b437d37a88b039463eabf224e83f3788919bb1283c4e'
SNAPSHOT = PARENT / 'frozen_20261011_r2'
METHODS = ('ramp','unaligned','clipped_zscore','minmax','percentile','harmony_direct','fastmnn','cycombine','cytonorm','maxfuse')
COUNTS = {'J1': dict(bone_marrow=540,tnbc=540,melanoma=540,lung_cancer=540,lung_story=540,melanoma_story=180,hnscc_story=180),
          'J2': dict(bone_marrow=7560,tnbc=5940,lung_cancer=4320,lung_story=4860,melanoma_story=2670,hnscc_story=3960)}
CODE = ('common.py','binding.py','scores.py','replay.py')

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()

def code_records():
    return {n:record(HERE/n) for n in CODE}

def verified_bytes(root, name, records):
    p = member(root, name)
    data = p.read_bytes()
    require({'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} == records[name], 'Bound file changed: '+name)
    return data

def bound_json(root,name,records):
    return json.loads(verified_bytes(root,name,records))

def key(row):
    j=row['job']
    return tuple(j.get(k,'') for k in ('task_type','task','method','marker','source','target','family','seed'))

def check_grid(rows, expected=None):
    keys=[key(r) for r in rows]
    require(len(keys)==len(set(keys)), 'Duplicate prediction tasks')
    if expected is not None:
        require(set(keys)=={key(r) for r in expected}, 'Prediction task grid differs')
    else:
        for kind,cases in COUNTS.items():
            for case,n in cases.items():
                group=[r for r in rows if r['job']['task_type']==kind and r['job']['task']==case]
                require(len(group)==n, 'Task count differs: '+kind+'/'+case)
        require(len(rows)==32370, 'Unexpected primary tasks')
        groups={}
        for r in rows:
            k=key(r); groups.setdefault(k[:2]+k[3:],[]).append(r)
        for g in groups.values():
            require(len(g)==10 and {r['job']['method'] for r in g}==set(METHODS),'Unpaired methods')
            require(len({r['job']['supervision_sha256'] for r in g})==1,'Method-specific supervision')

def phase1_anchor():
    require(record(SNAPSHOT/'catalog.json')['sha256']==ANCHOR,'Phase-1 catalog changed')
    require(record(PARENT/'validation/handoff_20261011_r2/report.json')['sha256']==HANDOFF,'Phase-1 handoff changed')
    return read(SNAPSHOT/'catalog.json')

def output(path, source, protected=()):
    # Check before creating even a parent directory; code/evidence are read-only inputs.
    resolved=Path(path).resolve()
    if resolved.is_relative_to(PARENT.parents[1]):
        require(any(resolved.is_relative_to(HERE/n) and resolved!=HERE/n for n in ('runs','validation','bindings')), 'Output outside new phase-2 evidence roots')
    return reserve(path,[source, SNAPSHOT, PARENT/'validation', PARENT.parents[2]/'ramp-proteomics', *protected])

def environment():
    import platform, numpy, pandas, scipy, sklearn
    return dict(python=platform.python_version(),numpy=numpy.__version__,pandas=pandas.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__)

