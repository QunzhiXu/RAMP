"""Read-only integrity check for a clean source candidate, not a license grant."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

BUDGET = 780_000_000
FORBIDDEN = {'.npy','.npz','.rds','.h5ad','.h5','.fcs','.pt','.pth','.pdf','.png','.jpg','.gz','.zip','.whl'}


def relative(value):
    if not isinstance(value,str):raise ValueError('Path must be text')
    p=PurePosixPath(value)
    if not value or p.is_absolute() or p.as_posix()!=value or '..' in p.parts or '\\' in value:
        raise ValueError('Unsafe manifest path')
    return p


def unique_object(pairs):
    out={}
    for k,v in pairs:
        if k in out:raise ValueError('Duplicate JSON key')
        out[k]=v
    return out


def verify(root):
    root=Path(root).absolute()
    if root.is_symlink() or not root.is_dir():raise ValueError('Candidate must be a physical directory')
    manifest=root/'RELEASE_MANIFEST.json'
    if manifest.is_symlink():raise ValueError('Manifest link rejected')
    m=json.loads(manifest.read_text(),object_pairs_hook=unique_object)
    if m.get('status')!='internal_source_candidate_publication_blocked' or m.get('publish_allowed') is not False:
        raise ValueError('This check does not approve publication')
    if m.get('budget_bytes')!=BUDGET or m.get('full_study_runnable_without_external_evidence') is not False:
        raise ValueError('Release scope differs')
    if m.get('data_payloads_copied') is not False or not isinstance(m.get('files'),dict) or not m['files']:
        raise ValueError('Invalid payload declaration')
    files=m['files'];expected=set(files)|{'RELEASE_MANIFEST.json'}
    actual=set()
    for d,dirs,names in os.walk(root,followlinks=False):
        for name in dirs[:]:
            p=Path(d)/name
            if p.is_symlink():raise ValueError('Directory link rejected')
            if p==root/'.git':dirs.remove(name)
        for name in names:
            p=Path(d)/name
            if p.is_symlink() or not p.is_file():raise ValueError('Nonregular payload')
            actual.add(p.relative_to(root).as_posix())
    if actual!=expected:raise ValueError('Candidate file set differs')
    total=manifest.stat().st_size;python_files=0
    for name,record in files.items():
        p=relative(name)
        if name=='RELEASE_MANIFEST.json' or 'runs' in p.parts or '.git' in p.parts or p.suffix.lower() in FORBIDDEN:
            raise ValueError('Disallowed source-distribution payload')
        if not isinstance(record,dict) or set(record)!={'bytes','sha256'} or type(record['bytes']) is not int or record['bytes']<0:
            raise ValueError('Malformed file record')
        data=root.joinpath(*p.parts).read_bytes()
        if len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']:
            raise ValueError('Candidate hash differs: '+name)
        total+=len(data)
        if p.suffix=='.py':ast.parse(data,filename=name);python_files+=1
    if total>BUDGET:raise ValueError('Candidate exceeds size budget')
    return {'status':'source_integrity_verified','files':len(actual),'bytes':total,'python_files_parsed':python_files,
            'publish_allowed':False,'full_study_reproduction_tested':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    try:print(json.dumps(verify(p.parse_args().root),indent=2))
    except (ValueError,OSError,SyntaxError) as e:p.exit(1,str(e)+'\n')
