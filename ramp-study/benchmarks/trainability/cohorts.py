"""Read frozen native matrices/metadata only; no provider extraction or fitting."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import inventory as inv
from safe_io import read, require


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def check_matrix(native, md, markers, pool_id, pool_sha):
    require(native.shape == (len(md),len(markers)) and np.isfinite(native).all(), 'Bad native matrix')
    require(len(set(markers))==len(markers) and md.cell_id.is_unique, 'Duplicate identities')
    require(md[['cell_id','dataset','unit_key']].ne('').all().all(), 'Missing identity')
    require(md.groupby('unit_key').dataset.nunique().eq(1).all(), 'Unit crosses datasets')
    expected=digest(dict(pool_id=pool_id,cell_ids=list(md.cell_id),datasets=list(md.dataset),units=list(md.unit_key)))
    require(expected==pool_sha,'Native metadata pool/order differs')
    return {'cells':len(md),'markers':list(markers),'dtype':str(native.dtype),'units':md.unit_key.nunique(),
            'pool_sha256':expected,'min':float(native.min()),'max':float(native.max()),
            'datasets':[{'dataset':d,'cells':len(g),'units':g.unit_key.nunique()} for d,g in md.groupby('dataset',sort=False)],
            'identity_levels':sorted(md.identity_level.unique()) if 'identity_level' in md else None}


def inspect(source):
    source=Path(source);r=source/inv.ROOT;rows={}
    for case in inv.CASES:
        base=r/'data/real_inputs_v0_3_r2'/case;m=read(base/'manifest.json')
        md=pd.read_csv(base/'metadata.csv.gz',dtype=str,keep_default_na=False)
        x=np.load(base/'native.npy',mmap_mode='r',allow_pickle=False)
        rows[case+':formal']=check_matrix(x,md,m['markers'],m['pool_id'],m['pool_sha256'])
        rows[case+':formal']['contracts']=m['contracts']
    base=source/'story_lung_HLA_ABC/releases/senosain140k_20260921/inputs'
    md=pd.read_csv(base/'CELL_METADATA.csv.gz',dtype=str,keep_default_na=False).rename(columns={'cell_key':'cell_id'})
    m=read(r/inv.J2/'MANIFEST.json');c=read(r/inv.J2/'INPUT_CONTRACT.json')
    x=np.load(base/'RAW_SHARED9.npy',mmap_mode='r',allow_pickle=False)
    rows['lung_story:current']=check_matrix(x,md,c['markers'],'lung_story:senosain140k_release_20260921:all84000',m['pool_sha256'])
    rows['lung_story:current']['contracts']=c['contracts']
    return rows
