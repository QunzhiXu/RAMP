"""Real label-free adapters. External algorithms are read-only, artifacts are v2-only.

No adapter receives labels, survival, spatial coordinates or target answers.
Native-scale RAMP/Harmony and assay-ready comparators are selected by the caller.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tempfile
import platform
import importlib.metadata
import numpy as np
import pandas as pd
from .core import ROOT, digest, file_sha256, output_path, require, write_json
from .inputs import FittingPool, LabelFreeInput, PreparedRepresentation, simple_representation

FX = ROOT.parent
ALG = FX / 'algorithm'
HERE = Path(__file__).parent
SIMPLE = {'unaligned', 'clipped_zscore', 'minmax', 'percentile'}


def environment(work):
    env = os.environ.copy()
    env.update({k: '1' for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS', 'LOKY_MAX_CPU_COUNT']})
    env.update(PYTHONDONTWRITEBYTECODE='1', LC_COLLATE='en_US.UTF-8', LANG='en_US.UTF-8')
    for key, sub in [('TMPDIR','tmp'), ('NUMBA_CACHE_DIR','numba'),
                     ('MPLCONFIGDIR','matplotlib'), ('XDG_CACHE_HOME','xdg')]:
        p = work / sub
        p.mkdir(exist_ok=True)
        env[key] = str(p)
    return env


def run_command(command, work, logname, timeout):
    env=environment(work)
    # R rejects spaces in TMPDIR. A disposable path alias points INTO this run;
    # generated temporary data still live below the v2 folder.
    with tempfile.TemporaryDirectory(prefix='trainability-r-') as alias:
        link=Path(alias)/'tmp';link.symlink_to(work/'tmp',target_is_directory=True)
        env['TMPDIR']=str(link)
        with (work / logname).open('w') as log:
            result = subprocess.run([str(x) for x in command], cwd=work,
                env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    require(result.returncode == 0, f'Adapter failed ({result.returncode}); see {work / logname}')


def source_files(method):
    paths = [HERE/f for f in ['adapters.py','inputs.py','core.py','adapter_ramp.py','adapter_r.R']]
    if method == 'ramp': paths += list((ALG/'ramp').glob('*.py'))
    elif method == 'maxfuse':
        paths += list((ALG/'maxfuse').glob('*.py'))
        paths += [FX/'benchmark_results/bone_marrow/maxfuse_v2/pairwise_config.json']
    elif method not in SIMPLE:
        name = {'harmony_direct':'harmony', 'cytonorm':'CytoNorm'}.get(method,method)
        paths += list((ALG/name).glob('*.R')) + list((ALG/name).glob('*lock*'))
        paths += list((HERE/'vendor').glob('*.R'))
        if method == 'cytonorm': paths += [ALG/name/'config.json']
    return {str(p):file_sha256(p) for p in paths if p.is_file()}


def _maxfuse(data, work, timeout):
    datasets = list(dict.fromkeys(data.datasets))
    require(len(datasets) in {2,3}, 'MaxFuse requires two or three views')
    x = data.x.copy()
    ds = np.asarray(data.datasets)
    ids = np.asarray(data.cell_ids)
    markers = np.asarray(data.markers)
    for name in datasets:
        m = ds == name
        sd = x[m].std(axis=0, ddof=1)
        require(np.all(sd>0), f'MaxFuse constant marker in {name}; no silent feature removal')
        x[m] = (x[m] - x[m].mean(axis=0))/sd
    profile = json.loads((FX/'benchmark_results/bone_marrow/maxfuse_v2/pairwise_config.json').read_text())
    dims = min(len(markers)-1, min(int((ds==d).sum())-1 for d in datasets))
    require(dims >= 1, 'Insufficient MaxFuse dimensions')
    for block in ['graph','initial','refine','propagation']:
        for k,v in profile[block].items():
            if ('svd_components' in k or k == 'cca_components') and v is not None:
                profile[block][k] = min(v,dims)
    archive = work/'input.npz'
    payload = {}
    for i,d in enumerate(datasets):
        m=ds==d
        if len(datasets)==2:
            for kind in ['shared','active']:
                payload[f'{kind}{i+1}']=x[m]
                payload[f'{kind}{i+1}_feature_names']=markers
            payload[f'cell_ids{i+1}']=ids[m]
        else:
            for kind in ['shared','active']:
                payload[f'view{i}_{kind}']=x[m]
                payload[f'view{i}_{kind}_feature_names']=markers
            payload[f'view{i}_cell_ids']=ids[m]
    np.savez_compressed(archive,**payload)
    out=work/'maxfuse'
    python=ALG/'maxfuse/.venv/bin/python'
    if len(datasets)==2:
        cfg=write_json(work/'profile.json',profile)
        cmd=[python, ALG/'maxfuse/run_maxfuse_pairwise.py','--input',archive,'--config',cfg,'--output',out]
    else:
        # Frozen first-dataset hub; never select a hub using prediction scores.
        cfg=write_json(work/'profile.json',{'view1':profile,'view2':profile})
        cmd=[python,ALG/'maxfuse/run_maxfuse_three_view_extension.py','--input',archive,
             '--views','view0','view1','view2','--hub','view0','--pair-config-map',cfg,
             '--svd-components','none','none','none','--gcca-components',str(dims),
             '--outer-reuse-policy','allow','--output',out]
    run_command(cmd,work,'adapter.log',timeout)
    ys=[]; ii=[]
    for i in range(len(datasets)):
        stem=f'cca_embedding{i+1}' if len(datasets)==2 else f'view{i}_gcca_embedding'
        idstem=stem+'_cell_ids' if len(datasets)==2 else f'view{i}_gcca_cell_ids'
        ys.append(np.load(out/(stem+'.npy'),allow_pickle=False))
        ii.append(np.load(out/(idstem+'.npy'),allow_pickle=False))
    outputids=np.concatenate(ii)
    require(len(set(outputids))==len(ids) and set(outputids)==set(ids), 'MaxFuse cell coverage mismatch')
    return np.concatenate(ys)[pd.Index(outputids).get_indexer(ids)]


def fit_representation(data: LabelFreeInput, method: str, *, fitting_pool: FittingPool,
                       out: Path, input_contract: dict, family: str, timeout=7200):
    fitting_pool.validate(data)
    out=output_path(out)
    sources=source_files(method)
    runtime=dict(python=sys.version,platform=platform.platform(),
        packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn']})
    key=digest(dict(method=method,pool=fitting_pool.sha256,markers=data.markers,
        mask=data.masked_markers,values=hashlib.sha256(data.x.tobytes()).hexdigest(),
        contract=input_contract,family=family,sources=sources,runtime=runtime,threads=1))
    if (out/'COMPLETE.json').exists():
        receipt=json.loads((out/'COMPLETE.json').read_text())
        require(receipt['key']==key, 'Representation cache provenance mismatch')
        require(file_sha256(out/'values.npy')==receipt['values_sha256'], 'Representation cache changed')
        return PreparedRepresentation(np.load(out/'values.npy'),data.cell_ids,fitting_pool.sha256,method,data.masked_markers)
    require(not out.exists(), 'Incomplete adapter directory exists; inspect it and use a new run ID')
    out.mkdir(parents=True)
    start=time.monotonic()
    write_json(out/'START.json',dict(key=key,method=method,sources=sources,contract=input_contract,
        family=family,pool_sha256=fitting_pool.sha256,masked=data.masked_markers))
    md=pd.DataFrame(dict(cell_id=data.cell_ids,dataset=data.datasets,unit_key=data.units))
    try:
        if method in SIMPLE:
            y=simple_representation(data,method,fitting_pool=fitting_pool)
        elif method=='ramp':
            np.save(out/'input.npy',data.x)
            md.to_csv(out/'identity.csv',index=False)
            write_json(out/'request.json',dict(family=family,markers=data.markers,
                datasets=list(dict.fromkeys(data.datasets)),algorithm=str(ALG)))
            run_command([sys.executable,HERE/'adapter_ramp.py',out],out,'adapter.log',timeout)
            y=np.load(out/'aligned.npy',allow_pickle=False)
        elif method=='maxfuse': y=_maxfuse(data,out,timeout)
        else:
            require(method in {'harmony_direct','cycombine','cytonorm','fastmnn'}, 'Unknown real adapter')
            pd.concat([md,pd.DataFrame(data.x,columns=data.markers)],axis=1).to_csv(out/'input.csv.gz',index=False)
            (out/'markers.txt').write_text('\n'.join(data.markers)+'\n')
            run_command(['Rscript','--vanilla',HERE/'adapter_r.R',out,method,ALG,HERE],out,'adapter.log',timeout)
            frame=pd.read_csv(out/'corrected_expression.csv.gz',dtype={'cell_id':str})
            require(frame.cell_id.is_unique and set(frame.cell_id)==set(data.cell_ids), 'R adapter lost cell coverage')
            y=frame.set_index('cell_id').loc[list(data.cell_ids)].to_numpy(float)
        require(y.ndim==2 and len(y)==len(data.x) and y.shape[1]>0 and np.isfinite(y).all(), 'Invalid adapter result')
        require(source_files(method)==sources,'Algorithm source changed during fitting')
        np.save(out/'values.npy',y)
        np.save(out/'cell_ids.npy',np.asarray(data.cell_ids))
        write_json(out/'COMPLETE.json',dict(key=key,method=method,shape=list(y.shape),
            output_type='embedding' if method in {'fastmnn','maxfuse'} else 'marker_values',
            input_contract=input_contract,sources=sources,runtime=runtime,pool_sha256=fitting_pool.sha256,
            values_sha256=file_sha256(out/'values.npy'),elapsed_seconds=time.monotonic()-start,
            cell_ids_sha256=file_sha256(out/'cell_ids.npy'),labels_used=False))
        return PreparedRepresentation(y,data.cell_ids,fitting_pool.sha256,method,data.masked_markers)
    except Exception as exc:
        write_json(out/'FAILED.json',dict(error=repr(exc),elapsed_seconds=time.monotonic()-start))
        raise
