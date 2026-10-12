"""Synthetic file contracts; never fit real cohorts."""
import copy
from contextlib import contextmanager
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cytonorm_io as io
import inventory as inv
import study


@contextmanager
def fixture(root,kind='technical',platforms=3):
    import numpy as np
    import pandas as pd
    root=Path(root).resolve(strict=True);source=root/'source';source.mkdir();snapshot=root/'snapshot';snapshot.mkdir()
    real=inv.verify_local();old=next(s for s in real['branches'] if s['kind']==kind);spec=copy.deepcopy(old)
    markers=[f'M{i}' for i in range(5)];datasets=[f'platform_{i}' for i in range(platforms)]
    n=platforms*12;g=np.repeat(np.arange(platforms),12);x=np.random.default_rng(1729).normal(3,.7,(n,5));x[0,:]=0
    inp=pd.DataFrame(x,columns=markers);inp.insert(0,'unit_key',[f'{datasets[a]}_unit{i%3}' for i,a in enumerate(g)])
    inp.insert(0,'dataset',[datasets[a] for a in g]);inp.insert(0,'cell_id',[f'cell_{i:04}' for i in range(n)])
    spec.update(case='synthetic_'+kind,cells=n,units=3*platforms,markers=markers,datasets=datasets)
    path=source/spec['input'];path.parent.mkdir(parents=True)
    inp.to_csv(path,index=False,float_format='%.17g',compression={'method':'gzip','mtime':0})
    (source/spec['marker_file']).write_text('\n'.join(markers)+'\n')
    cfg=copy.deepcopy(io.read_json(inv.SNAPSHOT/'study_sources'/old['config']))
    if kind=='technical':
        manifest={'labels_read':False,'case':spec['case'],'design':{'cases':{spec['case']:{'datasets':datasets,'transforms':spec['preprocessing']}}},
                  'outputs':{Path(spec[k]).name:io.sha(source/spec[k]) for k in ('input','marker_file')}}
        io.create_json(source/spec['input_manifest'],manifest)
        cfg.update(input_sha256=io.sha(path),markers_sha256=io.sha(source/spec['marker_file']),preprocessing_sha256=io.sha(source/spec['input_manifest']))
    else:
        inp[['cell_id','dataset','unit_key']].to_csv(source/spec['metadata'],index=False,compression={'method':'gzip','mtime':0})
        with (source/spec['ids']).open('xb') as f:np.save(f,inp.cell_id.to_numpy(dtype='U'))
        manifest={'output_sha256':{Path(spec[k]).name:io.sha(source/spec[k]) for k in ('input','marker_file','metadata','ids')}}
        io.create_json(source/spec['input_manifest'],manifest)
        cfg.update(datasets=datasets,markers=markers,input_manifest_sha256=io.sha(source/spec['input_manifest']))
    (source/spec['config']).parent.mkdir(parents=True,exist_ok=True);io.create_json(source/spec['config'],cfg)
    needed=[spec[k] for k in ('input','marker_file','config','input_manifest')]
    if kind=='story':needed += [spec['metadata'],spec['ids']]
    files=[{'source':name,**io.record(source/name)} for name in needed]
    for r in real['files']:
        if r.get('copy','').startswith(('adapter/','environment/')) or (kind=='story' and r['source'] in [spec['base']+'/scripts/'+f for f in ('run_r.R','cytonorm_repro.R')]):
            io.copy_new(inv.SNAPSHOT/r['copy'],snapshot/r['copy'],r);files.append(r)
    io.copy_new(inv.SNAPSHOT/'registry.json',snapshot/'registry.json',real['registry'])
    c={'status':'complete','registry':real['registry'],'branches':[spec],'files':files,'links':[]}
    io.create_json(snapshot/'catalog.json',c);io.create_json(snapshot/'COMPLETE.json',{'status':'complete','manifest_sha256':io.sha(snapshot/'catalog.json')})
    with patch.object(inv,'SNAPSHOT',snapshot),patch.object(inv,'CATALOG_SHA',io.sha(snapshot/'catalog.json')):
        yield source,spec
