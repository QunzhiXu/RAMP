"""Prepare and verify seven frozen CytoNorm input bundles; no fit command."""
import argparse
from pathlib import Path

import cytonorm_io as io
import inventory as inv

HERE=Path(__file__).resolve().parent


def plan(case):
    c=inv.verify_local();matches=[s for s in c['branches'] if s['case']==case]
    if len(matches)!=1:raise ValueError('Unknown case')
    s=matches[0];by_source={r['source']:r for r in c['files']}
    needed=[s[k] for k in ('input','marker_file','config','input_manifest')]
    if s['kind']=='story':needed += [s['metadata'],s['ids']]
    files=[]
    for n in needed:
        r=by_source[n];files.append({'path':'inputs/'+n,'source':n,'origin':'source',**{k:r[k] for k in ('bytes','sha256')}})
    for r in c['files']:
        if r.get('copy','').startswith(('adapter/','environment/')):
            files.append({'path':r['copy'],'source':r['source'],'origin':'snapshot','snapshot_path':r['copy'],
                          **{k:r[k] for k in ('bytes','sha256')}})
    if s['kind']=='story':
        for f in ('run_r.R','cytonorm_repro.R'):
            r=by_source[s['base']+'/scripts/'+f]
            files.append({'path':'scripts/'+f,'source':r['source'],'origin':'snapshot','snapshot_path':r['copy'],
                          **{k:r[k] for k in ('bytes','sha256')}})
    return s,sorted(files,key=lambda r:r['path'])


def validate_inputs(bundle,s):
    root=Path(bundle)/'inputs';counts=inv.check_input(root,s,canonical=False)
    cfg=io.read_json(root/s['config']);im=io.read_json(root/s['input_manifest'])
    if s['kind']=='technical':
        if {k:cfg[k] for k in inv.PARAMETERS}!=inv.PARAMETERS:raise ValueError('Technical parameters differ')
        if im['labels_read'] is not False or im['case']!=s['case'] or im['design']['cases'][s['case']]['datasets']!=s['datasets']:raise ValueError('Preprocessing contract differs')
        if im['design']['cases'][s['case']]['transforms']!=s['preprocessing']:raise ValueError('Preprocessing scale differs')
        for key,path in [('input_sha256',s['input']),('markers_sha256',s['marker_file']),('preprocessing_sha256',s['input_manifest'])]:
            if cfg[key]!=io.sha(root/path):raise ValueError('Technical input hash differs')
        for key in ('input','marker_file'):
            if im['outputs'][Path(s[key]).name]!=io.sha(root/s[key]):raise ValueError('Preprocessing output hash differs')
        for key in cfg['code_hashes']:
            if cfg['code_hashes'][key]!=io.sha(Path(bundle)/'adapter'/key):raise ValueError('Frozen R code hash differs')
        if cfg['upstream_lock_sha256']!=io.sha(Path(bundle)/'adapter/upstream_lock.json') or cfg['runtime_lock_sha256']!=io.sha(Path(bundle)/'environment/runtime_lock.csv'):raise ValueError('Environment lock differs')
    else:
        if cfg['datasets']!=s['datasets'] or cfg['markers']!=s['markers'] or cfg['seed']!=42 or cfg['input_manifest_sha256']!=io.sha(root/s['input_manifest']):raise ValueError('Story configuration differs')
        for key in ('input','marker_file','metadata','ids'):
            if im['output_sha256'][Path(s[key]).name]!=io.sha(root/s[key]):raise ValueError('Story input hash differs')
        design=io.read_json(Path(bundle)/'adapter/config.json')
        if {k:v for k,v in design.items() if k!='cases'}!=inv.PARAMETERS:raise ValueError('Story method parameters differ')
    return counts


def verify_bundle(bundle):
    root=Path(bundle).resolve(strict=True);m=io.complete(root,'MANIFEST.json')
    if m.get('status')!='complete':raise ValueError('Incomplete bundle')
    spec,files=plan(m['case'])
    if m['catalog_sha256']!=inv.CATALOG_SHA or m['files']!=files:raise ValueError('Bundle contract differs')
    io.payload(root,{r['path']:r for r in files},('MANIFEST.json','COMPLETE.json'))
    if validate_inputs(root,spec)!=m['counts']:raise ValueError('Bundle counts differ')
    return spec,files


def code_records():return {n:io.record(HERE/n) for n in ('cytonorm_io.py','inventory.py','study.py')}


def prepare(source,output,case):
    source=Path(source).resolve(strict=True);spec,files=plan(case);codes=code_records()
    bindings=[(r,io.within(source,r['source']) if r['origin']=='source' else io.within(inv.SNAPSHOT,r['snapshot_path'])) for r in files]
    for r,p in bindings:io.check(p,r)
    out=io.reserve(output,(source,inv.SNAPSHOT,inv.REGISTRY,*(HERE/n for n in codes)))
    try:
        for r,p in bindings:io.copy_new(p,out/r['path'],r)
        counts=validate_inputs(out,spec)
        for r,p in bindings:io.check(p,r)
        if plan(case)!=(spec,files) or code_records()!=codes:raise ValueError('Code/contract changed during preparation')
        io.create_json(out/'MANIFEST.json',{'schema':1,'status':'complete','case':case,'catalog_sha256':inv.CATALOG_SHA,
            'files':files,'counts':counts,'source_root_at_preparation':str(source),'code_records':codes,
            'scope':'Frozen analysis-ready inputs and unchanged code; no fitted results; no preprocessing recomputation'})
        io.payload(out,{r['path']:r for r in files},('MANIFEST.json',))
        io.create_json(out/'COMPLETE.json',{'status':'complete','manifest_sha256':io.sha(out/'MANIFEST.json')})
    except BaseException as e:io.failure(out,e);raise
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('list','prepare','verify'))
    for n in ('source-root','output','bundle'):p.add_argument('--'+n,type=Path)
    p.add_argument('--case');a=p.parse_args()
    if a.command=='list':
        for s in inv.verify_local()['branches']:print(s['case'],s['cells'],len(s['markers']),s['units'])
    elif a.command=='prepare':
        if not all((a.source_root,a.output,a.case)):p.error('prepare requires source-root, output and case')
        print(prepare(a.source_root,a.output,a.case))
    else:
        if a.bundle is None:p.error('verify requires bundle')
        s,f=verify_bundle(a.bundle);print(s['case'],len(f),'files verified')


if __name__=='__main__':main()
