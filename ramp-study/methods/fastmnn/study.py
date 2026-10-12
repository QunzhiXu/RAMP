"""Prepare/verify seven frozen fastMNN input bundles. No real-fit command in phase 1."""
import argparse
from pathlib import Path

import fastmnn_io as io
import inventory as inv

HERE=Path(__file__).resolve().parent


def plan(case):
    catalog=inv.verify_local();matches=[s for s in catalog['branches'] if s['case']==case]
    if len(matches)!=1:raise ValueError('Unknown case')
    spec=matches[0];by_source={r['source']:r for r in catalog['files']}
    needed=[spec[k] for k in ('input','marker_file','config','input_manifest')]
    needed += [spec['record']] if spec['kind']=='technical' else [spec['metadata'],spec['ids']]
    files=[]
    for name in needed:
        row=by_source[name]
        files.append({'path':'inputs/'+name,'source':name,'origin':'source',**{k:row[k] for k in ('bytes','sha256')}})
    for row in catalog['files']:
        if row.get('copy','').startswith('adapter/'):
            files.append({'path':row['copy'],'source':row['source'],'origin':'snapshot','snapshot_path':row['copy'],
                          **{k:row[k] for k in ('bytes','sha256')}})
    if spec['kind']=='story':
        row=by_source[spec['base']+'/scripts/run_r.R']
        files.append({'path':'scripts/run_r.R','source':row['source'],'origin':'snapshot','snapshot_path':row['copy'],
                      **{k:row[k] for k in ('bytes','sha256')}})
    return spec,sorted(files,key=lambda r:r['path'])


def validate_inputs(bundle,spec):
    root=Path(bundle)/'inputs';counts=inv.check_input(root,spec,canonical=False)
    cfg=io.read_json(root/spec['config'])
    if spec['kind']=='technical':
        if cfg['algorithm_parameters']!=inv.PARAMETERS or cfg['batch_levels']!=spec['datasets']:raise ValueError('Technical parameters/order differ')
        if {k:cfg[k] for k in ('cell_id_column','batch_column','sample_column')}!={'cell_id_column':'cell_id','batch_column':'dataset','sample_column':'unit_key'}:raise ValueError('Fitting-column roles differ')
        scale=cfg['input_scale']
        if (scale['status']!='analysis_ready' or scale['algorithm_internal_cosine_normalization'] is not True or
            scale['algorithm_applies_other_expression_transform'] is not False):raise ValueError('Unexpected algorithm transform')
        for key,path in [('input_sha256',spec['input']),('marker_file_sha256',spec['marker_file']),('preprocessing_record_sha256',spec['record'])]:
            if scale[key]!=io.sha(root/path):raise ValueError('Technical input hash differs')
        pm=io.read_json(root/spec['record'])
        for key,path in [('combined_input_sha256',spec['input']),('correction_marker_sha256',spec['marker_file'])]:
            if pm[key]!=io.sha(root/path):raise ValueError('Preprocessing record hash differs')
        for key in ('shared_marker_selection_used_celltype_or_outcome_labels','merge_order_selected_using_posthoc_labels_or_metrics','canonical_parameters_selected_using_posthoc_labels_or_metrics'):
            if pm[key] is not False:raise ValueError('Label-free declaration differs')
        if [r['dataset'] for r in pm['datasets']]!=spec['datasets']:raise ValueError('Preprocessing order differs')
    else:
        if (cfg['datasets']!=spec['datasets'] or cfg['markers']!=spec['markers'] or cfg['seed']!=42 or
            cfg['input_manifest_sha256']!=io.sha(root/spec['input_manifest'])):raise ValueError('Story configuration differs')
        im=io.read_json(root/spec['input_manifest'])
        for key in ('input','marker_file','metadata','ids'):
            if im['output_sha256'][Path(spec[key]).name]!=io.sha(root/spec[key]):raise ValueError('Story input hash differs')
    return counts


def verify_bundle(bundle):
    root=Path(bundle).resolve(strict=True);m=io.complete(root,'MANIFEST.json')
    if m.get('status')!='complete':raise ValueError('Incomplete bundle')
    spec,files=plan(m['case'])
    if m['catalog_sha256']!=inv.CATALOG_SHA or m['files']!=files:raise ValueError('Bundle contract differs')
    io.payload(root,{r['path']:r for r in files},('MANIFEST.json','COMPLETE.json'))
    if validate_inputs(root,spec)!=m['counts']:raise ValueError('Bundle counts differ')
    return spec,files


def code_records():
    return {n:io.record(HERE/n) for n in ('fastmnn_io.py','inventory.py','study.py')}


def prepare(source,output,case):
    source=Path(source).resolve(strict=True);spec,files=plan(case);codes=code_records()
    bindings=[(r,io.within(source,r['source']) if r['origin']=='source' else io.within(inv.SNAPSHOT,r['snapshot_path'])) for r in files]
    for row,path in bindings:io.check(path,row)
    root=io.reserve(output,(source,inv.SNAPSHOT,inv.REGISTRY,*(HERE/n for n in codes)))
    try:
        for row,path in bindings:io.copy_new(path,root/row['path'],row)
        counts=validate_inputs(root,spec)
        for row,path in bindings:io.check(path,row)
        if plan(case)!=(spec,files) or code_records()!=codes:raise ValueError('Code/contract changed during preparation')
        io.create_json(root/'MANIFEST.json',{'schema':1,'status':'complete','case':case,'catalog_sha256':inv.CATALOG_SHA,
            'files':files,'counts':counts,'source_root_at_preparation':str(source),'code_records':codes,
            'scope':'Frozen analysis-ready inputs; no fitted results; no preprocessing recomputed'})
        io.payload(root,{r['path']:r for r in files},('MANIFEST.json',))
        io.create_json(root/'COMPLETE.json',{'status':'complete','manifest_sha256':io.sha(root/'MANIFEST.json')})
    except BaseException as error:io.failure(root,error);raise
    return root


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('list','prepare','verify'))
    p.add_argument('--case');p.add_argument('--source-root',type=Path);p.add_argument('--output',type=Path);p.add_argument('--bundle',type=Path)
    a=p.parse_args()
    if a.command=='list':
        for s in inv.verify_local()['branches']:print(s['case'],s['cells'],len(s['markers']),s['dimensions'])
    elif a.command=='prepare':
        if not all((a.case,a.source_root,a.output)):p.error('prepare requires case, source-root and output')
        print(prepare(a.source_root,a.output,a.case))
    else:
        if a.bundle is None:p.error('verify requires bundle')
        s,f=verify_bundle(a.bundle);print(s['case'],len(f),'files verified')


if __name__=='__main__':main()
