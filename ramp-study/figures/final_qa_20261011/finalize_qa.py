"""Seal the fixed, completed QA evidence; never accepts arbitrary run/report input."""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import fitz

from inspect_assets import HERE, ROOT, STUDY, PAPER, rec, sha, read, write, prior, source_index

INSPECTION=HERE/'runs/inspection_20261011'
BRIDGE=HERE/'runs/bridges_20261011_r3'
BRIDGE_SHA='42665a58b8de0c075bce70ec650151eb0aea99abd0f7f5b0458b02546b2dc1f9'
BUNDLED=Path('/Users/qxu6/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python')


def require(ok, message):
    if not ok:raise ValueError(message)


def pdf_findings():
    records=[]
    with fitz.open(PAPER) as pdf:
        for n in range(49,62):
            page=pdf[n-1]
            lines=[{'text':''.join(s['text'] for s in line['spans']), 'bbox':list(line['bbox'])}
                   for b in page.get_text('dict',clip=fitz.INFINITE_RECT())['blocks'] for line in b.get('lines',[])]
            stamps=[x for x in lines if x['text'].strip() in
                    ('available under a','CC-BY-NC-ND 4.0 International license')]
            titles=[x for x in lines if x['bbox'][1]>=65 and x['bbox'][1]<84 and
                    x['text'].strip() not in ('.','available under a','CC-BY-NC-ND 4.0 International license')]
            intersections=[{'title':a,'stamp':b,'intersection':list(fitz.Rect(a['bbox'])&fitz.Rect(b['bbox']))}
                           for a in titles for b in stamps if fitz.Rect(a['bbox']).intersects(fitz.Rect(b['bbox']))]
            overflow=[x for x in lines if x['text'].strip() and x['bbox'][3]>page.rect.height]
            records.append({'page':n,'figure':f'S{n-48}','title_header_intersections':intersections,
                            'text_crossing_bottom':overflow,'page_height_points':page.rect.height})
    require([x['page'] for x in records if x['title_header_intersections']]==list(range(49,58)),
            'Unexpected header-overlap evidence')
    require([x['page'] for x in records if x['text_crossing_bottom']]==[56],
            'Unexpected caption-overflow evidence')
    return records


def finalize(output):
    out=Path(output).absolute()
    require(out.is_relative_to(HERE/'validation') and not out.exists() and not out.is_symlink() and
            not any(p.is_symlink() for p in out.parents),'Output must be new and physical under validation/')
    require(shutil.disk_usage(HERE).free>20_000_000_000,'Insufficient disk space')
    source_index()
    require(sha(BRIDGE/'report.json')==BRIDGE_SHA,'Unrecognized numeric report')
    sealed=read(BRIDGE/'COMPLETE.json')
    require(sealed['report']==rec(BRIDGE/'report.json') and sealed['request']==rec(BRIDGE/'REQUEST.json'),
            'Bridge seal changed')
    request=read(BRIDGE/'REQUEST.json')
    require(request['controller']==rec(HERE/'check_bridges.py') and request['asset_inspector']==rec(HERE/'inspect_assets.py'),
            'QA computation code changed')
    numerical=read(BRIDGE/'report.json')
    require(len(numerical['checks'])==20,'Numerical check count differs')
    for key,c in numerical['checks'].items():
        if key!='Figure7C':
            require(math.isfinite(c['max_abs_difference']) and
                    0<=c['max_abs_difference']<=c['absolute_tolerance']<=1e-14,'Numerical tolerance differs')
    for path,record in numerical['sources'].items():
        require(rec(ROOT/path)==record,'Consumed bridge source changed: '+path)
    assets=read(INSPECTION/'assets.json')
    require(sha(INSPECTION/'assets.json')==read(INSPECTION/'COMPLETE.json')['assets_sha256'],'Asset inspection changed')
    require(all(not x['text_outside_page'] for x in assets),'Local asset has outside-page text')
    manual=read(HERE/'VISUAL_REVIEW.json')
    require(len(manual['figures'])==20 and {x['label'] for x in manual['figures']}=={x['label'] for x in assets},
            'Manual coverage differs')
    require([x['id'] for x in manual['findings']]==['LAYOUT-1','LAYOUT-2'],'Findings must not be omitted')
    baseline=read(INSPECTION/'PRIOR.json')
    require(prior()==baseline,'Pre-existing small study files changed')
    out.parent.mkdir(exist_ok=True)
    out.mkdir()
    try:
        write(out/'REQUEST.json',{'controller':rec(Path(__file__)),'bridge_report':rec(BRIDGE/'report.json'),
            'manual_review':rec(HERE/'VISUAL_REVIEW.json'),'paper':rec(PAPER),
            'scope':'asset/visual QA and selected numeric bridges; no figure rebuilding or approval'})
        tests=[]
        for label,python in [('bundled',BUNDLED),('system',Path(sys.executable))]:
            result=subprocess.run([str(python),'-B','-m','unittest','-v','test_bridges'],cwd=HERE,
                                  capture_output=True,text=True,check=False)
            with (out/f'tests_{label}.txt').open('x') as f:f.write(result.stdout+result.stderr)
            require(result.returncode==0 and 'Ran 17 tests' in result.stderr,'Bridge unit tests failed')
            tests.append({'interpreter':str(python),'tests':17,'returncode':result.returncode})
        geometry=pdf_findings()
        write(out/'published_layout_geometry.json',geometry)
        code={p.name:rec(p) for p in HERE.glob('*.py')}
        preview_records={str(p.relative_to(INSPECTION)):rec(p) for p in (INSPECTION/'previews').glob('*.png')}
        require(len(preview_records)==40,'Preview coverage differs')
        require(prior()==baseline,'Prior files changed during finalization')
        source_index()
        for path,record in numerical['sources'].items():
            require(rec(ROOT/path)==record,'Bridge source changed during finalization')
        status='internally_checked_with_layout_findings_pending_external_audit'
        report={'status':status,'local_assets_viewed':20,'published_figure_pages_viewed':20,
                'local_pdf_text_outside_page':0,'numeric_bridge_checks':20,
                'plot_table_rows_checked':2671,'full_hnscc_coordinate_rows_checked':94488,
                'max_table_abs_difference':max(c.get('max_abs_difference',0) for c in numerical['checks'].values()),
                'numeric_report':rec(BRIDGE/'report.json'),'asset_inspection':rec(INSPECTION/'assets.json'),
                'manual_review':rec(HERE/'VISUAL_REVIEW.json'),'findings':manual['findings'],
                'pdf_geometry':rec(out/'published_layout_geometry.json'),'preview_records':preview_records,
                'tests':tests,'current_code':code,'original_indexes':numerical['original_indexes'],
                'prior_small_study_files_unchanged':len(baseline),
                'module_logical_bytes_at_report':sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()),
                'free_bytes_at_report':shutil.disk_usage(HERE).free,
                'not_claimed':manual['not_claimed'],'external_audit_approval':False,
                'manuscript_or_scientific_files_modified':False}
        write(out/'report.json',report)
        write(out/'COMPLETE.json',{'status':status,'report':rec(out/'report.json'),'request':rec(out/'REQUEST.json')})
        return {'status':status,'report':str(out/'report.json')}
    except BaseException as exc:
        write(out/'FAILED.json',{'error':repr(exc)});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    print(json.dumps(finalize(a.out),indent=2))
