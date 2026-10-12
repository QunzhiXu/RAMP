from pathlib import Path
import json,re,hashlib,ast
HERE=Path(__file__).resolve().parent;SUP=HERE.parent;ROOT=SUP.parent
CASES=['Bone marrow','TNBC','Melanoma three-platform','Lung cancer benchmark','Lung HLA-ABC story','Melanoma story','HNSCC story']
TARGETS={case:(['RAMP'] if i==2 else ['Harmony-direct','RAMP'] if i>=4 else ['Harmony-direct']) for i,case in enumerate(CASES)}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pub=json.loads((SUP/'CURRENT_PUBLICATION.json').read_text())
protected=[SUP/'CURRENT_PUBLICATION.json',SUP/'README.md',ROOT/'Main Figure/README.md']
protected += [p for p in (ROOT.parent/'bioarxiv_Ver2').rglob('*') if p.is_file()]
protected += [SUP/c/'CURRENT_PREVIEW.json' for c in CASES]
protected += [Path(p['source']) for p in pub]
(HERE/'PROTECTED_HASHES.json').write_text(json.dumps({str(p):sha(p) for p in protected},indent=2)+'\n')
(HERE/'INPUT_PUBLICATION.json').write_text(json.dumps(pub,indent=2)+'\n')
(HERE/'TARGETS.json').write_text(json.dumps(TARGETS,indent=2)+'\n')
source=(SUP/'ridge_full_range_update_20260922/ridge_common.py').read_text()
source=source.replace("STYLE=dict(","TARGETS=json.loads((Path(__file__).parent/'TARGETS.json').read_text())\n\nSTYLE=dict(")
source=source.replace('    level,previous=0.,None','    level,previous=0.,None\n    curve_records=[]\n    maximum=np.zeros(len(centers))')
source=source.replace("        assert density[0]==0 and density[-1]==0, 'Density must return to baseline'",
"        assert density[0]==0 and density[-1]==0, 'Density must return to baseline'\n        maximum=np.maximum(maximum,density)")
source=source.replace('        level+=.57;previous=d','        curve_records.append((audit[-1],vals,density))\n        level+=.57;previous=d')
needle='    ax.set_xlim(low,high);ax.set_ylim(-.09,level-.57+.70+.04);ax.set_yticks([])'
replacement='''    full_low,full_high=low,high
    compact=method in TARGETS[cfg['case']]
    if compact:
        occupied=centers[np.flatnonzero(maximum>=.005)[[0,-1]]]
        margin=(occupied[1]-occupied[0])*.04
        low=max(full_low,float(occupied[0]-margin));high=min(full_high,float(occupied[1]+margin))
        # Already-approved main-figure override: identical melanoma RAMP SOX10 view.
        if cfg['case']=='Melanoma three-platform' and method=='RAMP' and marker=='SOX10':low=-1.5
    for record,vals,density in curve_records:
        record.update(histogram_grid_low=float(full_low),histogram_grid_high=float(full_high),
            display_low=float(low),display_high=float(high),
            n_below=int((vals<low).sum()),n_above=int((vals>high).sum()),
            display_window_changed=compact,
            display_rule='Union of displayed-sample density >=0.5% of own peak, 4% span padding; S3 RAMP SOX10 left=-1.5' if compact else 'Original full range',
            display_endpoint_density_left=float(np.interp(low,centers,density)),
            display_endpoint_density_right=float(np.interp(high,centers,density)))
    ax.set_xlim(low,high);ax.set_ylim(-.09,level-.57+.70+.04);ax.set_yticks([])'''
assert needle in source;source=source.replace(needle,replacement)
source=source.replace('    assert a.n_below.eq(0).all() and a.n_above.eq(0).all()',
                      '    assert a.n_below.ge(0).all() and a.n_above.ge(0).all()')
source=source.replace("version='ridge_full_range_20260922',revision_scope='Complete ridge density support and axis margins; original bandwidth, selection and palettes retained'",
"version='ridge_compact_preview_20260924',revision_scope='Requested rows only: compact view; density estimates, all displayed cells, bandwidth and palettes retained'")
source=source.replace('ridge_full_distribution=True','ridge_full_distribution=False,ridge_full_density_estimation=True,preview_only=True,manuscript_updated=False')
ast.parse(source);(HERE/'ridge_common.py').write_text(source)
plan=[]
for n,case in enumerate(CASES,1):
    orig=SUP/case/'ridge_full_range_20260922';out=HERE/f'S{n}';out.mkdir(exist_ok=False)
    s=(orig/'build_figure.py').read_text()
    s=s.replace('HERE = Path(__file__).resolve().parent',f'HERE = Path({str(orig)!r})\nOUT = Path(__file__).resolve().parent')
    s=s.replace("str(HERE / '.cache/matplotlib')","str(OUT / '.cache/matplotlib')")
    s=s.replace("sys.path.insert(0,str(HERE.parents[1]/'ridge_full_range_update_20260922'))","sys.path.insert(0,str(OUT.parent))")
    s=s.replace("'Raw':'Raw (assay-transformed)'","'Raw':'Raw'")
    s=s.replace("out=HERE/'output'","out=OUT/'output'").replace(".to_csv(HERE/",".to_csv(OUT/")
    s=re.sub(r"\(HERE/('[^']+')\)\.write_text",r"(OUT/\1).write_text",s)
    s=s.replace('p.relative_to(HERE)','p.relative_to(OUT)')
    # The inputs remain at their read-only original paths; every output is local.
    assert not re.search(r"\(HERE/[^\n]*\.write_text|\.to_csv\(HERE/",s)
    ast.parse(s);(out/'build_figure.py').write_text(s)
    plan.append(dict(number=f'S{n}',case=case,source=str(orig),output=str(out),
                     old_publication=pub[n-1]['source'],targets=TARGETS[case],source_script_sha256=sha(orig/'build_figure.py')))
(HERE/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
print('Prepared seven isolated preview builders; manuscript and publication pointers remain read-only.')
