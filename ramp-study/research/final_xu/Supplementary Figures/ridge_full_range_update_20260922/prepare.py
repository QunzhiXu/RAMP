from pathlib import Path
import json, shutil, os
HERE=Path(__file__).resolve().parent
SUP=HERE.parent
prior=SUP/'ridge_style_update_20260922'
source=(prior/'ridge_common.py').read_text()
source=source.replace("bins=160,sigma_bins=6,range_quantiles=[.005,.995],range_reference='Full frozen cohort, unchanged from prior figures',", "bins='Extended grid at the original bin width',sigma_bins=6,range_quantiles=None,\n           bandwidth_reference_quantiles=[.005,.995],bandwidth_reference_bins=160,\n           gaussian_truncate_sigma=4,padding_bins=26,\n           range_reference='Full frozen cohort minimum to maximum, plus at least four Gaussian bandwidths at both ends',")
source=source.replace("    low,high=np.quantile(v,[.005,.995]);assert high>low", """    v=np.asarray(v,dtype=float)
    assert np.isfinite(v).all()
    old_low,old_high=np.quantile(v,[.005,.995]);assert old_high>old_low
    dx=(old_high-old_low)/160
    # Retain both the old bin width and its grid origin. This preserves the
    # smoothing bandwidth in expression units; only support is extended.
    left=int(np.floor((v.min()-old_low)/dx))-26
    right=int(np.ceil((v.max()-old_low)/dx))+26
    assert 0 < right-left < 1000000, 'Unexpected range; inspect without truncating'
    edges=old_low+np.arange(left,right+1,dtype=float)*dx
    low,high=edges[0],edges[-1]
    centers=(edges[:-1]+edges[1:])/2""")
source=source.replace("counts,edges=np.histogram(vals,bins=160,range=(low,high))", "counts,_=np.histogram(vals,bins=edges)\n        assert counts.sum()==len(vals), 'Every displayed cell must enter the histogram'")
source=source.replace("sigma=6,mode='constant')", "sigma=6,mode='constant',truncate=4)")
source=source.replace("        x=(edges[:-1]+edges[1:])/2;y=level+.70*density", "        assert density[0]==0 and density[-1]==0, 'Density must return to baseline'\n        x=centers;y=level+.70*density")
source=source.replace("# Keep every density bin, including flat tails. Extend the endpoint values\n        # by half a bin to the display boundaries so each outline reaches the frame.","# Fully evaluated tails reach the baseline before the frame. No artificial\n        # dropping of nonzero endpoint densities or quantile clipping is used.")
source=source.replace("bins=160,sigma_bins=6,peak_height_normalized=True", "bins=len(counts),sigma_bins=6,bin_width=float(dx),bandwidth_expression_units=float(6*dx),\n            histogram_count=int(counts.sum()),endpoint_density_left=float(density[0]),endpoint_density_right=float(density[-1]),\n            frozen_value_min=float(v.min()),frozen_value_max=float(v.max()),\n            old_display_low=float(old_low),old_display_high=float(old_high),\n            old_n_below=int((vals<old_low).sum()),old_n_above=int((vals>old_high).sum()),\n            peak_height_normalized=True")
source=source.replace("version='ridge_uniform_20260922',revision_scope='Ridge palette, selected displayed units, outlines and local legends only',", "version='ridge_full_range_20260922',revision_scope='Complete ridge density support and axis margins; original bandwidth, selection and palettes retained',\n        ridge_range_quantiles=None,ridge_bins='Extended at fixed original bin width',ridge_sigma_bins=6,\n        ridge_bandwidth_reference_quantiles=[.005,.995],ridge_bandwidth_reference_bins=160,\n        ridge_full_distribution=True,ridge_density_padding_bins=26,")
source=source.replace("    assert len(a)==18*cfg['ridge_units']", "    assert len(a)==18*cfg['ridge_units']\n    assert a.n_below.eq(0).all() and a.n_above.eq(0).all()\n    assert a.histogram_count.eq(a.n_cells).all()\n    assert a.endpoint_density_left.eq(0).all() and a.endpoint_density_right.eq(0).all()")
source=source.replace("x=centers;y=level+.70*density", "changed=np.diff(density)!=0\n        keep=np.r_[True,changed] | np.r_[changed,True]\n        x=centers[keep];y=level+.70*density[keep]")
(HERE/'ridge_common.py').write_text(source)
exec(compile(source,str(HERE/'ridge_common.py'),'exec'),ns:={})
style=ns['STYLE'];sha=ns['sha']
plan=[]
for item in json.loads((prior/'REVISION_PLAN.json').read_text()):
    old=Path(item['new']);new=old.parent/'ridge_full_range_20260922'
    new.mkdir(exist_ok=False)
    for p in old.iterdir():
        if p.is_symlink(): (new/p.name).symlink_to(os.path.relpath(p.resolve(),new),target_is_directory=p.is_dir())
        elif p.is_file() and (p.suffix=='.json' or p.name in ['extract_fastmnn.R','ridge_unit_selection.csv']):
            if p.name not in ['MANIFEST.json','QA.json','RIDGE_CONFIG.json','STATUS.json']:
                shutil.copy2(p,new/p.name)
    cfg=json.loads((old/'RIDGE_CONFIG.json').read_text());cfg['style']=style;cfg['previous_ridge_folder']=str(old)
    (new/'RIDGE_CONFIG.json').write_text(json.dumps(cfg,indent=2)+'\n')
    builder=(old/'build_figure.py').read_text().replace("'ridge_style_update_20260922'","'ridge_full_range_update_20260922'")
    needle="def text(fig,x,y,label,size=8.5,weight='bold',color=INK,ha='left',va='top'):\n"
    assert needle in builder
    builder=builder.replace(needle,needle+"    label=label.replace('Harmony-direct','Harmony')\n")
    (new/'build_figure.py').write_text(builder)
    plan.append(dict(name=item['name'],old=str(old),new=str(new),png=item['png'],
       old_hashes={str(p):sha(p) for p in [old/'build_figure.py',old/'MANIFEST.json',old/'ridge_display_audit.csv',*list((old/'output').glob('*.png')),*list((old/'output/pdf').glob('*.pdf'))]}))
(HERE/'REVISION_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
shutil.copy2(prior/'render_all.py',HERE/'render_all.py')
print('Prepared',len(plan),'new versions')
