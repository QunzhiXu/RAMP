from pathlib import Path
import shutil,json,hashlib
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'publication_display_20260922'
shutil.copytree(OLD/'assets',HERE/'assets')
p=OLD/'rendering_helpers.py';s=p.read_text()
a=s.index('def ridge(');b=s.index('\n\ndef scatter(',a)
s=s[:a]+'''def ridge(ax, values, meta, method, marker):
    values=np.asarray(values,dtype=float)
    assert np.isfinite(values).all()
    old_low,old_high=np.quantile(values,[.005,.995]);assert old_high>old_low
    # Retain the original bin width and bandwidth; extend the grid rather than
    # spreading the same 160 bins over a larger range.
    dx=(old_high-old_low)/160
    left=int(np.floor((values.min()-old_low)/dx))-26
    right=int(np.ceil((values.max()-old_low)/dx))+26
    edges=old_low+np.arange(left,right+1,dtype=float)*dx
    low,high=edges[0],edges[-1];centers=(edges[:-1]+edges[1:])/2
    datasets=meta.dataset.to_numpy();units=meta.unit_key.to_numpy()
    palette=dict(zip(meta.dataset.unique(),DS_COLORS))
    groups=[(d,u) for d in palette for u in sorted(set(units[datasets==d]))]
    level=0.;prev=None
    for i,(d,u) in enumerate(reversed(groups)):
        if prev is not None and prev!=d:level+=.36
        vals=values[(datasets==d)&(units==u)]
        hist,_=np.histogram(vals,bins=edges)
        assert hist.sum()==len(vals)
        smooth=gaussian_filter1d(hist.astype(float),sigma=6,mode='constant',truncate=4)
        if smooth.max()>0:smooth/=smooth.max()
        assert smooth[0]==0 and smooth[-1]==0
        # Eliminate redundant vertices only inside exactly flat runs. All
        # non-flat curve vertices and the full density array are preserved.
        changed=np.diff(smooth)!=0
        keep=np.r_[True,changed] | np.r_[changed,True]
        xx=centers[keep];yy=level+.70*smooth[keep]
        ax.fill_between(xx,level,yy,color=palette[d],alpha=.82,linewidth=0,zorder=2*i+1)
        ax.plot(np.r_[low,xx,high],np.r_[level,yy,level],color='#46505A',lw=.35,zorder=2*i+2)
        AUDIT.append(dict(benchmark='melanoma',method=method,marker=marker,
            smoothing_sigma_bins=6,dataset=d,unit_key=u,n_cells=len(vals),
            display_low=float(low),display_high=float(high),
            n_below_display=int((vals<low).sum()),n_above_display=int((vals>high).sum()),
            old_display_low=float(old_low),old_display_high=float(old_high),
            old_n_below=int((vals<old_low).sum()),old_n_above=int((vals>old_high).sum()),
            histogram_count=int(hist.sum()),bins=len(hist),bin_width=float(dx),bandwidth_expression_units=float(6*dx),
            frozen_value_min=float(values.min()),frozen_value_max=float(values.max()),
            endpoint_density_left=float(smooth[0]),endpoint_density_right=float(smooth[-1]),
            density_sha256=hashlib.sha256(smooth.tobytes()).hexdigest(),ridge_color=palette[d]))
        level+=.57;prev=d
    ax.set_xlim(low,high);ax.set_ylim(-.09,level-.57+.70+.04)
    if low<0<high:ax.axvline(0,color='#666666',lw=.35,ls=(0,(1.5,2.2)),alpha=.7,zorder=100)
    ax.set_yticks([])
    ticks=MaxNLocator(3).tick_values(low,high);ax.set_xticks(ticks[(ticks>=low)&(ticks<=high)])
    ax.xaxis.set_major_formatter(FuncFormatter(tick))
    frame(ax);ax.tick_params(axis='x',labelsize=5.5,length=1.3,pad=1.0)
'''+s[b:]
(HERE/'rendering_helpers.py').write_text(s)
s=(OLD/'build_figure.py').read_text()
needle="pd.testing.assert_frame_equal(pd.DataFrame(h.AUDIT)[prior_ridge.columns],prior_ridge,check_exact=False,rtol=1e-12,atol=1e-12)"
assert needle in s
s=s.replace(needle,"""current_ridge=pd.DataFrame(h.AUDIT)
    keys=['benchmark','method','marker','dataset','unit_key','n_cells','smoothing_sigma_bins']
    pd.testing.assert_frame_equal(current_ridge[keys],prior_ridge[keys])
    assert current_ridge.n_below_display.eq(0).all() and current_ridge.n_above_display.eq(0).all()
    assert current_ridge.histogram_count.eq(current_ridge.n_cells).all()
    assert np.allclose(current_ridge.bin_width,(prior_ridge.display_high-prior_ridge.display_low)/160,rtol=2e-6,atol=1e-9)
    assert current_ridge.endpoint_density_left.eq(0).all() and current_ridge.endpoint_density_right.eq(0).all()""")
s=s.replace("panel_b_all_cells_retained=True", "panel_b_all_cells_retained=True,panel_b_complete_density_support=True,panel_b_smoothing_sigma_bins=6,panel_b_bandwidth_reference_quantiles=[.005,.995],panel_b_bandwidth_reference_bins=160,panel_b_range='Full frozen-cohort extrema plus at least four smoothing bandwidths at each end',panel_b_original_palette_and_layering_retained=True")
(HERE/'build_figure.py').write_text(s)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=[p for p in OLD.iterdir() if p.is_file()]+[p for p in (OLD/'assets').iterdir() if p.is_file()]
(HERE/'OLD_SOURCE_HASHES.json').write_text(json.dumps({str(p):sha(p) for p in old},indent=2)+'\n')
print('Prepared a separate Figure 1 full-range version')
