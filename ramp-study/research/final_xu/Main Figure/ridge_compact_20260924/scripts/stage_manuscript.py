from pathlib import Path
import hashlib,json,shutil,numpy as np,pandas as pd,fitz,difflib
ROOT=Path(__file__).resolve().parents[1];LIVE=ROOT.parents[1].parent/'bioarxiv_Ver2'
STAGE=ROOT/'stage';OLD=ROOT.parent/'ridge_full_range_20260922'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
new=pd.read_csv(ROOT/'ridge_display_audit.csv');old=pd.read_csv(OLD/'ridge_display_audit.csv')
assert len(new)==len(old)==552
for key in ['n_cells','bin_width','density_sha256','smoothing_sigma_bins']:
 pd.testing.assert_series_equal(new[key],old[key])
assert new.histogram_count.eq(new.n_cells).all()
assert new.histogram_grid_endpoint_left.eq(0).all() and new.histogram_grid_endpoint_right.eq(0).all()
for rel in ['median_shift_box_statistics.csv','panel_d_display_overflow.csv']:assert sha(ROOT/rel)==sha(OLD/rel)
a=fitz.open(LIVE/'figure/figure1_ramp_overview.pdf');b=fitz.open(ROOT/'figure_1_melanoma.pdf');assert a[0].rect==b[0].rect
pa=a[0].get_pixmap(dpi=180,alpha=False);pb=b[0].get_pixmap(dpi=180,alpha=False)
aa=np.frombuffer(pa.samples,np.uint8).reshape(pa.height,pa.width,3);bb=np.frombuffer(pb.samples,np.uint8).reshape(pb.height,pb.width,3)
diff=np.any(aa!=bb,axis=2);mask=np.zeros(diff.shape,bool)
for row in range(4):
 y=16+25*row
 mask[int((y-.7)/25.4*180):int((y+24.6)/25.4*180)+1,int(63.5/25.4*180):int(118/25.4*180)+1]=True
assert not diff[~mask].any(),int(diff[~mask].sum())
for word in b[0].get_text('words'):
 x0,y0,x1,y1=word[:4];assert 0<=x0<=x1<=b[0].rect.width and 0<=y0<=y1<=b[0].rect.height
b[0].get_pixmap(dpi=190).save(ROOT/'renders/figure1_updated.png')
qa=dict(ridge_panels=12,ridge_curves=552,all_60000_cells_used_for_density=True,all_46_sample_units_retained=True,
 density_arrays_bitwise_identical=True,bandwidth_unchanged=True,umaps_and_panels_a_c_d_pixel_identical=True,
 panel_d_statistics_identical=True,all_figure_text_inside_page=True,ramp_sox10_left_limit=-1.5,
 display_tail_counts=new.groupby(['method','marker'])[['n_cells','n_below_display','n_above_display']].sum().reset_index().to_dict('records'))
(ROOT/'QA.json').write_text(json.dumps(qa,indent=2)+'\n')
for rel in ['main.tex','bibliography.bib','ramp_numeric.bst','ramp_harvard.bst']:shutil.copy2(LIVE/rel,STAGE/rel)
(STAGE/'figure').mkdir()
for p in (LIVE/'figure').iterdir():
 if p.name=='figure1_ramp_overview.pdf':shutil.copy2(ROOT/'figure_1_melanoma.pdf',STAGE/'figure'/p.name)
 else:(STAGE/'figure'/p.name).symlink_to(p,target_is_directory=p.is_dir())
(STAGE/'supplementary_data').symlink_to(LIVE/'supplementary_data',target_is_directory=True)
p=STAGE/'main.tex';s=p.read_text();original=s
before='''Each ridge is peak-normalized; method-specific axes include the full
range and smoothing tails.'''
after=r'''Ridges are peak-normalized, with method-specific display windows (Methods).'''
assert s.count(before)==1;s=s.replace(before,after)
a=s.index(r'\paragraph{Ridge plots.}');z=s.index(r'\paragraph{UMAP.}',a)
s=s[:a]+r'''\paragraph{Ridge plots.} All cells enter the histograms
(Fig.~\ref{fig:overview}B; Supplementary Figs.~\ref{fig:s1}--\ref{fig:s7}). For each method--marker,
bin width is the pooled 0.5th--99.5th percentile span divided by 160.
Gaussian smoothing uses sigma 6 bins, truncation at four standard deviations
and constant boundary handling. Grids cover all observed values plus at least
four bandwidths at each end. Curves are peak-normalized; height does not
represent cell number. Figure~\ref{fig:overview}B shows the union of sample-curve
ranges above 0.5\% of peak height, padded by 4\% of this span within the grid;
the RAMP SOX10 left limit is $-1.5$. Each window retains at least 99.8\% of
observed values; all values contribute to density estimation. Supplementary
ridge axes show the full grid. Raw values in Figure~\ref{fig:overview}B use
$\operatorname{asinh}(x/1)$; supplementary transformations are specified in
their legends. Each method retains its output scale, so horizontal values
are not comparable across method rows.

'''+s[z:]
p.write_text(s)
(ROOT/'manuscript_changes.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='before/main.tex',tofile='after/main.tex')))
cap=s[s.index(r'\paperfig{figure1_ramp_overview}') : s.index('}{fig:overview}')+len('}{fig:overview}')]
(ROOT/'FIGURE1_CAPTION.tex').write_text(cap+'\n')
print(json.dumps({'figure_verified':True,'only_ridge_pixels_changed':True,'manuscript_changes':'Figure 1 caption and Ridge plots paragraph'},indent=2))
