from pathlib import Path
import hashlib,json,shutil,ast
import pandas as pd
import numpy as np
import fitz
HERE=Path(__file__).resolve().parent;MAIN=HERE.parent;OLD=MAIN/'publication_display_20260922';LIVE=MAIN.parent.parent/'bioRxiv_Ver1';STAGE=HERE/'manuscript_stage'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for p,want in json.loads((HERE/'OLD_SOURCE_HASHES.json').read_text()).items():assert sha(Path(p))==want,p
manifest=json.loads((HERE/'manifest.json').read_text())
for p,want in manifest['all_source_hashes'].items():assert sha(Path(p))==want,p
old=pd.read_csv(OLD/'ridge_display_audit.csv');new=pd.read_csv(HERE/'ridge_display_audit.csv')
assert len(new)==12*46==len(old)
keys=['benchmark','method','marker','dataset','unit_key','n_cells','smoothing_sigma_bins']
pd.testing.assert_frame_equal(new[keys],old[keys])
assert new.histogram_count.eq(new.n_cells).all()
assert new.n_below_display.eq(0).all() and new.n_above_display.eq(0).all()
assert new.endpoint_density_left.eq(0).all() and new.endpoint_density_right.eq(0).all()
assert np.allclose(new.bin_width,(old.display_high-old.display_low)/160,rtol=2e-6,atol=1e-9)
assert ((new.frozen_value_min-new.display_low)>=4*new.bandwidth_expression_units).all()
assert ((new.display_high-new.frozen_value_max)>=4*new.bandwidth_expression_units).all()
for rel in ['median_shift_box_statistics.csv','panel_d_display_overflow.csv']:assert sha(HERE/rel)==sha(OLD/rel)
a=fitz.open(OLD/'figure_1_melanoma.pdf');b=fitz.open(HERE/'figure_1_melanoma.pdf')
assert len(a)==len(b)==1 and a[0].rect==b[0].rect
p1=a[0].get_pixmap(dpi=180,alpha=False);p2=b[0].get_pixmap(dpi=180,alpha=False)
im1=np.frombuffer(p1.samples,np.uint8).reshape(p1.height,p1.width,3);im2=np.frombuffer(p2.samples,np.uint8).reshape(p2.height,p2.width,3)
diff=np.any(im1!=im2,axis=2)
mask=np.zeros(diff.shape,bool)
# Only ridge axes and their x-axis tick labels may change.
for row in range(4):
    y=16+25*row
    mask[int((y-.7)/25.4*180):int((y+24.6)/25.4*180)+1,int(63.5/25.4*180):int(118/25.4*180)+1]=True
assert not diff[~mask].any(),f'{diff[~mask].sum()} changed pixels outside ridge region'
for x0,y0,x1,y1,*_ in b[0].get_text('words'):assert 0<=x0<=x1<=b[0].rect.width and 0<=y0<=y1<=b[0].rect.height
qa=dict(ridge_panels=12,ridge_curves=552,all_values_included=True,all_endpoints_zero=True,bandwidth_preserved=True,same_60000_cells_46_units=True,umaps_and_panels_a_c_d_pixel_identical=True,panel_d_statistics_unchanged=True,old_outputs_and_source_data_unchanged=True,all_pdf_text_within_page=True)
(HERE/'QA.json').write_text(json.dumps(qa,indent=2)+'\n')
ranges=new.groupby(['method','marker'],sort=False).first()[['old_display_low','old_display_high','display_low','display_high','bins','bandwidth_expression_units']]
ranges.to_csv(HERE/'range_audit.csv')
(HERE/'tmp/pdfs').mkdir(parents=True,exist_ok=True);b[0].get_pixmap(dpi=170).save(HERE/'tmp/pdfs/figure_1.png')
STAGE.mkdir(exist_ok=False)
for rel in ['template.tex','bibliography.bib','ramp_harvard.bst','FIGURE1_CAPTION.md','README.md','REVISION_NOTES.md']:
    shutil.copy2(LIVE/rel,STAGE/rel)
shutil.copytree(LIVE/'Main_Overleaf/figure',STAGE/'figure')
shutil.copy2(HERE/'figure_1_melanoma.pdf',STAGE/'figure/figure1_ramp_overview.pdf')
p=STAGE/'template.tex';s=p.read_text()
start=s.index(r'\paragraph{Ridge plots.}');end=s.index(r'\paragraph{UMAP.}',start)
s=s[:start]+r'''\paragraph{Ridge plots.} For Figure~\ref{fig:overview}B and Figures S1--S7,
all displayed values enter the histograms. For each method~$\times$~marker
combination, bin width is the pooled 0.5th--99.5th percentile span divided
by 160, and Gaussian smoothing uses a standard deviation of 6 bins.
The histogram grid and axes cover the complete frozen-cohort range plus
at least four smoothing bandwidths at each end; the Gaussian kernel is
truncated at four standard deviations, with constant boundary handling.
Each curve is normalized to its own maximum, so ridge height does not encode
cell number. Figure~\ref{fig:overview}B Raw ridges use
$\operatorname{asinh}(x/1)$ for display; platform-specific transformations
for Figures S1--S7 are listed in the supplementary legends. Aligned rows
retain each method's own output scale, so ridge positions are not comparable
between rows.

'''+s[end:]
needle='peak-normalized, with method-specific horizontal scales.';assert s.count(needle)==1
s=s.replace(needle,'peak-normalized, with method-specific horizontal scales covering the full\nobserved range and smoothing tails.')
p.write_text(s)
p=STAGE/'FIGURE1_CAPTION.md';s=p.read_text();assert needle in s;p.write_text(s.replace(needle,'peak-normalized, with method-specific horizontal scales covering the full\nobserved range and smoothing tails.'))
p=STAGE/'README.md';s=p.read_text().replace('Main Figure/publication_display_20260922/','Main Figure/ridge_full_range_20260922/');p.write_text(s+'\n\n## Figure 1 density-tail correction (2026-09-22)\n\nFigure 1B now displays complete ridge distributions with padded axes and the original expression-unit bandwidths. All 60,000 cells and 46 sample units remain. Other panels, UMAPs, palettes and numerical scores are unchanged. Caption, Methods and Main_Overleaf.zip are synchronized; supplementary files retain the preceding full-range correction.\n')
p=STAGE/'REVISION_NOTES.md';p.write_text(p.read_text()+'\n\n## Figure 1B full-density repair (2026-09-22)\n\nRemoved quantile truncation from histogram support, preserved original expression-unit smoothing bandwidths, and padded both ends by at least four bandwidths. Twelve method-marker panels / 552 ridges pass full-count and zero-endpoint checks. Outside the ridge axes/ticks, Figure 1 pixels are unchanged. Caption and Methods updated; no alignment or metrics rerun.\n')
# Local figure-folder captions and entry points are installed only after final checks.
text=(MAIN/'FIGURE_CAPTION.md').read_text().replace('publication_display_20260922','ridge_full_range_20260922')
old_en='Ridges use pooled method–marker 0.5th–99.5th percentile display ranges, 160-bin histograms and Gaussian smoothing with sigma 6 bins;'
new_en='Ridges include all values on a grid spanning the full frozen-cohort range plus at least four Gaussian smoothing bandwidths at each end. Bin width is the pooled method–marker 0.5th–99.5th percentile span divided by 160, preserving the original bandwidth of 6 bins;'
assert old_en in text;text=text.replace(old_en,new_en)
old_zh='曲线使用 pooled 0.5%–99.5% 范围、160-bin 直方图及 sigma=6 bins 的高斯平滑；'
new_zh='曲线使用完整冻结数值范围，两端各保留至少 4 个平滑带宽的空间；bin 宽度仍为 pooled 0.5%–99.5% 跨度除以 160，高斯 sigma 仍为 6 bins，保持原实际带宽；'
assert old_zh in text;text=text.replace(old_zh,new_zh)
(HERE/'FIGURE_CAPTION_NEXT.md').write_text(text)
p=MAIN/'README.md';s=p.read_text().replace('publication_display_20260922','ridge_full_range_20260922');s+='\nFigure 1B 密度尾部已修复：完整取值进入直方图，扩展轴范围及平滑网格，保留原实际带宽。12 个 ridge 面板 / 552 条样本曲线无数值遗漏，两端回到基线；UMAP 和 A/C/D 像素相同。旧版在 publication_display_20260922 中保留。\n';(HERE/'README_NEXT.md').write_text(s)
(HERE/'README.md').write_text('# Figure 1 — full ridge distributions\n\nCurrent outputs: figure_1_melanoma.pdf / .png / .svg; isolated panels are in Panel A–D. Rebuild: `python3 build_figure.py`.\n\nOnly Panel B density support and horizontal ranges changed. Bin width remains (q99.5−q0.5)/160; Gaussian sigma remains 6 bins. Full-cohort extrema plus at least 4 bandwidths on either side define the grid. All 60,000 cells and all 46 units are retained; every ridge reaches baseline without omitting tail observations. Redundant flat-segment vertices are omitted from vector export without changing the curve.\n\nQA.json checks 12 panels / 552 curves, exact pixels outside ridge axes, input hashes and unchanged C/D statistics. range_audit.csv and ridge_display_audit.csv record limits and cell counts. Old version remains at ../publication_display_20260922.\n')
shutil.copytree(LIVE/'Main_Overleaf',HERE/'Main_Overleaf')
for rel in ['template.tex','figure/figure1_ramp_overview.pdf']:shutil.copy2(STAGE/rel,HERE/'Main_Overleaf'/rel)
p=HERE/'Main_Overleaf/README.md';s=p.read_text();s+='\nFigure 1B also uses the complete density range after the 2026-09-22 repair. Caption and Methods match the corrected figure; numerical results and all other main figures remain unchanged.\n';p.write_text(s)
files=['template.tex','template.pdf','FIGURE1_CAPTION.md','README.md','REVISION_NOTES.md','MAIN_OVERLEAF_CHECKS.json','figure/figure1_ramp_overview.pdf']
original={rel:sha(LIVE/rel) for rel in files+['Main_Overleaf.zip','Supplementary_Overleaf.zip','supplementary_material.pdf']}
original.update({str(p.relative_to(LIVE)):sha(p) for p in (LIVE/'Main_Overleaf').rglob('*') if p.is_file()})
(HERE/'original_live_hashes.json').write_text(json.dumps(original,indent=2)+'\n');(HERE/'install_files.json').write_text(json.dumps(files,indent=2)+'\n')
check=HERE/'independent_compile';check.mkdir()
for p in (HERE/'Main_Overleaf').iterdir():
    if p.suffix in ['.tex','.bib','.bst']:shutil.copy2(p,check/p.name)
shutil.copytree(HERE/'Main_Overleaf/figure',check/'figure')
print('Verified complete support, original bandwidth, unchanged inputs and identical pixels outside ridge axes. Staged manuscript and clean upload build.')
