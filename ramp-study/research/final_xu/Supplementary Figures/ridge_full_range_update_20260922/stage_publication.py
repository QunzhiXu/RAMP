from pathlib import Path
import json,shutil,hashlib
HERE=Path(__file__).resolve().parent;SUP=HERE.parent;ROOT=SUP.parent;LIVE=ROOT.parent/'bioRxiv_Ver1';STAGE=HERE/'stage'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
STAGE.mkdir(exist_ok=False)
for rel in ['template.tex','bibliography.bib','ramp_harvard.bst','supplementary_material.tex','supplementary_captions.tex','SUPPLEMENTARY_CAPTIONS.md','SUPPLEMENTARY_FIGURE_SOURCES.json','README.md','REVISION_NOTES.md']:
    shutil.copy2(LIVE/rel,STAGE/rel)
shutil.copytree(LIVE/'figure',STAGE/'figure')
for name in ['Main_Overleaf','Supplementary_Overleaf']:shutil.copytree(LIVE/name,HERE/name)
plan=json.loads((HERE/'REVISION_PLAN.json').read_text());sources=json.loads((STAGE/'SUPPLEMENTARY_FIGURE_SOURCES.json').read_text())
old_sentence='Histograms use 160 bins and Gaussian smoothing with sigma 6 bins. Horizontal limits are the pooled 0.5th-99.5th percentiles of the complete frozen cohort for each method-marker pair; scales are method-specific.'
new_sentence='Histograms include all displayed values. Bin width is the pooled 0.5th-99.5th percentile span divided by 160; Gaussian smoothing uses sigma 6 bins. Axes span the complete frozen cohort range plus at least four smoothing bandwidths at each end, retaining full tails; scales are method-specific.'
for rel in ['supplementary_captions.tex','SUPPLEMENTARY_CAPTIONS.md']:
    p=STAGE/rel;s=p.read_text();assert old_sentence in s;p.write_text(s.replace(old_sentence,new_sentence))
for item,record in zip(plan,sources[:7]):
    assert item['name']==record['case']
    new=Path(item['new']);stem=Path(item['png']).stem;pub=HERE/'published'
    for ext,sub in [('pdf','pdf'),('png','png'),('svg','svg')]:
        src=new/'output'/('pdf' if ext=='pdf' else '')/(stem+'.'+ext)
        dst=pub/sub/Path(record['destination']).with_suffix('.'+ext).name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    newpdf=pub/'pdf'/Path(record['destination']).name
    record.update(previous_source=record['source'],previous_sha256=record['sha256'],source=str(newpdf),sha256=sha(newpdf),change='Full ridge density support and padded axes; same bandwidth, cells, unit selection and palettes. UMAP pixels unchanged.')
    shutil.copy2(newpdf,STAGE/record['destination'])
    cfg=json.loads((new/'RIDGE_CONFIG.json').read_text())
    (new/'README_ZH.md').write_text(f"# {item['name']}：完整密度尾部版本\n\n由上一版 ridge_uniform_20260922 重绘，保留 {cfg['ridge_units']} 个展示单位、{cfg['ridge_cells']:,} 个细胞以及原配色、堆叠顺序和带宽。UMAP 图框内像素逐一核验相同。\n\n旧图先按全队列 0.5%–99.5% 截取直方图；本版全部数值进入直方图，沿原有 bin 宽扩展网格，左右各加至少 4 个高斯标准差的空间。高斯 sigma 仍为 6 bins。极端长尾使用完整线性横轴，可能使主峰视觉上变窄。\n\n运行 `python3 build_figure.py` 重绘。`ridge_display_audit.csv` 记录每条密度的样本数、完整范围、带宽、旧图遗漏数及尾部归零检查。共享 renderer 位于 `../../ridge_full_range_update_20260922/ridge_common.py`。\n\n旧版未改动；对齐、模型、UMAP 和评分均未重新计算。\n")
    (new/'CAPTION.md').write_text('# Ridge display conventions\n\n'+new_sentence+' Each ridge is normalized to its own maximum. Frozen sample selection and technology colors are unchanged.\n')
(STAGE/'SUPPLEMENTARY_FIGURE_SOURCES.json').write_text(json.dumps(sources,indent=2)+'\n')
p=STAGE/'template.tex';s=p.read_text();needle=r'\paragraph{Ridge plots.} Densities are shown over the pooled 0.5th--99.5th';assert needle in s
s=s.replace(needle,r'\paragraph{Ridge plots.} For Figure~\ref{fig:overview}B, densities are shown over the pooled 0.5th--99.5th')
needle='positions are not comparable between rows.\n';assert s.count(needle)==1
s=s.replace(needle,needle+'\nFor Figures S1--S7, histograms include the full observed range, retaining\nthe same bin width and smoothing bandwidth defined above. The display\nadds at least four smoothing bandwidths beyond each endpoint; the Gaussian\nkernel is truncated at four standard deviations. Platform-specific display\ntransformations are listed in the supplementary legends.\n')
p.write_text(s)
for rel in ['README.md','REVISION_NOTES.md']:
    p=STAGE/rel;p.write_text(p.read_text()+'\n\n## Supplementary density-tail correction (2026-09-22)\n\nS1–S7 now include full ridge distributions with padded axes and unchanged expression-unit bandwidths. UMAPs, selected cells, palettes, alignments and benchmark scores are unchanged. Captions, Methods display settings and both Overleaf snapshots are synchronized. Earlier PDFs and upload packages are backed up.\n')
# Update clean upload snapshots; data tables and other figures retain their bytes.
sp=HERE/'Supplementary_Overleaf'
for rel in ['supplementary_material.tex','supplementary_captions.tex','SUPPLEMENTARY_CAPTIONS.md']:shutil.copy2(STAGE/rel,sp/rel)
for record in sources[:7]:shutil.copy2(STAGE/record['destination'],sp/record['destination'])
shutil.copy2(STAGE/'SUPPLEMENTARY_FIGURE_SOURCES.json',sp/'reference/SUPPLEMENTARY_FIGURE_SOURCES.json')
dat=sp/'supplementary_data/ridge_display';dat.mkdir()
for i,item in enumerate(plan,1):shutil.copy2(Path(item['new'])/'ridge_display_audit.csv',dat/f's{i}_ridge_display_audit.csv')
for rel in ['RANGE_AUDIT.csv','QA_SUMMARY.json']:shutil.copy2(HERE/rel,dat/rel)
(dat/'README.md').write_text('Complete-range density display audit. All plotted cell values enter the histograms; n_below/n_above are zero. old_n_below/old_n_above count observations formerly outside the display limits. sigma_bins remains 6 and bandwidth_expression_units preserves the earlier bandwidth. Density endpoints are zero after padding. No alignment or benchmark score was changed. Range expansion includes full-cohort outliers and can narrow the main peaks visually.\n')
count=sum(p.is_file() for p in (sp/'supplementary_data').rglob('*'))
p=sp/'README.md';s=p.read_text().replace('191 个数据',f'{count} 个数据');s+='\nS1–S7 已修复密度截断：全部入选细胞参与密度计算，横轴覆盖完整冻结队列范围并留出平滑尾部。带宽、配色、抽样和 UMAP 不变；新增的 ridge_display 审计文件记录这一修改。\n';p.write_text(s)
p=sp/'author_notes/SUPPLEMENTARY_REVIEW_ZH.md';p.write_text(p.read_text()+'\n\n## 本轮密度截断修复\n\nS1–S7 的 126 个 method×marker 面板均使用完整冻结数值范围，保留旧版表达单位带宽，左右至少 4 sigma padding。每条 ridge 无观察值遗漏且两端回到基线；原有 UMAP 图框内像素相同。详见 supplementary_data/ridge_display/。S8–S13 和统计数据保持不变。\n')
mp=HERE/'Main_Overleaf';shutil.copy2(STAGE/'template.tex',mp/'template.tex')
p=mp/'README.md';p.write_text(p.read_text()+'\n2026-09-22 密度尾部修复：Methods 明确区分 Figure 1 的原显示范围与附图 S1–S7 的完整范围。主图文件与数值结果均未改动。\n')
files=['template.tex','template.pdf','README.md','REVISION_NOTES.md','supplementary_material.pdf','supplementary_captions.tex','SUPPLEMENTARY_CAPTIONS.md','SUPPLEMENTARY_FIGURE_SOURCES.json','Main_Overleaf.zip','Supplementary_Overleaf.zip']+[r['destination'] for r in sources[:7]]
(HERE/'original_live_hashes.json').write_text(json.dumps({str(p.relative_to(LIVE)):sha(p) for p in [*(LIVE/r for r in files),*list((LIVE/'Main_Overleaf').rglob('*')),*list((LIVE/'Supplementary_Overleaf').rglob('*'))] if p.is_file()},indent=2)+'\n')
(HERE/'install_files.json').write_text(json.dumps([r for r in files if not r.endswith('.zip')],indent=2)+'\n')
print('Staged 7 revised plots, captions, Methods and upload sources; supplementary data files:',count)
