from pathlib import Path
import json,hashlib,shutil,csv,zipfile,re
import fitz
HERE=Path(__file__).resolve().parent;SUP=HERE.parent;LIVE=SUP.parent.parent/'bioRxiv_Ver1';STAGE=HERE/'stage'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks={}
for task,entry,pkg in [('main','template','Main_Overleaf'),('supplement','supplementary_material','Supplementary_Overleaf')]:
    a=fitz.open(STAGE/(entry+'.pdf'));b=fitz.open(HERE/('independent_'+task)/(entry+'.pdf'));old=fitz.open(LIVE/(entry+'.pdf'))
    assert len(a)==len(b)==(48 if task=='main' else 17)
    changed=[]
    for i in range(len(a)):
        assert a[i].get_text()==b[i].get_text(),(task,i,'text')
        assert a[i].get_pixmap(dpi=72).samples==b[i].get_pixmap(dpi=72).samples,(task,i,'pixels')
        if a[i].get_pixmap(dpi=72).samples!=old[i].get_pixmap(dpi=72).samples:changed.append(i+1)
        for x0,y0,x1,y1,*_ in a[i].get_text('words'):
            assert x0>=-1 and y0>=-1 and x1<=a[i].rect.width+1 and y1<=a[i].rect.height+1,(task,i)
    if task=='supplement':assert changed==[1,5,6,7,8,9,10,11],changed
    for root in [STAGE,HERE/('independent_'+task)]:
        log=(root/(entry+'.log')).read_text(errors='replace');assert not re.search(r'Overfull|undefined|LaTeX Error',log),(task,root)
    report=dict(engine='pdfLaTeX via latexmk',pages=len(a),independent_compilation=True,all_page_text_and_render_pixels_match_reference=True,missing_references=False,overfull_boxes=False,changed_pages=changed,update='S1–S7 full ridge density ranges; associated display documentation updated. All numerical results unchanged.')
    if task=='supplement':
        report.update(figure_pages=13,legend_pages=4,source_data_files=201,ridge_panels=126,ridge_curves=3636,umap_pixels_unchanged=True,complete_density_support=True)
        shutil.copy2(STAGE/(entry+'.pdf'),HERE/pkg/'reference'/(entry+'.pdf'))
    else:report.update(main_figures=7,all_seven_main_figure_files_unchanged=True,main_document='template.tex',compiler='pdfLaTeX')
    (HERE/pkg/'COMPILATION_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
    checks[task]=report
    print(task,'verified',len(a),'pages; changed pages:',changed,flush=True)
# Preserve all existing source-data files and untouched figures exactly.
for p in (LIVE/'Supplementary_Overleaf/supplementary_data').rglob('*'):
    if p.is_file():assert sha(p)==sha(HERE/'Supplementary_Overleaf/supplementary_data'/p.relative_to(LIVE/'Supplementary_Overleaf/supplementary_data'))
for p in (LIVE/'Main_Overleaf/figure').rglob('*'):
    if p.is_file():assert sha(p)==sha(HERE/'Main_Overleaf/figure'/p.relative_to(LIVE/'Main_Overleaf/figure'))
for r in json.loads((STAGE/'SUPPLEMENTARY_FIGURE_SOURCES.json').read_text())[7:]:assert sha(STAGE/r['destination'])==sha(LIVE/r['destination'])
shutil.copy2(HERE/'Main_Overleaf/COMPILATION_CHECK.json',STAGE/'MAIN_OVERLEAF_CHECKS.json')
# Final standalone upload archives.
for name in ['Main_Overleaf','Supplementary_Overleaf']:
    out=HERE/name
    with (out/'FILE_MANIFEST.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['relative_path','bytes','sha256'])
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='FILE_MANIFEST.csv':w.writerow([str(p.relative_to(out)),p.stat().st_size,sha(p)])
    z=HERE/(name+'.zip')
    with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as zf:
        for p in sorted(out.rglob('*')):
            if p.is_file():zf.write(p,p.relative_to(out))
    with zipfile.ZipFile(z) as zf:
        assert zf.testzip() is None
        for p in out.rglob('*'):
            if p.is_file():assert hashlib.sha256(zf.read(str(p.relative_to(out)))).hexdigest()==sha(p)
    print(name,round(z.stat().st_size/1e6,2),'MB, ZIP content and hashes verified.',flush=True)
checks['all_previous_source_data_unchanged']=True;checks['main_figure_files_unchanged']=True;checks['s8_s13_files_unchanged']=True
(HERE/'PUBLICATION_QA.json').write_text(json.dumps(checks,indent=2)+'\n')
# Reproducibility entry points. Old versions are retained untouched.
(HERE/'CURRENT_PUBLICATION_NEXT.json').write_text((STAGE/'SUPPLEMENTARY_FIGURE_SOURCES.json').read_text())
readme=(SUP/'README.md').read_text()
readme=readme.replace('全套 S1–S13 的当前 PDF/PNG/SVG 位于 [harmony_display_20260922](harmony_display_20260922/)，公开名称统一为 Harmony。其余像素、数据及原有抽样/配色规则均不变；`MANIFEST.json` 记录逐图旧来源和哈希。下文各分析目录保留数据与原绘图脚本，当前正式发布路径以 `CURRENT_PUBLICATION.json` 为准。','S1–S7 最新 PDF/PNG/SVG 位于 [ridge_full_range_update_20260922/published](ridge_full_range_update_20260922/published/)，已修复密度截断；S8–S13 仍使用 [harmony_display_20260922](harmony_display_20260922/) 的原文件。公开名称统一为 Harmony。完整来源与哈希见 `CURRENT_PUBLICATION.json`。')
readme=readme.replace('ridge_uniform_20260922','ridge_full_range_20260922')
readme=readme.replace('本次仅统一 ridge 的技术配色、轮廓、独立图例和堆叠规则，并减少 HNSCC/Lung story 展示的患者数。原有 UMAP 内容、对齐数据和评分保持不变。','本轮修复将完整表达范围纳入直方图，并在两端增加至少 4 个平滑带宽的空间。保留旧版实际带宽、技术配色、患者选择与堆叠顺序；原有 UMAP 图框内像素、对齐数据和评分保持不变。原先按 0.5%–99.5% 截取的版本仍保存在各组 ridge_uniform_20260922 下。极端长尾可能使主峰在完整线性横轴上显得较窄。')
readme=readme.replace('共享脚本、统一色表、原文件哈希和 QA 总表位于 `ridge_style_update_20260922/`。','当前共享脚本、范围审计、原文件哈希和 QA 总表位于 `ridge_full_range_update_20260922/`；旧配色修订保存在 `ridge_style_update_20260922/`。')
readme=readme.replace('轮廓修正：七页均恢复完整横轴上的连续轮廓线，零密度尾部也保留；密度、抽样和配色不变。','历史轮廓修正：上一版只恢复当时显示范围上的连续轮廓线，未恢复范围外的密度。本轮 full-range 修复进一步解决完整分布被截取的问题。')
(HERE/'README_NEXT.md').write_text(readme)
(HERE/'README.md').write_text('# Supplementary ridge truncation repair\n\nS1–S7: 126 density panels, 3,636 sample-level ridge curves. All displayed observations enter the histogram; endpoints reach zero after padding. The full frozen-cohort extrema determine axis ranges. Original bin widths and Gaussian sigma=6 bins are retained to numerical precision, rather than resmoothing with 160 bins over the expanded range.\n\nScripts: `prepare.py` creates versioned sources; `render_all.py` renders seven pages; `verify.py` audits full support, frozen inputs and unchanged UMAP pixels; `stage_publication.py` prepares captions/Methods and upload sources; `finalize.py` checks independent builds and creates ZIPs; `install.py` backs up and installs reviewed outputs. Do not rerun preparation over existing version directories.\n\n`RANGE_AUDIT.csv` records all 126 old/new ranges. `QA_SUMMARY.json` covers the seven figures; `PUBLICATION_QA.json` covers compiled documents and packages. `published/` contains the revised figures. Source data, sample selections, transformations, UMAPs and scores remain frozen. Rare long tails are retained and can compress the main peak visually.\n')
