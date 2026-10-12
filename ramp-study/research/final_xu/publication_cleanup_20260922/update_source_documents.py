from pathlib import Path
import re, json, shutil, hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;STAGE=HERE/'manuscript_stage'
changed=[]
def write(p,t):
 p=Path(p)
 if p.exists():
  b=HERE/'backup_local_docs'/p.relative_to(ROOT)
  if not b.exists():b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,b)
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(t);changed.append(str(p.relative_to(ROOT)))
def rename(t):return t.replace('Harmony-direct','Harmony').replace('Harmony direct','Harmony')
main=ROOT/'Main Figure';folder=main/'publication_display_20260922'
p=main/'FIGURE_CAPTION.md';t=rename(p.read_text())
t=t.replace('2026-09-15','2026-09-22').replace('icon_update_20260915/figure_1_icon_updated.png','publication_display_20260922/figure_1_melanoma.png')
a=t.index('**(B) Cross-platform alignment in TNBC.**');b=t.index('\n\n**(C)',a)
t=t[:a]+'''**(B) Cross-platform alignment in melanoma.** The frozen three-platform benchmark contains 60,000 cells, 46 sample units and 12 shared markers: Hickey CODEX, Hoch IMC and the Ptacek/Vesely MIBI cohort. Legend labels denote the technologies CODEX, IMC and MIBI, respectively. Rows show Raw, Harmony, cyCombine and RAMP. Each row includes sample-level distributions of CD3, CD8 and SOX10, followed by two views of the same UMAP colored by source dataset and harmonized cell type. All frozen cells, existing coordinates and ten display labels are retained. Cell-type colors and layer order follow the previously accepted melanoma display. Raw ridges use asinh with cofactor 1 for display; corrected ridges retain their method's output scale, so horizontal values should not be compared across rows. Ridges use pooled method–marker 0.5th–99.5th percentile display ranges, 160-bin histograms and Gaussian smoothing with sigma 6 bins; each curve is normalized to its own maximum and does not encode population abundance. All 46 sample-unit curves are shown per marker and method. Cell-type annotations are applied after fitting for evaluation and display. Integration performance is compared in Figure 2.''' + t[b:]
a=t.index('**（B）TNBC 对齐。**');b=t.index('\n\n**（C）',a)
t=t[:a]+'''**（B）Melanoma 对齐。** 原三平台 benchmark 共 60,000 个冻结细胞、46 个样本单位和 12 个共享 marker，来源为 Hickey CODEX、Hoch IMC 和 Ptacek/Vesely MIBI 队列。四行依次展示 Raw、Harmony、cyCombine、RAMP；每行包含 CD3、CD8、SOX10 的逐样本 ridge，以及按平台和细胞类型着色的同一套 UMAP 坐标。保留全部细胞、冻结坐标、十种展示标签及之前认可的 melanoma 配色与图层顺序。Raw ridge 用 asinh(x/1) 显示，其他行保留各方法输出尺度，横轴数值不跨行直接比较。曲线使用 pooled 0.5%–99.5% 范围、160-bin 直方图及 sigma=6 bins 的高斯平滑；全部 46 个样本单位均展示，每条曲线按自身峰高归一化，不表示细胞丰度。整合性能另见 Figure 2。''' + t[b:]
t=t.replace('All six methods are evaluated','Harmony, cyCombine, CytoNorm, reconstructed fastMNN and RAMP are evaluated').replace('六条输出流程','五条输出流程')
t=t.replace('- 当前图与重画入口：[shared_cd4_h1_20260914](shared_cd4_h1_20260914/README.md)。','- 当前图与重画入口：[publication_display_20260922](publication_display_20260922/README.md)。')
t=t.replace('(shared_cd4_h1_20260914/median_shift_box_statistics.csv)','(publication_display_20260922/median_shift_box_statistics.csv)').replace('(shared_cd4_h1_20260914/panel_d_display_overflow.csv)','(publication_display_20260922/panel_d_display_overflow.csv)')
write(p,t)
p=main/'README.md';t=p.read_text();table=t[t.index('| Benchmark |'):t.index('B 保留全部冻结细胞')]
write(p,'''# Main Figure / Figure 1

更新：2026-09-22。当前图为 A–D，尺寸 180 × 188 mm。

- [当前整图](publication_display_20260922/figure_1_melanoma.png)
- [当前 PDF](publication_display_20260922/figure_1_melanoma.pdf)
- [英文及中文 caption](FIGURE_CAPTION.md)
- [重画入口与记录](publication_display_20260922/README.md)
- [RAMP Methods](METHODS.md)

| Panel | 当前内容 |
|---|---|
| A | 三平台汇合概念图 + 预处理、无标签匹配、单调变换 |
| B | Melanoma：Raw/Harmony/cyCombine/RAMP；CD3/CD8/SOX10 ridge 与平台/细胞类型 UMAP |
| C | H1 CD4：Harmony/CytoNorm；h17 CD8：cyCombine；H2 CD14：重建 fastMNN |
| D | Harmony、cyCombine、CytoNorm、fastMNN reconstructed、RAMP 的中位百分位位移；每种方法使用相同的 6,173 个组合 |

## 冻结范围

'''+table+'''B 沿用此前认可的三平台 melanoma 版本：全部 60,000 个细胞、46 个样本单位、12 个 shared markers、十种展示标签。UMAP 坐标、矩阵和分析结果不变。C 例子不变；D 移除 Harmony 的 fixed-arcsinh 分支，保留五种方法的完整原数值。

公开方法名称统一为 Harmony，Methods 保留它直接处理 source-scale marker 坐标的具体配置。内部标识符、冻结文件名和历史版本保留，避免破坏数据追溯。

当前版本同步至 `../../bioRxiv_Ver1/`。早期 TNBC 示例与六条输出流程的版本保存在 `icon_update_20260915/`，本轮修改前根文档备份于 `../publication_cleanup_20260922/backup_local_docs/`。旧目录不代表现用图。
''')
write(folder/'README.md','''# Figure 1 — melanoma and Harmony display update

Current: `figure_1_melanoma.pdf`, `.png`, `.svg`.
Rebuild from final_xu: `python3 'Main Figure/publication_display_20260922/build_figure.py'`.

- B uses the accepted three-platform melanoma example from `../Figure 2/Panel C/umap_style_review/`; all 60,000 cells and all 46 sample units retained. CD3/CD8/SOX10; 12 shared markers. No alignment, UMAP fitting or cell selection is rerun.
- C retains the three selected artifact examples. Public labels use Harmony.
- D excludes the fixed-arcsinh Harmony branch and retains the same 6,173 observations for each of five methods. Numerical box statistics are unchanged.
- `ridge_display_audit.csv` is checked against the saved melanoma display; `manifest.json` records original source hashes.
- `Panel A/` through `Panel D/` provide isolated PNGs. Earlier same-day TNBC label-only previews, where present, are superseded by `figure_1_melanoma`.

Caption: `../FIGURE_CAPTION.md`. Manuscript QA and synchronization: `../../publication_cleanup_20260922/`.
''')
base=ROOT/'cross_platform_trainability_v2_20260919/Figure'
write(base/'CAPTION.md',(STAGE/'FIGURE3_CAPTION.md').read_text())
write(base/'README.md','''# Cross-platform learning — current Figure 3 A–C

Current main figure contains one compact row, **A–C only**.

- [Combined PNG](Combined%20ABC/figure_cross_platform_learning_ABC.png) · [PDF](Combined%20ABC/figure_cross_platform_learning_ABC.pdf)
- [Panel A](Panel%20A/panel_a_cross_platform_learning.png)
- [Panel B](Combined%20ABC/Panel%20B/panel_b_cell_type_annotation.png)
- [Panel C](Combined%20ABC/Panel%20C/panel_c_missing_marker_prediction.png)
- [Current caption](CAPTION.md)

B/C read `../CURRENT_FIGURE3_RESULTS.json` and the `figure3_bc_update_20260922` release. The new display preserves all per-model and per-benchmark scores exactly and uses Harmony as the public method name.

B: seven benchmarks, three model families; RAMP balanced accuracy 0.772900 and macro-F1 0.751133. C: six benchmarks and 79 benchmark–marker combinations; RAMP Spearman 0.349041 and MAE 0.223443. Each displayed benchmark point averages the three model families equally. Model-specific results are shown in Supplementary Figures S10–S13.

D/E/F and `Combined ABCDEF/` are historical resource experiments, removed from the current main figure and not added to supplementary figures. Their scripts and results are preserved for reference; `build_panels_def.py` is not the current assembly entry point.

Current rebuild from final_xu:
`python3 publication_cleanup_20260922/build_display_assets.py figure3`

The active manuscript is `../../bioRxiv_Ver1/` relative to final_xu. Caption, Results, Methods and Discussion were synchronized on 2026-09-22.
''')
p=base.parent/'CURRENT_FIGURE3_RESULTS.json';d=json.loads(p.read_text());d.update(figure_directory=str(base/'Combined ABC'),panels_DEF='Archived; not in current main or supplementary figures.',remaining_scope='Main Figure 3 A–C only; current B/C results unchanged.',display_update='2026-09-22: A–C layout, Harmony public label.')
write(p,json.dumps(d,indent=2)+'\n')
p=ROOT/'Supplementary Figures/README.md';t=p.read_text();intro='''## 当前发布显示版本（2026-09-22）

全套 S1–S13 的当前 PDF/PNG/SVG 位于 [harmony_display_20260922](harmony_display_20260922/)，公开名称统一为 Harmony。其余像素、数据及原有抽样/配色规则均不变；`MANIFEST.json` 记录逐图旧来源和哈希。下文各分析目录保留数据与原绘图脚本，当前正式发布路径以 `CURRENT_PUBLICATION.json` 为准。

Figure 3 D–F 已从主图移除，未自动追加为附录。逐 marker 预测图和 Lung/HNSCC 全患者验证分布暂不增加。现有 S1–S13 编号不变。

'''
write(p,t.split('\n',1)[0]+'\n\n'+intro+t.split('\n',1)[1].lstrip())
write(ROOT/'Supplementary Figures/CURRENT_PUBLICATION.json',(STAGE/'SUPPLEMENTARY_FIGURE_SOURCES.json').read_text())
write(ROOT/'Supplementary Figures/harmony_display_20260922/README.md','''# Current publication labels: Harmony

S1–S13 PDFs, PNGs and SVGs are in `pdf/`, `png/` and `svg/`. Only the visible method name was changed. PDF raster comparison confirmed identical pixels outside the edited text-label rectangles for every figure. `MANIFEST.json` records all old and new paths/hashes and validation results.

Frozen inputs, statistical values, internal method identifiers and older source figures are unchanged. Reproduce from final_xu: `python3 publication_cleanup_20260922/build_display_assets.py supplementary`.
''')
# Root Figure 2 documentation: display labels only, no computational key or path replacement.
for name in ['README.md','FIGURE_CAPTION.md','CAPTION.md']:
 p=main/'Figure 2'/name
 if p.exists():
  t=p.read_text();new=re.sub(r'Harmony-direct(?![\w/\[])','Harmony',t)
  if new!=t:write(p,new)
(HERE/'local_documents_updated.json').write_text(json.dumps(changed,indent=2)+'\n')
print('Updated',len(changed),'active source documents; previous versions backed up.')
