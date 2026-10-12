from pathlib import Path
import json,shutil,re
S=Path(__file__).resolve().parents[2];U=S/'update_20260921';R=S/'releases/senosain140k_20260921';M=U/'bioRxiv_Ver1'
# Update the detailed bilingual story caption without changing its biological interpretation.
p=S/'FIGURE_CAPTION.md';s=p.read_text()
changes={'2026-09-15':'2026-09-21','Figure 5':'Figure 6','图 5':'图 6','0.834':'0.870','0.790':'0.827','0.692':'0.696','0.239–0.883':'0.275–0.889','0.66694':'0.63407','77.7%':'75.1%','53.8%':'56.7%','6.83 × 10⁻⁵':'1.92 × 10⁻⁴','1.40':'1.36','1.11–1.75':'1.07–1.71','0.00407':'0.0104',
 'the previously computed overall CAF fraction and global CAF HLA-ABC median score':'the overall CAF fraction and global CAF HLA-ABC median score from the current three-platform analysis',
 '此前计算的整体 CAF 比例及全 CAF HLA-ABC 中位数分数':'整体 CAF 比例及当前三平台分析得到的全 CAF HLA-ABC 中位数分数',
 'Spearman correlation from unchanged, unclipped cell values':'Spearman correlation from the updated, unclipped cell values'}
for a,b in changes.items():s=s.replace(a,b)
a='with 1,000 cells sampled per patient across all cell types without cell-type quotas.'
b=a+' For Senosain, these cells were resampled from the updated 140,000-cell annotation resource using QC-kept cells in a frozen random order; mixed-label cells passing QC were retained. The same patients and the original Cords and Zheng reference cells were preserved. Cell labels were used for display only.'
assert a in s;s=s.replace(a,b)
a='每人从全部细胞类型中抽取 1,000 个细胞，不设置细胞类型配额。'
b=a+'Senosain 使用更新后的 140,000 细胞注释资源，在固定随机顺序下选择通过 QC 的细胞；保留通过 QC 的 Mix 细胞及 C02。患者名单、Cords 与 Zheng 的参考细胞不变，细胞标签只用于展示。'
assert a in s;s=s.replace(a,b)
s=s.replace('核对日期：2026-09-21；','核对日期：2026-09-21；正式分析版本 `releases/senosain140k_20260921`；')
p.write_text(s)
combined=S/'Figure/Combined ABCDEF'
(combined/'CAPTION_EN.md').write_text(s.split('## English\n\n',1)[1].split('\n---\n',1)[0]+'\n')
(combined/'CAPTION_ZH.md').write_text(s.split('## 中文对照\n\n',1)[1])
# Manuscript changes are confined to the lung result/caption and lung methods sections.
p=M/'template.tex';s=p.read_text();old=s
start=s.index('\\subsection{Predicted vCAF HLA-ABC state');end=s.index('\\subsection{',start+15)
part=s[start:end]
for a,b in {
 '0.834':'0.870','0.692':'0.696','0.239--0.883':'0.275--0.889','0.790':'0.827',
 r'77.7\%':r'75.1\%',r'53.8\%':r'56.7\%',r'6.83\times10^{-5}':r'1.92\times10^{-4}',
 '1.43 per standard deviation':'1.36 per standard deviation','1.12--1.81':'1.07--1.71','P=0.0037':'P=0.0104',
 r'13.9\% versus 21.7\%':r'14.1\% versus 21.6\%',r'6.7\% versus 11.1\%':r'7.0\% versus 10.8\%',
 r'6.5\% versus':r'6.1\% versus',r'10.3\%)':r'10.7\%)',r'18.4\% versus 8.3\%':r'17.4\% versus 9.2\%'
}.items():part=part.replace(a,b)
part=part.replace('platform, 1,000 cells per patient, without cell-type quotas.','platform, 1,000 cells per patient, without cell-type quotas. CyTOF reference\ncells were sampled from the updated QC-filtered annotation resource\n(Methods); IMC and mIF reference cells were retained.')
s=s[:start]+part+s[end:]
start=s.index('\\subsection{Lung vCAF HLA-ABC inference');end=s.index('\\subsection{',start+15);part=s[start:end]
a='giving 84,000 reference cells. The nine aligned features were CD3, CD4, CD8,'
b='''giving 84,000 reference cells. For this application, the Senosain reference
was refreshed from the updated 140,000-cell annotation resource: the first
1,000 QC-kept cells per selected patient in a frozen random order were used.
Mixed annotations passing QC were retained, and neither cell identities,
HLA values nor outcomes were used to set sampling quotas. The patient list
and the original Cords and Zheng reference cells were unchanged. This
refresh applies to the HLA-ABC application; the integration and trainability
benchmarks retain their separately frozen reference datasets. The nine
aligned features were CD3, CD4, CD8,'''
assert a in part;part=part.replace(a,b).replace('0.66694','0.63407')
a='''B-cell and neutrophil fractions were $2.6\\times10^{-5}$,
$2.6\\times10^{-5}$, $5.2\\times10^{-3}$ and $1.3\\times10^{-4}$, respectively.'''
b='''B-cell and neutrophil fractions were $2.03\\times10^{-4}$,
$6.32\\times10^{-4}$, $5.49\\times10^{-4}$ and $6.32\\times10^{-4}$, respectively.'''
assert a in part;part=part.replace(a,b)
s=s[:start]+part+s[end:];p.write_text(s)
# Record readable manuscript differences for verification.
import difflib
(U/'MANUSCRIPT_DIFF.txt').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='before/template.tex',tofile='after/template.tex')))
shutil.copy2(S/'Figure/Combined ABCDEF/panels_ABCDEF_preview.png',M/'figure/figure6_lung_hla.png')
cap=part # keep manuscript caption extracted exactly from the staged TeX
start=s.index('\\paperfig{figure6_lung_hla}{%');end=s.index('}{fig:lunghla}',start)
(M/'FIGURE6_CAPTION.md').write_text('# Figure 6 caption\n\nCurrent source: `template.tex`, Figure 6. Analysis version: `senosain140k_20260921`.\n\n```latex\n'+s[start:end]+'}{fig:lunghla}\n```\n')
p=M/'README.md';x=p.read_text();a='| 6 | [Lung HLA-ABC](figure/figure6_lung_hla.png) | Former Figure 5; image unchanged |';b='| 6 | [Lung HLA-ABC](figure/figure6_lung_hla.png) | Updated Senosain reference, 2026-09-21; source: `../final_xu/story_lung_HLA_ABC/Figure/Combined ABCDEF/` |';assert a in x;x=x.replace(a,b);p.write_text(x)
# Entry points make the current analysis unambiguous, without overwriting legacy frozen inputs.
(S/'README.md').write_text('''# Lung HLA-ABC story

当前正式版本：**senosain140k_20260921**（2026-09-21）。

- [正式合并图](Figure/Combined%20ABCDEF/panels_ABCDEF_preview.png)
- [图注](FIGURE_CAPTION.md)
- [当前分析数据与脚本](releases/senosain140k_20260921/)
- [复核结果与新旧比较](releases/senosain140k_20260921/RESULTS_ZH.md)
- [步骤 3–6 的当前结果摘要](summary_steps3_6/README_ZH.md)
- [更新与同步记录](update_20260921/UPDATE_RECORD.md)

Senosain 使用新版 140k 资源中通过 QC 的细胞，每位原参考患者按固定随机顺序抽取 1,000 个；保留 Mix 和 C02，不按标签、HLA 值或结局选细胞。Cords/Zheng 参考细胞及全部目标细胞不变。重新拟合相同 RAMP、重做预测和下游统计，未改变算法及展示阈值。

`data_umap`、`umap`、`validation`、`data_cords`、`survival`、`mechanism_support` 是上一版冻结包，保留用于复现和旧新核对；**当前预测、统计和 UMAP 以 releases/senosain140k_20260921 为准**。未重跑或替换 Comparison、Figure 2 或 trainability benchmark。

旧图保存在各 Figure 子目录的 `versions/before_senosain140k_20260921/`；旧 caption 和 summary 保存在 `update_20260921/backup_story/`。
''')
Q=S/'summary_steps3_6'
(Q/'README_ZH.md').write_text('''# 肺癌故事步骤 3–6：当前正式摘要

版本：`senosain140k_20260921`。完整分析位于 [正式结果包](../releases/senosain140k_20260921/)，旧摘要已备份到 `../update_20260921/backup_story/summary_steps3_6/`。

- `03_projection`：527 位患者的 vCAF 汇总、分子阈值和纳入流程。
- `04_vcaf_markers`：新版预测关联的 CD73/IDO ROI、患者和总体统计；BH 校正为这次明确指定的 2 项分子终点。
- `05_immune_environment`：新版患者分组及全部 12 项组成终点；`COMPOSITION_COMPARISONS.csv` 的组均值为描述性统计，p、CI、q 来自调整模型，非未调整检验。
- `06_DFS`：主配置 RAMP k=30、平均 vCAF 得分、280 人/96 事件的 KM、Cox 和风险人数。其他配置及敏感性结果在正式包 `analysis/` 中完整保留。

Panel E/F 读取本摘要，使用同一 280 人及相同分组。中位数 0.63407，各 140 人；5 年 DFS 75.1% / 56.7%；调整 HR 1.36（95% CI 1.07–1.71，P=0.0104）。全 CAF 得分协变量来自当前三平台模型。

复现 KM 显示数据：`Rscript scripts/export_KM.R "$PWD"`。完整统计重跑脚本保存在正式包 `scripts/`；版本包是已完成试验的独立副本，来源清单保留原测试路径用于追溯。
''')
(combined/'README_ZH.md').write_text('''# Lung HLA-ABC 正式图（bioRxiv_Ver1 Figure 6）

版本：`senosain140k_20260921`。A–F 两行版式不变；更新 B 的联合 UMAP、D 的预测值及 E/F 的分组和统计。

D 保留原 389 细胞 ROI、合成背景、几何、0.25–0.55 色阶；局部 Spearman 0.870，整 core 0.827。E/F 使用同一 280 人，HR 1.36，5 年 DFS 75.1% / 56.7%。

图注：`CAPTION_EN.md` / `CAPTION_ZH.md`；完整正文说明见 story 根目录 `FIGURE_CAPTION.md`。

重建：依次运行各 Panel B/D/E/F 的绘图脚本、`../Combined ABC/scripts/assemble_ABC.py` 和 `scripts/assemble_ABCDEF.py`。

旧版保存在 `versions/before_senosain140k_20260921/`。
''')
for name in ['B','D','E','F']:
 p=S/f'Figure/Panel {name}/README_ZH.md';p.write_text(f'''# Panel {name}\n\n当前版本：senosain140k_20260921。\n\n读取新版正式分析 / summary_steps3_6；布局沿用上一版。详情见 story 根目录 `README.md` 和 `FIGURE_CAPTION.md`。\n\n绘图脚本：`scripts/draw_panel_{name}.py`。当前数据与检验记录：`data/`、`provenance/VERIFICATION.json`。旧版完整备份：`versions/before_senosain140k_20260921/`。\n''')
# Current traceability record, completed after PDF QA and installation.
(U/'UPDATE_RECORD.md').write_text('''# Lung story 正式更新（2026-09-21）

已授权将新 Senosain 参考测试提升为正式结果。正式包为 `../releases/senosain140k_20260921/`，与测试包独立，保留原数据哈希及复核报告。

1. 保持 84 位参考患者、原目标细胞、算法、k、临床纳入、ROI 与显示阈值；仅源参考改用新版 QC 后随机样本，并相应重新拟合、预测及统计。
2. B/D/E/F 已更新，A/C 不变；当前中文/英文 caption、summary 和 README 与新数据一致。
3. bioRxiv_Ver1 更新范围：Figure 6、肺癌 Results/Methods/caption、README；其余 6 张图及其他正文保持不变。
4. 旧正式图备份位于各 panel 的 versions；旧 root caption 和 summary 在本目录 backup_story。
5. 主结论仍成立。局部空间 rho=0.870；患者中位 rho=0.696；调整 HR=1.36，P=0.0104；低/高分组 5 年 DFS=75.1%/56.7%。

后续同步和编译校验见 `FINAL_VERIFICATION.json`。
''')
print('Current captions, manuscript lung text, summary README, and source pointers updated.')
