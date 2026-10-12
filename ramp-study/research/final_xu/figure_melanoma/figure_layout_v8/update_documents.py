"""Update the root story documentation to match the two-scale Panel F."""
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
STORY=HERE.parent

EN='''**(F) Cross-patient state enrichment.** Two horizontal box plots show AUROC (upper) and target-state enrichment in the highest-scoring 20% of candidate CD8 cells (lower), using independent horizontal scales. Each plot contains the same 14 evaluable IMC patients with at least 20 candidate CD8 cells and both target classes. Enrichment is the target-cell fraction among selected cells divided by the target-cell fraction among all candidate CD8 cells in the same patient; selection uses the whole patient image, as in E. Median AUROC is 0.707 and median enrichment is 1.98-fold; enrichment exceeds one in all 14 patients. Dashed lines indicate reference values of 0.5 for AUROC and one-fold for enrichment. Boxes show quartiles and medians; whiskers extend to the most extreme observations within 1.5 interquartile ranges; all patient values are overlaid. Standardization and classifier fitting excluded the evaluated patient; the RAMP alignment was frozen from the complete cohort.'''
ZH='''**（F）跨患者的空间状态富集。** 两条横向箱线图分别展示 AUROC（上）和得分最高 20% 的候选 CD8 中的目标状态富集倍数（下），使用独立横轴。每条包含同样 14 位可评估 IMC 患者，纳入条件为至少 20 个候选 CD8 细胞且同时存在正负类。富集倍数定义为高分组目标细胞比例除以该患者全部候选 CD8 中的目标比例；与 E 一致，在整位患者范围内选取最高分的 20%。AUROC 中位数为 0.707，富集倍数中位数为 1.98，14 位患者均大于 1 倍。虚线分别表示 AUROC 的 0.5 和富集的 1 倍参考值。箱体展示四分位范围和中位数，须线延伸到 1.5 IQR 内的最远观测，叠加全部患者点。每折标准化器与分类器均排除被评估患者；输入为完整队列的冻结 RAMP 对齐结果。'''

def main():
    caption=(HERE/'previous_docs/FIGURE_CAPTION.md').read_text()
    caption=caption.replace('figure_layout_v7/melanoma_three_row_redesign_v7','figure_layout_v8/melanoma_three_row_redesign_v8')
    caption=re.sub(r'\*\*\(F\).*?(?=\n\n\*\*\(G\))',EN,caption,flags=re.S)
    caption=re.sub(r'\*\*（F）.*?(?=\n\n\*\*（G）)',ZH,caption,flags=re.S)
    caption=caption.replace('[当前绘图记录](figure_layout_v7/LAYOUT_AUDIT.json)','[当前绘图记录](figure_layout_v8/LAYOUT_AUDIT.json)')
    caption=caption.replace('v7 排版没有重新拟合坐标','v8 排版没有重新拟合坐标')
    caption+='\n\nPanel F 的完整患者级 AUROC、AP、Precision、Recall 及富集倍数保留在 [完整指标表](figure_layout_v8/data/panel_F_all_metrics_by_patient.csv)；预测和 20% 截点均未修改。\n'
    (STORY/'FIGURE_CAPTION.md').write_text(caption)
    readme=(HERE/'previous_docs/README.md').read_text()
    readme=readme.replace('当前采用 **figure_layout_v7**，更新日期：2026-09-14。','当前采用 **figure_layout_v8**，更新日期：2026-09-15。')
    readme=readme.replace('figure_layout_v7/melanoma_three_row_redesign_v7','figure_layout_v8/melanoma_three_row_redesign_v8')
    readme=readme.replace('figure_layout_v7/build_figure.py','figure_layout_v8/build_figure.py').replace('figure_layout_v7/prepare_visual_data.py','figure_layout_v8/prepare_visual_data.py')
    readme=readme.replace('figure_layout_v7/panels','figure_layout_v8/panels').replace('figure_layout_v7/LAYOUT_AUDIT.json','figure_layout_v8/LAYOUT_AUDIT.json')
    a=readme.index('本次 v7 更新');b=readme.index('\n\n## 当前',a)
    readme=readme[:a]+'''本次 v8 只修改 F：上下排列两条横向箱线图，展示 AUROC 和 Top-20% 目标富集倍数，各自使用独立横轴和参考线。每条包含原来的 14 位患者。A–E、G、H 与 v7 像素一致；预测、阈值、模型和 UMAP 均未修改。旧版完整保留在 figure_layout_v7。已同步 manuscript 的 Figure 4 PNG/SVG、Panel F 图注与两份 PDF；校验见 [同步记录](figure_layout_v8/manuscript_sync/INSTALL_VERIFY.json)。

核对记录：[数据核对](figure_layout_v8/data/DATA_QA.json)、[版面核对](figure_layout_v8/REVISION_QA.json)。完整原始指标保留在 [患者级指标表](figure_layout_v8/data/panel_F_all_metrics_by_patient.csv)。'''+readme[b:]
    readme=readme.replace('当前图以 v7 为准','当前图以 v8 为准')
    readme=readme.replace('| 3 | F | Held-out validation | 14 位可评估患者的 AUROC、AUPRC、Precision、Recall box plot |','| 3 | F | State enrichment | 14 位患者的 AUROC 与 Top-20% 目标富集倍数；独立横轴的横向箱线图 |')
    readme=re.sub(r'\| F：患者留出验证 \|[^\n]+','| F：跨患者状态富集 | 同样 14 位可评估患者；中位 AUROC 0.706879，Top-20% 目标富集倍数中位数 1.976246，14/14 大于 1。富集为高分组目标比例 / 该患者全部候选 CD8 的目标比例。各轴参考线为 0.5 和 1 倍，箱体为四分位范围及中位数，须线为 1.5 IQR 内最远观测。 |',readme)
    readme=readme.replace('v7 未重新拟合 UMAP','v8 未重新拟合 UMAP').replace('`figure_layout_v7/` | 当前整图','`figure_layout_v8/` | 当前整图')
    readme=readme.replace('至 `figure_layout_v6/` | 排版历史','至 `figure_layout_v7/` | 排版历史')
    readme=readme.replace('仅重绘当前 v7','仅重绘当前 v8').replace('最后一行生成当前 v7','最后一行生成当前 v8').replace('之后需另运行 v7 绘图命令','之后需另运行 v8 绘图命令').replace('不是 v7 排版检查报告','不是 v8 排版检查报告')
    (STORY/'README.md').write_text(readme)
    (HERE/'README.md').write_text('''# Melanoma figure v8

2026-09-15：仅重绘 Panel F。上方 AUROC，横轴 0–1，参考线 0.5；下方目标富集倍数，横轴 0–3.5 倍，参考线 1 倍。两条均为横向箱线图，每条 14 位患者，单位不共用。中位数为 0.706879 和 1.976246 倍。

- [整图 PNG](melanoma_three_row_redesign_v8.png) · [SVG](melanoma_three_row_redesign_v8.svg)
- [Panel F](panels/F.png)
- [绘图数据](data/panel_F_two_metrics_by_patient.csv)
- [完整指标表](data/panel_F_all_metrics_by_patient.csv)：保留原始 AUROC、AP、Precision、Recall 和富集倍数。

保持原患者留出预测、20% 截点、患者纳入条件及所有其他 panels。富集定义为 precision_top20 / patient target prevalence。原始数据与预测只读；v7 完整保留。图注、根目录 README 和 manuscript 均已更新。两份 PDF 保持 18 页，仅第 7 页与第 8 页改变；同步记录见 manuscript_sync/INSTALL_VERIFY.json。

复现：

```bash
python3 -B figure_melanoma/figure_layout_v8/prepare_visual_data.py
python3 -B figure_melanoma/figure_layout_v8/build_figure.py
```

其他 panels 复用 v7 的冻结缓存及绘图代码。验证见 data/DATA_QA.json、LAYOUT_AUDIT.json 和 REVISION_QA.json。
''')

if __name__=='__main__':main()
