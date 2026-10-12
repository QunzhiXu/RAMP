"""Update current-version pointers and F-H descriptions after successful sync."""
from pathlib import Path
import json
import re
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def replace_paragraph(text, begin, end, replacement):
    a = text.index(begin); b = text.index(end,a)
    return text[:a]+replacement+'\n\n'+text[b:]


def main():
    sync = json.loads((HERE/'manuscript_sync/INSTALLED.json').read_text())
    assert sync['status'] == 'PASS'
    backup = HERE/'previous_docs'; backup.mkdir(exist_ok=True)
    for name in ['README.md','FIGURE_CAPTION.md']:
        assert not (backup/name).exists()
        shutil.copy2(ROOT/name,backup/name)
    readme = (ROOT/'README.md').read_text().replace('figure_layout_v6','figure_layout_v7').replace('melanoma_three_row_redesign_v6','melanoma_three_row_redesign_v7').replace('v6','v7')
    a = readme.index('本次 v7 只将'); b = readme.index('## 当前 A–H 顺序',a)
    readme = readme[:a]+'''本次 v7 更新 F、G、H：F 为四个患者级指标 box plot；G 圈出主要高分区域；H 为 High/Lower 两组患者平均表达秩的配对 box plot。A–E 与 v6 像素完全一致。模型、得分、UMAP 和分组均未重新拟合或调整。新版已同步 manuscript 的 Figure 4、对应 F–H 图注及两份 PDF。检查记录见 [数据核对](figure_layout_v7/data/DATA_QA.json)、[版面核对](figure_layout_v7/REVISION_QA.json) 与 [同步记录](figure_layout_v7/manuscript_sync/INSTALLED.json)。

'''+readme[b:]
    readme = readme.replace('14 位可评估患者的 AUROC 与 AUPRC/阳性比例','14 位可评估患者的 AUROC、AUPRC、Precision、Recall box plot')
    readme = readme.replace('CD11a、CD18、CD2、PD-1 的患者内蛋白差异','CD11a、CD18、CD2、PD-1 的 High/Lower 组患者平均表达秩')
    readme = re.sub(r'\| F：患者留出验证 \|.*?\|\n', '| F：患者留出验证 | 同样 14 位可评估患者，每个指标 14 个点；中位 AUROC 0.706879、AUPRC 0.211742、Precision 0.193093、Recall 0.395249。AUPRC 为 average precision，后两项沿用全患者候选 CD8 得分最高 20% 的调用规则。箱体为四分位范围及中位数，须线为 1.5 IQR 范围内最远观测。 |\n', readme)
    readme = readme.replace('点代表一位患者的“高得分组平均表达秩 − 其余组平均表达秩”，横线为 8 位患者的均值；', 'High 为患者内得分最高约 25% 的 CD8 细胞，Lower 为其余细胞。每个蛋白有两组 box plot，每组 8 个点，每点为一位患者在该组的平均表达秩；细线连接同一患者。箱体和须线定义同 F；')
    readme = readme.replace('得分色标 0–0.9。v7', '得分色标 0–0.9，虚线圈为主要高分聚集区的视觉指引，不作为分组门限。v7')
    readme = readme.replace('`figure_layout_v1/` 至 `figure_layout_v5/`', '`figure_layout_v1/` 至 `figure_layout_v6/`')
    readme = readme.replace('python3 -B figure_melanoma/figure_layout_v7/build_figure.py', 'python3 -B figure_melanoma/figure_layout_v7/prepare_visual_data.py\npython3 -B figure_melanoma/figure_layout_v7/build_figure.py')
    (ROOT/'README.md').write_text(readme)

    caption = (ROOT/'FIGURE_CAPTION.md').read_text().replace('figure_layout_v6','figure_layout_v7').replace('melanoma_three_row_redesign_v6','melanoma_three_row_redesign_v7').replace('v6 排版','v7 排版')
    caption = replace_paragraph(caption, '**(F)', '**(G)', '''**(F) Patient-held-out prediction performance.** Box plots show AUROC, AUPRC, precision and recall for the same 14 evaluable IMC patients with at least 20 candidate CD8 cells and both target classes. Median values are 0.707, 0.212, 0.193 and 0.395, respectively. AUPRC is average precision. Precision and recall use the highest-scoring 20% of candidate CD8 cells across each whole patient image, as in E. Boxes show quartiles and medians; whiskers extend to the most extreme observations within 1.5 interquartile ranges; all 14 patient values are overlaid. Standardization and classifier fitting excluded the evaluated patient; the RAMP alignment was frozen from the complete cohort.''')
    caption = caption.replace('The color scale saturates at 0.9.', 'The color scale saturates at 0.9. The dashed ellipse highlights the main high-score concentration for visual guidance and is not used to select cells.')
    caption = replace_paragraph(caption, '**(H)', 'Together, these analyses', '''**(H) Protein measurements associated with the transferred state.** CD11a, CD18, CD2 and PD-1 were excluded from the alignment and classifier inputs. Following cell-level CLR normalization of ADT counts, expression was ranked within each patient. High (red) comprises the highest score quartile and Lower (blue) the remaining cells, preserving the previous grouping. Each protein has two box plots with eight patient-level mean expression ranks per group; thin lines connect the same patient. Box and whisker definitions follow F. All four proteins are higher in the High group in all eight patients. The four-marker rank module has a mean within-patient Spearman correlation of 0.480 with the transferred score.''')
    caption = replace_paragraph(caption, '**（F）', '**（G）', '''**（F）患者留出预测性能。** 四个 box plot 分别展示 AUROC、AUPRC、Precision 和 Recall，均包含同样 14 位可评估 IMC 患者，纳入条件为至少 20 个候选 CD8 细胞且同时存在正负类。四项中位数依次为 0.707、0.212、0.193 和 0.395。AUPRC 使用 average precision；Precision 和 Recall 沿用 E 的规则，取每位患者整张图像中候选 CD8 得分最高的 20%。箱体展示四分位范围和中位数，须线延伸到 1.5 IQR 内的最远观测，每组叠加全部 14 位患者的点。每折标准化器与分类器均排除被评估患者；输入为完整队列的冻结 RAMP 对齐结果。''')
    caption = caption.replace('色标在 0.9 达到最高显示强度。', '色标在 0.9 达到最高显示强度。虚线圈标出主要高分聚集区，仅用于视觉指引，不参与细胞筛选或 High/Lower 分组。')
    caption = replace_paragraph(caption, '**（H）', '这些结果将', '''**（H）与转移状态相关的额外蛋白。** CD11a、CD18、CD2、PD-1 均未用于对齐或分类器输入。ADT 先进行细胞级 CLR 校正，再转为患者内表达秩。High（红色）为患者内得分最高约 25% 的 CD8 细胞，Lower（蓝色）为其余细胞，分组沿用原分析。每个蛋白展示两组 box plot，每组各 8 个点，每点为一位患者在该组的平均表达秩，细线连接同一患者；箱体和须线定义同 F。四个蛋白在全部 8 位患者中均为 High 组更高。四蛋白表达秩均值模块与转移得分的患者内 Spearman 相关系数均值为 0.480。''')
    caption = caption.replace('图中的 0.480 为 8 位患者相关系数的算术均值，横线也为患者效应均值。', '模块相关系数 0.480 为 8 位患者相关系数的算术均值；v7 的 H 改画两组患者平均表达秩，不再在图面展示该相关数值。')
    (ROOT/'FIGURE_CAPTION.md').write_text(caption)

    (HERE/'README.md').write_text('''# Melanoma figure v7

更新日期：2026-09-14。

- [整图 PNG](melanoma_three_row_redesign_v7.png) · [SVG](melanoma_three_row_redesign_v7.svg)
- [F](panels/F.png)：AUROC、AUPRC、Precision、Recall，每组 14 位患者。
- [G](panels/G.png)：原始 UMAP 坐标和分数不变，虚线圈为主要高分区域的视觉指引。
- [H](panels/H.png)：4 个蛋白 × High/Lower 两组，每组 8 位患者的平均表达秩；细线为患者配对。

High 为原分析的患者内最高得分四分位组，Lower 为其余细胞。H 的两组均值从冻结的差值及细胞数量精确恢复：患者内平均并列秩除以 n+1 的整体均值恒等于 0.5，故 High=0.5+(1-n_high/n)×差值，Lower=0.5-(n_high/n)×差值。组细胞数量已与原始查询细胞表核对，未生成新的细胞观测。

F 的四项指标已从冻结患者留出预测重新核算，与原表完全一致；Precision/Recall 沿用整位患者得分最高 20% 的截点。数据表位于 `data/`，没有重新训练模型或搜索阈值。

箱体为 25%–75% 分位数，中线为中位数，须线为 1.5 IQR 内最远观测；所有患者点均显示，包括须线以外的点。

复现（从 final_xu）：

```bash
python3 -B figure_melanoma/figure_layout_v7/prepare_visual_data.py
python3 -B figure_melanoma/figure_layout_v7/build_figure.py
```

核对：`data/DATA_QA.json`、`REVISION_QA.json`、`LAYOUT_AUDIT.json`、`manuscript_sync/COMPILE_QA.json`。A–E 像素保持一致。前版和更新前根文档分别保留在 v6、`previous_docs/`。

已同步 Figure 4、F–H 图注及两份 manuscript PDF；其余 TeX 正文不变。原稿备份：

`'''+sync['backup']+'''`
''')
    print('Updated README and bilingual caption; previous documents backed up.')


if __name__ == '__main__':
    main()
