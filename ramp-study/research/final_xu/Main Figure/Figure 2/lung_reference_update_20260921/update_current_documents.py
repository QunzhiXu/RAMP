"""Promote checked Figure 2 outputs and point the lung story to its new comparison."""
from pathlib import Path
import json
import shutil
import os
import re

import pandas as pd

B = Path(__file__).resolve().parent
F = B.parent
ROOT = B.parents[2]
LUNG = ROOT / 'story_lung_HLA_ABC'


def markdown(frame):
    lines = ['| ' + ' | '.join(frame.columns) + ' |',
             '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    for row in frame.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in row) + ' |')
    return '\n'.join(lines)


def backup(path, relative):
    target = B / 'previous_root_documents' / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not target.exists():
        shutil.copy2(path, target)


def main():
    verified = json.loads((B / 'RELEASE_VERIFICATION.json').read_text())
    assert verified['status'] == 'PASS' and verified.get('visual_review_passed') is True
    ranks = pd.read_csv(B / 'selection/primary_overall_method_ranking.csv').sort_values('mean_rank')
    tasks = pd.read_csv(B / 'selection/primary_method_ranking_by_benchmark.csv')
    lung = tasks[tasks.benchmark.eq('lung_story')].sort_values('mean_rank')
    claims = json.loads((B / 'CHECKED_CLAIMS.json').read_text())
    table = markdown(ranks[['method', 'mean_rank', 'overall_position']])
    scope = '''本次只更新 Lung story：Senosain 参考细胞来自当前 HLA-ABC 故事的 140k 版本；Cords、Zheng 参考细胞身份保持不变。RAMP 使用新版故事已完成的对齐；Harmony-direct、fastMNN、cyCombine、CytoNorm 和 MaxFuse 全部重新拟合。每种方法使用同样的 84,000 cells、评价注释和 20 次匹配抽样。MaxFuse 保留完整 active protein panels（34 / 41 / 33），并按原协议整合三种 hub。

其余六组 benchmark 的原始得分不变。A–D 仍使用原来的 23 项指标：细胞类型结构 5 项、平台/donor 混合 14 项、标签转移 2 项、UMAP 2 项；四类各占 25%，七个任务等权。E/F 仍是原四组的 fixed-gate transfer，像素及数值核对不变。本次不更新独立的 cross-platform trainability 实验。

23 项清单最初是在查看结果后选定；本次沿用原清单及权重，在新版结果产生前冻结。完整 31 项、单指标等权和四类等权结果均保留。七个任务可能共享来源队列。
'''
    body = '# Figure 2：新版 Lung story 比较结果\n\n' + scope
    body += '\n## 当前综合名次\n\n' + table + '\n\n'
    body += 'RAMP: ' + claims['RAMP_rank_distribution'] + '.\n\n'
    body += '新版 Lung story：\n\n' + markdown(lung[['method', 'mean_rank', 'position']]) + '\n\n'
    body += '''## 文件和复现

- `Combined/figure_2_lung_reference_update.pdf`：整图；另有 PNG、SVG。
- `Panel A`–`Panel F`：分图、脚本和实际绘制数值。
- `selection/`：新旧差异、23/31 项表和全部敏感性汇总。
- `FIGURE_CAPTION.md`：完整中英文图注。
- `CHECKED_CLAIMS.json`：正文中的排名及优势陈述逐项核对。
- `RELEASE_VERIFICATION.json`、`visual_qa/`：数字、图件和编译稿核验。
- `manuscript_stage/`：用于安装到 bioRxiv_Ver1 的编译稿；`INSTALL_VERIFY.json` 记录安装结果。
- `previous_root_documents/`：被替换的根目录说明备份。

重现顺序：完成 `story_lung_HLA_ABC/Comparison_senosain140k_20260921` 后，运行 `build_selection.py`、`build_all.py`。`stage_manuscript.py` 仅用于首次构建独立稿件副本，不覆盖已有 stage。
'''
    (B / 'README.md').write_text(body)
    for name in ['README.md', 'FIGURE_CAPTION.md', 'CURRENT_DRAFT.json', 'CURRENT_DRAFT.md']:
        backup(F / name, Path('figure2') / name)
    root_text = body.replace('# Figure 2：新版 Lung story 比较结果', '# Figure 2：整合性能与蛋白门槛转移', 1)
    root_text = root_text[:root_text.index('## 文件和复现')]
    prefix = 'lung_reference_update_20260921/'
    root_text = root_text.replace('\n\n本次只更新', '\n\n[整图 PDF](' + prefix + 'Combined/figure_2_lung_reference_update.pdf) · [PNG](' + prefix + 'Combined/figure_2_lung_reference_update.png) · [Caption](FIGURE_CAPTION.md) · [完整来源和复现](' + prefix + 'README.md)\n\n本次只更新', 1)
    root_text += '当前绘图目录：`lung_reference_update_20260921/`。旧版本 `louvain_ari_fastmnn_20260914/` 和历史 `metric_selection_main23/` 原样保留；当前表见新版目录下的 `selection/`。\n'
    (F / 'README.md').write_text(root_text)
    caption = (B / 'FIGURE_CAPTION.md').read_text()
    caption = re.sub(r'\[([^\]]+)\]\(([^)]+)\)',
        lambda match: '[' + match[1] + '](' + os.path.relpath((B / match[2]).resolve(), F) + ')', caption)
    (F / 'FIGURE_CAPTION.md').write_text(caption)
    (F / 'CURRENT_DRAFT.json').write_text(json.dumps({
        'version': B.name, 'figure': prefix + 'Combined/figure_2_lung_reference_update.pdf',
        'selection': prefix + 'selection', 'comparison': 'story_lung_HLA_ABC/Comparison_senosain140k_20260921',
        'scope': 'Only lung_story refreshed; E/F gating and other six integration tasks unchanged',
    }, indent=2) + '\n')
    (F / 'CURRENT_DRAFT.md').write_text('# Current Figure 2\n\n'
        '[PDF](' + prefix + 'Combined/figure_2_lung_reference_update.pdf) · '
        '[PNG](' + prefix + 'Combined/figure_2_lung_reference_update.png) · '
        '[Source and verification](' + prefix + 'README.md)\n\n'
        'Lung story uses the current Senosain140k reference. The other six integration tasks and E/F gating are unchanged.\n')
    for name in ['README.md', 'CURRENT_ANALYSIS.json']:
        backup(LUNG / name, Path('lung_story') / name)
    pointer = json.loads((LUNG / 'CURRENT_ANALYSIS.json').read_text())
    pointer['comparison'] = 'Comparison_senosain140k_20260921'
    pointer['comparison_figure'] = '../Main Figure/Figure 2/lung_reference_update_20260921'
    (LUNG / 'CURRENT_ANALYSIS.json').write_text(json.dumps(pointer, indent=2) + '\n')
    p = LUNG / 'README.md'
    text = p.read_text()
    text = text.replace('未重跑或替换 Comparison、Figure 2 或 trainability benchmark。',
        '新版比较方法与 Figure 2 已更新，见下方 `Comparison_senosain140k_20260921`；旧 `Comparison` 及独立 trainability benchmark 保持原样。')
    note = '\n\n## 新版参考细胞的比较方法评价（2026-09-21）\n\n' + \
        '[Comparison_senosain140k_20260921](Comparison_senosain140k_20260921/README.md) 使用与当前正式故事一致的 84,000 个参考细胞，重跑五个比较方法及全部 31 项指标；Figure 2 使用原 23 项、四类等权汇总。旧 `Comparison` 目录保留。独立 trainability benchmark 不在此次更新范围内。\n'
    assert '## 新版参考细胞的比较方法评价（2026-09-21）' not in text
    p.write_text(text + note)
    print('Current figure and comparison pointers updated; prior root documents backed up.')


if __name__ == '__main__':
    main()
