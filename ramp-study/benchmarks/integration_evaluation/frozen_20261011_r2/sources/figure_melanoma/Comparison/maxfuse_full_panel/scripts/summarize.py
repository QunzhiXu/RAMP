"""Update the six-method comparison using full-panel MaxFuse; retain joint-only runs."""
import os,sys,json,hashlib,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parents[1];COMP=HERE.parent
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
for k,n in [('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba'),('XDG_CACHE_HOME','xdg')]:os.environ[k]=str(HERE/'.cache/evaluation'/n)
import pandas as pd
import numpy as np
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def loadmodule(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def readme(cfg,avg,stats):
 manifest=json.loads((HERE/'data/manifest.json').read_text())
 table='| 平台 | 细胞 | donor | 原始通道 | Active 蛋白 | Shared |\n|---|---:|---:|---:|---:|---:|\n'+''.join(f"| {v['dataset']} | {v['cells']:,} | {v['donors']} | {v['raw_features']} | {v['active_features']} | {v['shared_features']} |\n" for v in manifest['views'])
 ranking='| 方法 | 30 项平均排名 ↓ | 各类等权平均排名 ↓ |\n|---|---:|---:|\n'+''.join(f'| {r.method} | {r.mean_rank_30:.3f} | {r.family_balanced_mean_rank:.3f} |\n' for r in avg.sort_values('mean_rank_30').itertuples())
 exclusions='\n'.join(f"- **{v['dataset']}**："+('；'.join(f'{k}: {reason}' for k,reason in v['excluded_features'].items()) or '无额外排除。') for v in manifest['views'])
 eval_commands='\n'.join('PYTHONDONTWRITEBYTECODE=1 python3 scripts/evaluate.py '+k for k in (['maxfuse_hub0','maxfuse_hub1','maxfuse_hub2'] if len(cfg['datasets'])==3 else ['maxfuse']))
 text=f'''# {cfg['case']}：MaxFuse 完整蛋白 panel 重跑

已完成。此文件夹保存按原 RAMP 精确细胞范围重建的 MaxFuse 全 panel 数据、模型、UMAP 和指标。上一轮 joint-only 结果保留在上一级 `Comparison`，未被覆盖。

{table}
## 数据位置

每个平台均在 `data/<平台>/` 单独保存：

- `raw_all.npy` + `raw_feature_names.txt`：该批细胞的全部原始蛋白/技术通道，包含被 QC 排除的通道；不包含标签、空间坐标或临床数值。
- `active_raw.npy`、`active_preprocessed.npy`、`active.npy`：可用蛋白的原值、assay 变换后值及最终 z-score 输入。
- `shared.npy`：同一份 active 输入中切出的共同 marker，列顺序与 RAMP 一致。
- `cell_ids.npy`、`cell_metadata.csv.gz`、`global_row_indices.npy`：精确原始细胞顺序、平台及 donor。
- `feature_inventory.csv`、`shared_mapping.csv`、`mean.npy`、`scale.npy`：完整通道清单、保留/排除理由、对应列及可重现的变换参数。
- `data/manifest.json`：所有输入来源、哈希和逐平台核验；原始来源全程只读。

仅 Lung 数据包的 Cords 部分额外保存源 counts 与作者 c_counts_asinh 的完整二进制矩阵及原始 SCE 行身份（`data/cords_imc`）。`input` 是指向上一级冻结共同输入的只读用途软链接，仅供评价复用原始元数据、RAMP 和 raw reference；实际 MaxFuse 拟合读取本文件夹 `data`。

## 拟合协议

使用 MaxFuse 0.0.2 和原四组锁定 pairwise profile：shared 为原 RAMP 共同蛋白，active 为各平台全部可用蛋白。保留原始细胞、donor 和顺序，无重新抽样；无 cell type、RNA、坐标或结局参与拟合。

各 assay 的变换延续这三组上一轮设置：Senosain asinh(x/5)，IMC/mIF asinh(x/1)，ADT counts log1p，Haist 作者提供的带符号处理值直接使用。随后各平台逐蛋白按样本标准差 z-score（ddof=1）；shared 是 active 的精确列切片。**这里统一的是 full-active 特征策略和 MaxFuse 数值 profile，不能笼统称所有七组的 assay 预处理完全一致**；例如旧 bone marrow ADT 使用全 panel CLR，旧 Cords 使用作者 p99 截断后的 c_counts_asinh，而本轮继续使用与此前三组相同的明确变换。

SVD/CCA 维度用旧 benchmark 的原始 `resolve_pairwise_dimensions` 规则按 shared/active 各自宽度分别机械裁剪；不再把 active 维度限制在 joint 宽度内。两平台使用 pairwise CCA；Lung 使用三个 cyclic hub/order 的官方 Archive 三视图方法，跳过蛋白分支 SVD，GCCA=15。所有 hub 均保留，按每个 seed、每个 metric 取三者中位数，不选择表现最好的 hub。参数文件、维度适配及完整命令见 `data/*config*.json`、`data/*dimension_audit.json` 和 `logs/*execution.json`。

维度和 hub 顺序的精确变化见 `PROTOCOL_DIFFERENCES.json`。其中 Lung 的 hub1 恢复为旧四组使用的 cyclic 顺序。因此新旧对照比较两套完整配置，而非固定维度的单变量消融。

### 预先固定的通道筛选

{exclusions}

原始通道全部保存，排除仅作用于 active 拟合输入。Moldoveanu 排除依据本地既有 provider-paper QC 清单；Senosain CD90 沿用旧 MaxFuse 排除原因。没有依据本轮结果筛选蛋白。

完整 active 中允许出现故事关注的蛋白（如 HLA-A、HLA-ABC、CD39、TIGIT、LAG3），因为本轮评估的是无标签整合与细胞类型结构。这些结果**不作为**相应 biomarker 留出预测或空间回投验证；原故事 RAMP 预测和验证分析没有修改。

## 查看结果

- `evaluation/comparison_metrics.png`：四项固定代表指标，所有六种方法与 raw reference。
- `evaluation/umap_overview_celltype.png` / `umap_overview_dataset.png`：全细胞 UMAP 总览。
- `evaluation/umap/plots/maxfuse*.png`：重跑 MaxFuse 单独 UMAP；相应完整坐标和参数见同级 coordinates/manifests。
- `methods/maxfuse*/values.npy`：回到原 RAMP 全细胞行顺序的完整嵌入。
- `evaluation/metric_battery_summary.csv`：31 项均值、标准差、有效抽样数。
- `evaluation/metric_battery_replicates.csv`：7 组（含 raw）×20 次抽样。
- `evaluation/metric_battery_replicates_all_variants.csv`：Lung 包含三个 hub 的所有结果。
- `evaluation/full_vs_joint_only_maxfuse.csv`：完整 panel 与上一轮 joint-only 的所有指标差异。
- `evaluation/method_ranking.csv`：新版本六方法排名；原 joint-only 排名在上一级。

## 更新后的排名

{ranking}
不是每个单项的优胜声明；mean rank 按原有 30 项主指标计算，geometry 指标仍完整保留在 all31。family-balanced 先在五类内部取平均排名，再等权平均，不按结果增删指标。

## 评价与复用

重跑 MaxFuse 使用上一级未修改的 `scripts/evaluate.py` 和原 benchmark 的 31 项数值实现，只重定向输出目录。所有原 RAMP、Harmony-direct、fastMNN、cyCombine、CytoNorm、raw 评价直接复用。20 个 seed、每平台 5000 细胞、标签及 joint mask、UMAP 参数与之前完全一致；每次抽样的行哈希跨方法核对一致。所有 31 项应各有完整 20 次结果。

UMAP：pooled StandardScaler，无额外 PCA，n_neighbors=30，min_dist=0.15，Euclidean，seed=42，全量细胞。20 次是固定对齐结果上的重复抽样，不是 20 次对齐拟合，也不是独立队列。细胞类型转移是 transductive 评价，并非 donor-heldout 或 biomarker-heldout 预测。

## 复现

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/prepare.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_maxfuse.py
# 对每个已完成的 maxfuse / maxfuse_hub0 / maxfuse_hub1 / maxfuse_hub2：
{eval_commands}
PYTHONDONTWRITEBYTECODE=1 python3 scripts/summarize.py
```

仅 Lung 从原始 SCE 重新导出时需要先运行 `scripts/export_cords.R` 按 `data/cords_imc/requested_cells.csv` 导出原始 SCE 对应细胞，可复现命令见 Lung 包中的 `CORDS_EXPORT_COMMAND.json`，运行记录见 `logs/export_cords.log`。已冻结数据和模型存在时脚本核验哈希后复用。
'''
 (HERE/'README.md').write_text(text)

def main():
 if (HERE/'COMPLETE.json').exists():
  prior=json.loads((HERE/'COMPLETE.json').read_text())
  if prior.get('full_panel_input_manifest_sha256')==sha(HERE/'data/manifest.json'):
   assert all(sha(HERE/'evaluation'/p)==h for p,h in prior['artifacts'].items())
   print('FULL-PANEL SUMMARY VERIFIED CACHED',flush=True);return
 m=loadmodule(COMP/'scripts/summarize.py','unchanged_summarizer');m.HERE=HERE;m.OUT=HERE/'evaluation';m.build_readme=readme
 m.main()
 new=pd.read_csv(HERE/'evaluation/metric_battery_summary.csv');old=pd.read_csv(COMP/'evaluation/metric_battery_summary.csv')
 a=new[new.method.eq('MaxFuse')];b=old[old.method.eq('MaxFuse')]
 compare=a[['metric','direction','mean','sd']].merge(b[['metric','mean','sd']],on='metric',suffixes=('_full_panel','_joint_only'));compare['change_full_minus_joint']=compare.mean_full_panel-compare.mean_joint_only
 compare.to_csv(HERE/'evaluation/full_vs_joint_only_maxfuse.csv',index=False)
 assert new.valid_resamples.eq(20).all()
 r=loadmodule(COMP/'scripts/render_overview.py','unchanged_renderer');r.HERE=HERE;r.OUT=HERE/'evaluation';r.main()
 parent_notes=(COMP/'evaluation/runtime_notes.md').read_text() if (COMP/'evaluation/runtime_notes.md').exists() else 'No parent notes available.'
 new_notes=(HERE/'evaluation/runtime_notes.md').read_text()
 (HERE/'evaluation/runtime_notes.md').write_text('# Runtime notes\n\n## New full-panel MaxFuse evaluation\n\n'+new_notes+'\n## Reused evaluations: original runtime notes\n\n'+parent_notes)
 with (HERE/'README.md').open('a') as f:f.write('\n运行告警（包含复用结果的原记录）见 `evaluation/runtime_notes.md`；Melanoma 原 fastMNN UMAP 的谱初始化回退仍属于复用坐标的记录。\n')
 mf=json.loads((HERE/'data/manifest.json').read_text());sources={p:sha(p)==h for p,h in mf['source_sha256'].items()};outputs={p:sha(HERE/'data'/p)==h for p,h in mf['output_sha256'].items()};assert all(sources.values()) and all(outputs.values())
 for key in ['raw','ramp','harmony','fastmnn','cycombine','cytonorm']:
  for suffix in ['.csv','_config.json','_COMPLETE.json']:
   assert sha(HERE/'evaluation/replicates'/(key+suffix))==sha(COMP/'evaluation/replicates'/(key+suffix))
 report=json.loads((HERE/'COMPLETE.json').read_text());report.update(full_panel_input_manifest_sha256=sha(HERE/'data/manifest.json'),full_panel_sources_unchanged=sources,full_panel_frozen_data_unchanged=outputs,old_other_method_evaluations_reused_byte_identically=True)
 (HERE/'COMPLETE.json').write_text(json.dumps(report,indent=2)+'\n');(HERE/'STATUS.json').write_text(json.dumps({'status':'complete','maxfuse_full_panel_refit':True,'metrics':31,'matched_resamples':20},indent=2)+'\n')
 marker='## MaxFuse 完整 panel 更新（2026-09-14）'
 p=COMP/'README.md';t=p.read_text()
 if marker not in t:p.write_text(t+'\n'+marker+'\n\n新增三组已用完整 native active protein panel 重跑 MaxFuse。**更新后的比较、UMAP 和排名见 [maxfuse_full_panel/README.md](maxfuse_full_panel/README.md)。** 本 README 上方及本层 evaluation 仍是保留的 joint-only 版本，不能与更新版混用。其他方法及原 RAMP 直接复用。\n')
 current=p.read_text()
 if '> **最新结果：**' not in current:
  first,rest=current.split('\n',1)
  p.write_text(first+'\n\n'+'> **最新结果：** MaxFuse 已使用完整 native active protein panel 重跑，见 [maxfuse_full_panel/README.md](maxfuse_full_panel/README.md)。本页下方及本层 `evaluation/` 保留的是上一轮 joint-only 结果。\n\n'+rest.lstrip('\n'))
 print('FULL-PANEL COMPARISON COMPLETE',mf['case'],flush=True)
if __name__=='__main__':main()
