from pathlib import Path
import json,shutil,hashlib,importlib.util
import pandas as pd
import numpy as np
C=Path(__file__).resolve().parents[1];MF=C/'maxfuse_full_panel';ROOT=C.parent.parent

def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p,n):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
for k in ['ramp','raw','harmony','fastmnn','cycombine','cytonorm']:
 assert (C/f'evaluation/replicates/{k}_COMPLETE.json').exists(),k
for i in range(3):
 assert (MF/f'evaluation/replicates/maxfuse_hub{i}_COMPLETE.json').exists()
for p in (MF/'evaluation').rglob('*'):
 if p.is_file():
  out=C/'evaluation'/p.relative_to(MF/'evaluation');out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out)
# Preserve every raw evaluation; use the unchanged numerical aggregator.
m=load(C/'scripts/summarize.py','frozen_aggregator')
def readme(cfg,avg,stats):
 (C/'README.md').write_text('''# Lung story comparison - Senosain140k reference

当前正式输入来自 `../releases/senosain140k_20260921`：84,000 cells，84 patients，9 shared proteins。

- 5 个比较方法重新拟合；RAMP 使用同一新版正式对齐值，Unaligned 单列。
- MaxFuse full-active protein panels：34 / 41 / 33；三个 hub 每 seed、每 metric 取中位数；全部 hub 输出保留。
- 全 31 metrics，20 个固定匹配重采样；所有方法抽样行及评价标签完全一致。所有全细胞 UMAP 按同一参数新拟合。
- `evaluation/metric_battery_summary.csv` 保存全部指标；`metric_battery_replicates_all_variants.csv` 保存每次抽样和全部 hub。
- 自动汇总产生的 main30 表只是历史辅助表，**Figure 2 仍固定使用 METRIC_POLICY.json 的 23 项和四类等权**，不使用 main30 作为主结果。
- `input/LABEL_COUNTS.csv` 为评价标签计数。Other / mixed 与原 Other / uncertain 一样不纳入标签指标；joint 条件仍为抽样中至少两个平台出现、总数至少 30。
- 模型、结果、原始表达、嵌入、参数、环境和来源哈希分别在 methods / input / maxfuse_full_panel / evaluation / logs。
- 重跑入口：`python3 -B scripts/orchestrate.py`；各模型及每 seed 结果均可校验后续跑。全部完成后运行 `python3 -B scripts/finalize_comparison.py`。

原 Comparison 只读保留。RAMP、比较方法均未按新分数调参；指标清单及权重来自前版，已在本次分数产生前冻结。旧版本的指标选择历史仍在 METRIC_POLICY.json 中。
''')
m.build_readme=readme;m.main()
stats=pd.read_csv(C/'evaluation/metric_battery_summary.csv');assert len(stats)==217 and stats.valid_resamples.eq(20).all()
old=pd.read_csv(C.parent/'Comparison/maxfuse_full_panel/evaluation/metric_battery_summary.csv');compare=stats.merge(old,on=['benchmark','method','metric','family','direction'],suffixes=('_new','_old'),validate='one_to_one');compare['new_minus_old']=compare['mean_new']-compare['mean_old'];compare.to_csv(C/'evaluation/OLD_NEW_ALL31.csv',index=False)
r=load(C/'scripts/render_overview.py','overview');r.main()
inputs=json.loads((C/'input/manifest.json').read_text());full=json.loads((MF/'data/manifest.json').read_text())
for p,h in full['source_sha256'].items():assert sha(p)==h
for p,h in full['output_sha256'].items():assert sha(MF/'data'/p)==h
# Verify row identity of all model outputs and intended numerical source scope.
cell_sha=sha(C/'input/cell_ids.npy');ids=np.load(C/'input/cell_ids.npy');raw=np.load(C/'input/raw.npy');meta=pd.read_csv(C/'input/fit_metadata.csv.gz')
for k in ['harmony','fastmnn','cycombine','cytonorm']:
 j=json.loads((C/f'methods/{k}/COMPLETE.json').read_text());assert j['cell_ids_sha256']==cell_sha and j['input_manifest_sha256']==sha(C/'input/manifest.json')
for v in full['views']:
 p=MF/'data'/v['slug'];rows=np.load(p/'global_row_indices.npy');assert np.array_equal(ids[rows],np.load(p/'cell_ids.npy'))
 mapping=pd.read_csv(p/'shared_mapping.csv');native=(p/'raw_feature_names.txt').read_text().splitlines();x=np.load(p/'raw_all.npy');assert np.max(np.abs(x[:,[native.index(n) for n in mapping.native_feature]]-raw[rows]))<1e-10
for i in range(3):
 j=json.loads((MF/f'methods/maxfuse_hub{i}/COMPLETE.json').read_text());assert j['cell_ids_sha256']==cell_sha
j=json.loads((C/'COMPLETE.json').read_text());j.update(status='PASS',full_active_maxfuse=True,full_panel_manifest_sha256=sha(MF/'data/manifest.json'),figure2_main_metrics=23,figure2_primary_weighting='four_categories_equal',all_methods_same_cells_and_evaluation_labels=True,old_other_six_tasks_to_be_reused=True)
(C/'COMPLETE.json').write_text(json.dumps(j,indent=2)+'\n')
print('Comparison, UMAPs, all31, old/new comparison and identity audit PASS')
