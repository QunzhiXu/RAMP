"""Collect completed comparisons without selecting metrics by their result."""
import os,sys,json,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(HERE/'.cache/evaluation/matplotlib');os.environ['MPLBACKEND']='Agg'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib.util
ROOT=HERE.parent.parent;SOURCE=ROOT/'benchmark_results/comparison_evaluation/scripts/evaluate_benchmark.py'
os.environ['RAMP_EVALUATION_CACHE']=str(HERE/'.cache/evaluation')
spec=importlib.util.spec_from_file_location('benchmark_eval_summary',SOURCE);e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
OUT=HERE/'evaluation'
ORDER=['Unaligned','RAMP','Harmony-direct','cyCombine','CytoNorm','fastMNN','MaxFuse']
COLORS={'Unaligned':'#AAAAAA','RAMP':'#007F73','Harmony-direct':'#4477AA','cyCombine':'#EEAA33','CytoNorm':'#AA3377','fastMNN':'#CC6677','MaxFuse':'#8866AA'}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 cfg=json.loads((HERE/'run_config.json').read_text());isthree=len(cfg['datasets'])==3
 expected=['raw','ramp','harmony','cycombine','cytonorm','fastmnn']+(['maxfuse_hub0','maxfuse_hub1','maxfuse_hub2'] if isthree else ['maxfuse'])
 missing=[k for k in expected if not (OUT/'replicates'/(k+'_COMPLETE.json')).exists()]
 if missing:raise RuntimeError('Incomplete evaluations: '+', '.join(missing))
 configs=[json.loads((OUT/'replicates'/(k+'_config.json')).read_text())['configuration'] for k in expected]
 for field in ['metadata_sha256','evaluator_source_sha256','adapter_sha256','input_manifest_sha256','seeds','cells_per_dataset','umap_protocol']:
  assert len({json.dumps(c[field],sort_keys=True) for c in configs})==1,('Evaluation protocol differs',field)
 full=pd.concat([pd.read_csv(OUT/'replicates'/(k+'.csv')) for k in expected],ignore_index=True)
 assert full.groupby('sampling_seed').sampled_rows_sha256.nunique().eq(1).all()
 full.to_csv(OUT/'metric_battery_replicates_all_variants.csv',index=False)
 registry=pd.DataFrame(e.REGISTRY,columns=['metric','family','direction','definition']);registry.to_csv(OUT/'metric_registry.csv',index=False)
 metrics=registry.metric.tolist()
 canonical=full.loc[~full.method.str.startswith('MaxFuse hub')].copy()
 if isthree:
  hub=full.loc[full.method.str.startswith('MaxFuse hub')]
  assert len(hub)==60
  med=hub.groupby(['benchmark','sampling_seed'],as_index=False)[metrics].median()
  med['method']='MaxFuse';med['aggregation']='Median across all 3 hub choices separately per seed and metric'
  canonical=pd.concat([canonical,med],ignore_index=True)
 canonical.to_csv(OUT/'metric_battery_replicates.csv',index=False)
 rows=[]
 for metric,family,direction,definition in e.REGISTRY:
  for method,group in canonical.groupby('method'):
   rows.append({'benchmark':cfg['case'],'method':method,'metric':metric,'family':family,'direction':direction,'mean':group[metric].mean(),'sd':group[metric].std(),'valid_resamples':group[metric].notna().sum(),'definition':definition})
 stats=pd.DataFrame(rows);stats.to_csv(OUT/'metric_battery_summary.csv',index=False)
 stats.loc[stats.valid_resamples.lt(20)].to_csv(OUT/'metrics_with_missing_resamples.csv',index=False)
 eligible=stats.loc[stats.method.ne('Unaligned')].copy()
 eligible['rank']=np.nan
 for metric,family,direction,_ in e.REGISTRY:
  mask=eligible.metric.eq(metric);eligible.loc[mask,'rank']=eligible.loc[mask,'mean'].rank(ascending=direction=='lower',method='average')
 eligible.to_csv(OUT/'metric_method_ranks_all31.csv',index=False)
 primary=eligible.loc[eligible.family.ne('geometry')];primary.to_csv(OUT/'metric_method_ranks_main30.csv',index=False)
 avg=primary.groupby('method',as_index=False).agg(mean_rank_30=('rank','mean'),n_metrics=('rank','count'))
 familyrank=primary.groupby(['method','family'],as_index=False)['rank'].mean();familyrank.to_csv(OUT/'family_mean_ranks.csv',index=False)
 avg=avg.merge(familyrank.groupby('method',as_index=False)['rank'].mean().rename(columns={'rank':'family_balanced_mean_rank'}))
 avg.sort_values('family_balanced_mean_rank').to_csv(OUT/'method_ranking.csv',index=False)
 # The same four interpretable endpoints for every story, chosen before scores are reviewed.
 chosen=[('leiden_celltype_ari','Cell-type ARI'),('highdim_celltype_asw_macro','Cell-type silhouette'),('highdim_dataset_asw_conditional','Within-type platform mixing'),('crossplatform_knn_macro_f1','Cross-platform label transfer')]
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,4,figsize=(14,3.5));rng=np.random.default_rng(42)
 for ax,(metric,title) in zip(axes,chosen):
  for j,name in enumerate(ORDER):
   v=canonical.loc[canonical.method.eq(name),metric].dropna().to_numpy()
   if len(v):
    box=ax.boxplot([v],positions=[j],widths=.57,patch_artist=True,showfliers=False,medianprops={'color':'#222222','linewidth':1},boxprops={'linewidth':.7},whiskerprops={'linewidth':.7},capprops={'linewidth':.7})
    box['boxes'][0].set_facecolor(COLORS[name]);box['boxes'][0].set_alpha(.3)
    ax.scatter(j+rng.uniform(-.13,.13,len(v)),v,s=7,c=COLORS[name],edgecolors='none',alpha=.8,zorder=3)
  ax.set_title(title,fontweight='bold');ax.set_xticks(range(len(ORDER)),ORDER,rotation=55,ha='right',fontsize=8)
  ax.grid(axis='y',color='#eeeeee',lw=.6);ax.set_axisbelow(True)
 fig.suptitle({'hnscc_story':'HNSCC','lung_story':'Lung','melanoma_story':'Melanoma'}[cfg['case']]+' · 20 matched resamples',fontweight='bold');fig.tight_layout()
 for ext in ['png','svg']:fig.savefig(OUT/('comparison_metrics.'+ext),dpi=220,bbox_inches='tight')
 plt.close(fig)
 # Full registry values remain available, including geometry.
 pivot=eligible.pivot(index='metric',columns='method',values='rank').reindex(index=metrics,columns=ORDER[1:])
 fig,ax=plt.subplots(figsize=(8.5,10.8));im=ax.imshow(pivot.to_numpy(),vmin=1,vmax=6,cmap='YlGnBu_r',aspect='auto')
 for i in range(len(pivot)):
  for j in range(len(pivot.columns)):
   v=pivot.iloc[i,j]
   if pd.notna(v):ax.text(j,i,f'{v:g}',ha='center',va='center',fontsize=8,color='white' if v<2.5 else '#222222')
 ax.set_xticks(range(len(pivot.columns)),pivot.columns,rotation=30,ha='right');ax.set_yticks(range(len(pivot)),[m.replace('_',' ') for m in pivot.index],fontsize=7)
 ax.set_title('All 31 metrics · rank within this story (1 = best)',fontweight='bold');fig.colorbar(im,ax=ax,fraction=.03,pad=.03)
 fig.tight_layout();fig.savefig(OUT/'all_metric_ranks.png',dpi=180);plt.close(fig)
 build_readme(cfg,avg,stats)
 runtime_notes=[]
 for log in sorted((HERE/'logs').glob('eval_*.log')):
  content=log.read_text(errors='replace')
  if 'Falling back to random initialisation' in content:runtime_notes.append(f'- `{log.name}`: UMAP spectral initialization failed; the package fell back to seeded random initialization. High-dimensional metrics are unaffected; the three UMAP metrics use the realized coordinates.')
  if 'disconnected from manifold' in content:runtime_notes.append(f'- `{log.name}`: UMAP reported disconnected vertices; inspect the complete log.')
 (OUT/'runtime_notes.md').write_text('# Runtime notes\n\n'+('\n'.join(runtime_notes) if runtime_notes else 'No spectral-initialization fallback or disconnected-manifold warning was logged.')+'\n')
 if runtime_notes:
  with (HERE/'README.md').open('a') as f:f.write('\n## 运行记录\n\n'+'\n'.join(runtime_notes)+'\n')
 # Verify original sources again after fitting and evaluation.
 manifest=json.loads((HERE/'input/manifest.json').read_text())
 source_ok={p:sha(p)==h for p,h in manifest['source_sha256'].items()};assert all(source_ok.values())
 outputs_ok={p:sha(HERE/'input'/p)==h for p,h in manifest['output_sha256'].items()};assert all(outputs_ok.values())
 report={'case':cfg['case'],'all_original_source_hashes_unchanged':source_ok,'frozen_input_outputs_unchanged':outputs_ok,'methods':expected,'resamples_per_method':20,'metric_count':31,'main_registry_excludes_geometry_only':30,'matched_evaluation_rows_across_methods':True,'artifacts':{p.name:sha(p) for p in OUT.glob('*.csv')}}
 (HERE/'COMPLETE.json').write_text(json.dumps(report,indent=2)+'\n')
 print('STORY COMPARISON COMPLETE',cfg['case'],flush=True)
def build_readme(cfg,avg,stats):
 manifest=json.loads((HERE/'input/manifest.json').read_text())
 table='| Dataset | Cells | Donors / units |\n|---|---:|---:|\n'+''.join(f"| {x['dataset']} | {x['cells']:,} | {x['units']} |\n" for x in manifest['preprocessing'])
 ranktable='| Method | Mean rank (30 metrics) | Family-balanced mean rank |\n|---|---:|---:|\n'+''.join(f"| {r.method} | {r.mean_rank_30:.3f} | {r.family_balanced_mean_rank:.3f} |\n" for r in avg.sort_values('family_balanced_mean_rank').itertuples())
 text=f'''# {cfg['case']}：原故事固定输入的比较方法结果

本次仅写入当前 Comparison 文件夹。所有方法使用原 RAMP 的同一批细胞、顺序和共同 marker。RAMP 复用原存档，不重新拟合。原始输入文件在完成后再次核对 SHA-256。

{table}
共同 marker（{len(cfg['markers'])}）：{', '.join(cfg['markers'])}。

## 查看结果

- `evaluation/comparison_metrics.png`：统一四项代表指标，点为 20 次配对抽样。
- `evaluation/umap_overview_celltype.png` / `umap_overview_dataset.png`：所有方法紧凑总览。
- `evaluation/umap/plots/`：每个方法的全细胞 UMAP，分别按平台和细胞类型着色；对应坐标及拟合参数在 `coordinates/`、`manifests/`。
- `evaluation/metric_battery_summary.csv`：全部 31 项指标的均值、标准差及有效抽样数。
- `evaluation/metric_battery_replicates.csv`：方法级每次抽样原始分数。
- `evaluation/metric_battery_replicates_all_variants.csv`：包括全部 MaxFuse hub 版本，不挑选最佳 hub。
- `evaluation/metric_method_ranks_main30.csv`：沿用当前 Figure 2 的 30 项主指标；geometry 仍完整保留在 all31 文件。
- `methods/<method>/values.npy`：完整细胞输出；fastMNN 和 MaxFuse 为嵌入坐标，其他为 marker 轴数值。
- `input/manifest.json` 与 `COMPLETE.json`：来源、数据完整性及完成核验。

## 排名

{ranktable}
排名越低越好。Family-balanced rank 先在 biology、mixing、transfer、matching、visual 各类内部取平均排名，再等权平均各类；普通 30 项平均排名也同时列出。未对较差指标做结果驱动的删除。Unaligned 不进入六种方法的排名。

## 固定输入与拟合设置

Lung/HNSCC 直接读取原始存档矩阵。Melanoma 按原脚本的两个 build 函数，从已存 shared13 和原始 LAG3、ICOS、OX40 列拼回 clean15；核对原运行记录的源文件哈希、完整 cell ID、平台及 donor 顺序。没有从 RAMP 输出倒推原值，也没有重新抽样。

所有比较方法都只读取 `fit_metadata`（cell ID、平台、donor/unit）和共同 marker 数值。细胞类型仅在拟合后用于评价。坐标、RNA、临床结局及故事的非共享预测目标都不参与比较方法拟合。

- Harmony-direct 1.2.3：source-scale marker 轴直接校正，无 PCA；沿用锁定的默认参数。
- fastMNN / batchelor 1.18.1：沿用 k=20、cosine normalization、batch-balanced PCA、固定批次顺序；维度为 min(50, marker 数−1)。
- cyCombine 0.2.5：官方跨平台 profile，rank normalization、8×8 SOM、rlen=10、seed=840、parametric ComBat。
- CytoNorm 2.0.6：沿用已有 benchmark 的 batch aggregate 适配；每 donor 最多 2,000 个训练细胞，15×15 FlowSOM、10 clusters、99 quantiles、mean goal、seed=42。聚合群体不是配对技术控制，因此这是跨独立 cohort 的适配，不能据此声称满足了原方法的所有控制样本假设。完整细胞均输出；FCS 行身份和输出 spline replay 已核对。
- MaxFuse 0.0.2：沿用锁定 pairwise profile，SVD/CCA 维度显式限制为 marker 数−1；全部细胞获得嵌入。Lung 跑三个 hub 的 Archive 三视图方法，按每个 seed、每个 metric 取三者中位数；两平台故事用 pairwise CCA。

fastMNN/cyCombine/CytoNorm 输入采取预先固定的 assay 变换：CyTOF asinh(x/5)，原始 IMC/mIF 强度 asinh(x/1)，原始 ADT counts log1p；Haist 已处理且含负值的矩阵直接使用。MaxFuse 再按各 dataset、各 marker 作 sample-SD z-score（ddof=1）。未根据标签、UMAP 或评价结果调参。

**MaxFuse 的特征范围：本轮严格 active=shared，没有额外全 panel 特征。旧四组 benchmark 的 MaxFuse 曾采用 native active panel，因此后续合并七组时须标明此差异，或统一特征协议后再作相同协议的总表述。**

仅本地适配器将 cyCombine/CytoNorm wrapper 的“必须三个 batch”检查拓展为“两或三个”。官方数值实现不改；原脚本、适配副本、逐行 diff 和哈希见 `scripts/adapter_manifest.json`。

## 评价协议

复用原 `benchmark_results/comparison_evaluation/scripts/evaluate_benchmark.py` 的 31 项实现和 20 个固定 seed。每次每平台随机抽取 5,000 个细胞，各方法使用完全相同的行。所有方法先作 pooled dimension-wise StandardScaler，不额外 PCA；全细胞 UMAP 统一 n_neighbors=30、min_dist=0.15、Euclidean、seed=42。因此这里新生成的 UMAP 可以与故事排版时的 UMAP 形状不同，RAMP 对齐值仍是同一份。

标签使用原故事已有的统一类型映射；Melanoma 用既有 provider→coarse 映射扩展到全队列。未能可靠统一的类型不参与 joint-label 指标；某次抽样中需在至少两个平台出现且总数≥30。全数据的混合指标仍包含非共享细胞类型。独立 cohort 的真实组成不平衡会影响混合分数，不能把完美混合作为唯一目标。

Cross-platform 15-NN F1 / balanced accuracy 衡量冻结对齐空间里的细胞类型标签转移；对齐已使用全部无标签细胞，因此这是 transductive evaluation，不是重做 donor-heldout alignment，也不是 biomarker 留出预测。

20 个点反映固定对齐矩阵的抽样波动，**不是 20 次重新拟合，也不是 20 个独立 cohort**。三个新增故事是额外对齐配置，部分源队列与旧 benchmark 重叠，不应称为七个完全独立验证队列。

## 复现

在当前 Comparison 下运行（全部写入此目录）：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/prepare.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_r_methods.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_maxfuse.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_evaluation.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/summarize.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/render_overview.py
```

环境和参数：各 `methods/` 的 manifest、sessionInfo 与 COMPLETE 文件；运行日志在 `logs/`。缺失或失败检查不会伪造分数；可从 checkpoint 继续。
'''
 (HERE/'README.md').write_text(text)
if __name__=='__main__':main()
