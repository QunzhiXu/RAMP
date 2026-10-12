from pathlib import Path
import json,hashlib,os
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent/".mplconfig"))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullLocator
R=Path(__file__).resolve().parent
m=json.loads((R/'MANIFEST.json').read_text());complete=json.loads((R/'RUN_COMPLETE.json').read_text());assert complete['all_passed']
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[]
for kind,items in [('baseline',m['baselines']),('new',m['jobs'])]:
 for j in items:
  p=Path(j['result']) if kind=='baseline' else R/'jobs'/j['id']/'RESULT.json'
  if kind=='baseline':assert h(p)==j['result_sha256']
  else:
   receipt=json.loads((p.parent/'PILOT_COMPLETE.json').read_text());assert h(p)==receipt['result_sha256']
  r=json.loads(p.read_text());sel=r['selection']
  rows.append({k:j[k] for k in ['task','source','target','seed','fit_budget','width']}|dict(origin=kind,macro_f1=r['score']['macro_f1'],balanced_accuracy=r['score']['balanced_accuracy'],accuracy=r['score']['accuracy'],source_validation_macro_f1=sel['source_validation_score'],parameters=sel['trainable_head_parameters'],selected_epoch=sel['selected_epoch'],selected_scale=sel['candidate']['scale'],selected_alpha=sel['candidate']['alpha'],selected_learning_rate=sel['candidate']['learning_rate'],train_seconds=r['training_seconds_including_selection'],result=str(p)))
df=pd.DataFrame(rows);keys=['task','source','target'];assert not df.duplicated(keys+['fit_budget','width']).any();df.to_csv(R/'ALL_SCORES.csv',index=False)
per_task=df.groupby(['task','fit_budget','width'],as_index=False).agg(macro_f1=('macro_f1','mean'),balanced_accuracy=('balanced_accuracy','mean'),directions=('macro_f1','size'),source_validation_macro_f1=('source_validation_macro_f1','mean'));per_task.to_csv(R/'PER_TASK.csv',index=False)
# Pair every requested large-budget endpoint with its 2k baseline on exactly the same directions.
base=df[(df.fit_budget==2000)&(df.width==32)]
pairs=[];curves=[]
for budget in [5000,10000,20000,50000]:
 end=df[(df.fit_budget==budget)&(df.width==32)]
 paired=base.merge(end,on=keys,suffixes=('_2000','_large'),validate='one_to_one')
 assert len(paired)==len(end)
 paired['delta_macro_f1']=paired.macro_f1_large-paired.macro_f1_2000
 paired['requested_budget']=budget;pairs.append(paired)
 support=end[keys]
 for b in [2000]+[v for v in [5000,10000,20000,50000] if v<=budget]:
  part=df[(df.fit_budget==b)&(df.width==32)].merge(support,on=keys,validate='one_to_one')
  assert len(part)==len(support)
  taskvals=part.groupby('task').macro_f1.mean()
  curves.append(dict(support_max_budget=budget,fit_budget=b,score=float(taskvals.mean()),benchmarks=len(taskvals),directions=len(part)))
pairs=pd.concat(pairs,ignore_index=True);pairs.to_csv(R/'D_PAIRED_DIRECTION_COMPARISONS.csv',index=False)
pt=pairs.groupby(['requested_budget','task'],as_index=False).agg(baseline_2000=('macro_f1_2000','mean'),new_score=('macro_f1_large','mean'),delta=('delta_macro_f1','mean'),directions=('delta_macro_f1','size'));pt.to_csv(R/'D_PAIRED_TASK_COMPARISONS.csv',index=False)
d_over=pt.groupby('requested_budget',as_index=False).agg(baseline_2000=('baseline_2000','mean'),new_score=('new_score','mean'),delta=('delta','mean'),benchmarks=('task','size'),directions=('directions','sum'));d_over.to_csv(R/'D_MATCHED_SUPPORT_SUMMARY.csv',index=False)
curves=pd.DataFrame(curves);curves.to_csv(R/'D_MATCHED_SUPPORT_CURVES.csv',index=False)
# E: all 34 directions and seven tasks at 5k, original one-seed 500 anchors.
e=per_task[per_task.fit_budget.isin([500,5000])].groupby(['fit_budget','width'],as_index=False).agg(score=('macro_f1','mean'),benchmarks=('task','size'),directions=('directions','sum'),source_validation_macro_f1=('source_validation_macro_f1','mean'))
assert (e.benchmarks==7).all() and (e.directions==34).all();e.to_csv(R/'E_COMMON_SUPPORT_SUMMARY.csv',index=False)
epivot=per_task[per_task.fit_budget==5000].pivot(index='task',columns='width',values='macro_f1');epivot['gain_128_vs_32']=epivot[128]-epivot[32];epivot['gain_512_vs_32']=epivot[512]-epivot[32];epivot.to_csv(R/'E_PER_TASK_CAPACITY.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titleweight':'bold','axes.labelweight':'bold','axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,4,figsize=(16,4.0),layout='constrained')
for ax,maxb,letter in zip(axes[:3],[10000,20000,50000],['A','B','C']):
 c=curves[curves.support_max_budget==maxb]
 ax.plot(c.fit_budget,c.score,'o-',color='#007D79',lw=2,ms=5)
 for x,y in zip(c.fit_budget,c.score):ax.annotate(f'{y:.3f}',(x,y),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
 ax.set_xscale('log');ax.xaxis.set_minor_locator(NullLocator());ax.set_xticks(c.fit_budget);ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x/1000:g}k'))
 ax.set_ylim(.15,.9);ax.grid(axis='y',color='#e6e6e6');ax.set_xlabel('Source training cells')
 ax.set_title(f'D extension: matched support to {maxb//1000}k\n{int(c.benchmarks.iloc[0])} benchmark(s), {int(c.directions.iloc[0])} direction(s)',fontsize=10)
 ax.set_ylabel('Target macro-F1' if letter=='A' else '')
ax=axes[3]
for b,color in [(500,'#898989'),(5000,'#007D79')]:
 v=e[e.fit_budget==b];ax.plot(v.width,v.score,'o-',color=color,label=f'{b:,} training cells',lw=2,ms=5)
ax.set_xscale('log',base=2);ax.xaxis.set_minor_locator(NullLocator());ax.set_xticks([8,32,128,512],labels=['8','32','128','512']);ax.set_ylim(.4,.7);ax.set_xlabel('Hidden-layer width');ax.set_ylabel('Target macro-F1');ax.grid(axis='y',color='#e6e6e6');ax.set_title('E extension: model capacity\n7 benchmarks, 34 directions',fontsize=10);ax.legend(frameon=False,fontsize=9,loc='upper left')
fig.suptitle('RAMP exploratory pilot: one seed; 512 source-validation cells at every condition',fontsize=12,fontweight='bold')
fig.savefig(R/'ramp_large_budget_capacity_pilot.png',dpi=200);fig.savefig(R/'ramp_large_budget_capacity_pilot.svg');plt.close(fig)
def fmt(f):
 def val(x):
  if isinstance(x,(float,np.floating)):return f'{x:.4f}'
  return str(x)
 lines=['| '+' | '.join(map(str,f.columns))+' |','| '+' | '.join(['---']*len(f.columns))+' |']
 lines += ['| '+' | '.join(val(x) for x in row)+' |' for row in f.itertuples(index=False,name=None)]
 return '\n'.join(lines)
old3=pd.read_csv(R.parent/'Figure/Panel E/curve_means.csv');old3=old3[old3.method.eq('ramp')]
report='''# RAMP 大样本与网络容量探索结果

本轮为 **RAMP-only、1 个固定随机种子**的探索；复用冻结对齐矩阵，保持原患者/单元拆分、标签体系、目标细胞、512 个源验证细胞、8 个源验证候选及患者等权评分。正式图、正式三次重复结果和 manuscript 均未改动。

## D：扩大训练样本

每一行都在其所支持的**同一批方向**上比较 2k 与新增预算；不同行支持不同，不能把其均值连成一条完整七组学习曲线。

'''+fmt(d_over)+'''

100k 在冻结数据内没有可用方向。50k 只支持 HNSCC 的 CODEX → ADT，一个方向，不能代表七个 benchmark 的整体表现。5k 支持全部 34 个方向。

## E：提高训练预算后增大网络

下表全部基于七个 benchmark、34 个方向，按 benchmark 等权。500-cell 行是原正式结果中同一随机种子的结果，不能与三种子均值混用。

'''+fmt(e)+'''

各任务在 5k 下的容量效果：

'''+fmt(epivot.reset_index())+'''

原正式三种子 E：

'''+fmt(old3)+'''

## 如何使用

- 本轮能判断更多标签/更宽网络是否提高 RAMP 的目标平台预测，不代表与其他方法的新比较。
- 保留完整支持/排除记录。数据不足不补采重复细胞，也不把不足预算写成已测试该预算。
- 若目标分数仍接近平稳，E 的主要信息是容量饱和/小模型足够，而不是模型扩大带来明显收益。
- 主图替换前须确定希望强调的问题，并对保留条件完成匹配重复及必要比较方法；不根据目标分数选择某个网络作为已验证的最优方案。

详见 `ALL_SCORES.csv`、`D_MATCHED_SUPPORT_SUMMARY.csv`、`E_COMMON_SUPPORT_SUMMARY.csv` 和 `MANIFEST.json`。
'''
(R/'RESULTS_ZH.md').write_text(report)
print('D',d_over.to_string(index=False));print('E',e.to_string(index=False));print('E per task',epivot.to_string())

selected=df[df.fit_budget.eq(5000)].sort_values(['source_validation_macro_f1','width'],ascending=[False,True]).drop_duplicates(keys)
selected.to_csv(R/'E_SOURCE_SELECTED_WIDTH_DIAGNOSTIC.csv',index=False)
chosen_mean=selected.groupby('task').macro_f1.mean().mean()
p=R/'RESULTS_ZH.md'
s=p.read_text()
conclusion=f"""
## 本轮判断

- 全部 170 个新增配置成功；没有改写既有正式实验。
- D：完整 34 方向从 2k 到 5k 仅 +0.0044；10k 可用的 28 方向从 2k 到 10k 为 +0.0090。20k 在 3 组/5 个方向中为 +0.0686；50k 只在一个 HNSCC 方向中为 +0.0641，且该方向 50k 低于 20k。整体收益没有随样本数普遍大幅提升。
- E：在 5k 训练细胞下，宽度 8/32/128/512 的均值为 0.5364/0.5338/0.5440/0.5374。加宽到 512 没有稳定改善。
- 另按源验证分数选择宽度（并列时选较小宽度，不使用目标结果），平均目标 F1={chosen_mean:.4f}。这一诊断也没有明显解决问题；它使用了扩展后的宽度选择网格，不能混称为原正式协议。
- 源验证均值约 0.78–0.79，目标平台约 0.53–0.54，结果更符合迁移/可用信息仍有限这一解释，而非仅网络容量不足；本实验没有区分具体成因。
- 推荐当前 E 移到补充材料；若希望保留一个主图效率 panel，应围绕相近预测性能所需的模型参数或监督量设计匹配比较，不再以增大网络必然改善为前提。单种子初筛不支持直接替换三种子主图或新的方法优越性结论。
"""
p.write_text(s+conclusion)
