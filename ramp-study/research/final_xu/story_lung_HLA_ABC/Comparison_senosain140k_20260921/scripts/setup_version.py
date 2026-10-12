from pathlib import Path
import json,shutil,hashlib
C=Path(__file__).resolve().parents[1];S=C.parent;O=S/'Comparison';R=S/'releases/senosain140k_20260921'
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
assert not (C/'SETUP.json').exists()
for d in ['input','methods','evaluation','logs','.cache','tmp','maxfuse_full_panel/scripts','maxfuse_full_panel/data','maxfuse_full_panel/methods','maxfuse_full_panel/evaluation','maxfuse_full_panel/logs','maxfuse_full_panel/.cache','maxfuse_full_panel/tmp']:(C/d).mkdir(parents=True,exist_ok=True)
files=['prepare.py','run_r_methods.py','run_r.R','cycombine_repro.R','cytonorm_repro.R','evaluate.py','summarize.py','render_overview.py']
for name in files:shutil.copy2(O/'scripts'/name,C/'scripts'/name)
p=C/'scripts/prepare.py';s=p.read_text().replace("src=STORY/'data_umap'","src=STORY/'releases/senosain140k_20260921'").replace("src/'data/RAW_SHARED9.npy'","src/'inputs/RAW_SHARED9.npy'").replace("src/'data/CELL_METADATA.csv.gz'","src/'inputs/CELL_METADATA.csv.gz'").replace("src/'data/DESIGN.json'","src/'inputs/DESIGN.json'").replace("src/'reference/RAMP_SHARED9.npy'","src/'alignment/RAMP_SHARED9.npy'").replace("^Other$|uncertain|ambiguous|Non-joint|unclassified|Unknown","^Other$|uncertain|ambiguous|mixed|Non-joint|unclassified|Unknown")
s=s.replace('No extra active markers.','MaxFuse is fitted separately with the same full-native-active protocol as the preceding final comparison.')
p.write_text(s)
# Copy unchanged wrappers for MaxFuse; use the new frozen source cells.
for name in ['prepare.py','run_maxfuse.py','evaluate.py']:
 shutil.copy2(O/'maxfuse_full_panel/scripts'/name,C/'maxfuse_full_panel/scripts'/name)
p=C/'maxfuse_full_panel/scripts/prepare.py';s=p.read_text().replace("STORY/'data_umap/data/CELL_METADATA.csv.gz'","STORY/'releases/senosain140k_20260921/inputs/CELL_METADATA.csv.gz'").replace("src=ROOT/'story_test_lung/shared_inputs/senosain'","src=ROOT/'raw_data/lung_cancer/Senosain_2025_CyTOF_and_omics/annotation'").replace('senosain_raw35_markers.csv','senosain_140k_markers35.csv').replace('senosain_exact70k_raw35.npy','senosain_140k_raw35.npy').replace('senosain_exact70k_metadata.csv.gz','senosain_140k_cell_metadata.csv.gz');p.write_text(s)
# Cords reference IDs and values are unchanged. Reuse its independently exported full-panel counts.
shutil.copytree(O/'maxfuse_full_panel/data/cords_imc',C/'maxfuse_full_panel/data/cords_imc')
(C/'maxfuse_full_panel/input').symlink_to('../input',target_is_directory=True)
# Freeze metric policy before any new scores are evaluated.
policy=S.parent/'Main Figure/Figure 2/metric_selection_main23/metric_selection.json'
shutil.copy2(policy,C/'METRIC_POLICY.json')
save(C/'SETUP.json',{'original_adapters':{str(O/'scripts'/n):sha(O/'scripts'/n) for n in files},'original_full_panel_adapters':{str(O/'maxfuse_full_panel/scripts'/n):sha(O/'maxfuse_full_panel/scripts'/n) for n in ['prepare.py','run_maxfuse.py','evaluate.py']},'metric_policy_source':str(policy),'metric_policy_sha256':sha(policy),'new_release':str(R),'reason':'Same completed new lung-story reference, refit all comparator methods; no algorithm or metric tuning.'})
(C/'PLAN_ZH.md').write_text('''# Figure 2 肺癌 story 比较更新

冻结时间：2026-09-21，新分数计算前。

- 复用正式 `releases/senosain140k_20260921` 的 84,000 细胞、9 个共享 marker 和 RAMP 输出；Cords/Zheng 参考细胞未变。
- 重跑 Harmony-direct、fastMNN、cyCombine、CytoNorm 及 full-panel MaxFuse。所有方法 cell ID / 顺序完全一致；标签不进拟合。
- 原版本的 broad_type 评价规则不变：未明确标签不评分（新版 Other / mixed 对应原 Other / uncertain）；某次抽样中至少出现于两个平台、总数至少 30 的明确类别进入 joint 指标。其余混合指标保留全样本。无结果驱动筛类或合并。
- MaxFuse 各平台 full-active 蛋白；共享部分严格等于同批细胞的 9-marker 输入；沿用通道排除、变换、维度规则和三个 cyclic hub。每项每 seed 取三个 hub 的中位数。
- 每种方法 pooled StandardScaler，无额外 PCA；全细胞 UMAP n_neighbors=30、min_dist=0.15、seed=42。按旧 20 seeds，每平台抽 5,000 细胞，全部方法行哈希相同。
- 全 31 项保存；Figure 2 固定沿用 METRIC_POLICY.json 的 23 项与四类等权；其他六组复用旧分数，E/F gating 不变。
- 所有输出留在本目录。新 Figure 2 单独版本目录，完成核验后更新指针和 bioRxiv_Ver1。旧结果只读。
''')
print(C)
