# 发表论文的图—代码映射

所有简写路径相对于 `/Users/qxu6/Documents/New project/`。图号按 61 页的发表 PDF，不按旧 README 或旧 stage 文件名。以下均是静态追踪，没有执行生成程序。

## Figure 1 — 方法、形态及数值保真（PDF p.6）

**最终资产**：`final_xu/figure_raw_labels_20260924/output/pdf/figure_1_melanoma.pdf`，与本地稿 `figure1_ramp_overview.pdf` 字节一致。

| Panel | 发表内容 | 必须保留的主要代码/来源 |
|---|---|---|
| A | RAMP 示意图 | `final_xu/Main Figure/Panel A/` 的原始美术资产及后续 `icon_update_20260915/` 绘图/组装代码；示意图资产不是算法代码 |
| B | melanoma 三平台 UMAP，Raw/Harmony/cyCombine/RAMP；CD3、CD8、SOX10 ridge | `Main Figure/ridge_full_range_20260922/`、`ridge_range_preview_20260924/` 与最终修改记录；读取 melanoma benchmark 输出及现成 UMAP，不能用旧 lung 示范图替代 |
| C | BM H1 CD4、TNBC Shiao h17 CD8、BM H2 CD14 的 marker artifacts | `Main Figure/shared_cd4_h1_20260914/`、`fastmnn_cd14_positive_control_20260914/`、`fastmnn_reconstructed_audit_20260914/` 和组装脚本；早期 CD68 图不是此处全部例子 |
| D | 五个 value-output 方法的 celltype median percentile shift；6,173 个 unit×celltype×marker 项、103 个单位 | `Main Figure/celltype_percentile_shift_20260914/compute_celltype_shifts.py`；配套 `pair_inversion_boxplot_20260914/compute_pair_inversions.py` 是排序审计，不要混成同一个统计量 |

最后标签处理是 `figure_raw_labels_20260924/update_labels.py`，把旧 Raw 标签改短，保留绘图数据。这个脚本带 backup 不存在断言，是一次性历史发布步骤，不能原样重复运行。需要保留其输入资产和修改记录；后续再拆成可重复绘图入口。

注意：图上写 Raw 不意味着所有 assay 未经自身数据准备变换；方法输入尺度由各 benchmark 配置决定。fastMNN 在数值图中使用 reconstructed diagnostic，不是 integration 的 corrected-PC 坐标。MaxFuse 不提供对应的精确 marker value，因此不在这五个 value 方法中。

## Figure 2 — Integration 与固定阈值转移（PDF p.8）

**最终资产**：`final_xu/Main Figure/Figure 2/lung_reference_update_20260921/Combined/figure_2_lung_reference_update.pdf`。

**当前生成入口根目录**：`final_xu/Main Figure/Figure 2/lung_reference_update_20260921/`（不是其 `Combined/` 子目录）。该根目录包含 `build_selection.py`、`build_all.py`、`common.py`、`Panel A` 至 `Panel F/build_panel.py`、`verify_release.py`。不是旧 `metric_selection_main26/27/29` 或旧四块 UMAP 拼图。

| Panel | 内容 | 计算来源 |
|---|---|---|
| A | Louvain cell-type ARI | `benchmark_results/comparison_evaluation/scripts/evaluate_benchmark.py` + 三个 application `Comparison` 目录 |
| B | UMAP macro cell-type silhouette | 同上；数据空间和 UMAP 空间指标要区分 |
| C | 23 个主指标、四类评价 | 当前 `selection/` 的 main23 配置；31 指标全集保留为敏感性/完整输出 |
| D | 七项任务平均名次，RAMP≈2.01 | 当前七任务 rank 表；不是只选四个 benchmark 的均值 |
| E | BM AbSeq→CODEX，CD3/CD4/CD8 固定 gate 示范 | `final_xu/Main Figure/Figure 2/restored_means_metric_review_20260914/Panel E/prepare_data.py` 准备数据；当前作图为 `final_xu/Main Figure/Figure 2/lung_reference_update_20260921/Panel E/build_panel.py` |
| F | 四 benchmark 的 11 marker tasks、每项六个方向 | `final_xu/Main Figure/Figure 2/value_evaluation_20260914/evaluate_values.py`、`final_xu/Main Figure/Figure 2/value_evaluation_20260914/summarize_values.py`；fastMNN reconstructed 版本经 `final_xu/Main Figure/fastmnn_reconstructed_audit_20260914/evaluate_reconstructed.py` 汇入；当前作图为 `final_xu/Main Figure/Figure 2/lung_reference_update_20260921/Panel F/build_panel.py` |

E/F 的数据准备目录是 `lung_reference_update_20260921/` 的兄弟目录；不能替换成 `two_rows_three_markers_20260914/` 或 `compact_rank_boxplots_20260914/` 下的同名副本。

七任务＝四个技术 benchmark＋melanoma/HNSCC/lung HLA-ABC 三个 application alignment。方法＝RAMP、Harmony、cyCombine、CytoNorm、fastMNN、MaxFuse。20 次匹配 resampling，每平台每次 5,000 cells。主 23 指标子集的选择历史需如实保留，不能整理时改写成分析前预设。

## Figure 3 — 跨平台预测（PDF p.11）

**最终资产**：`final_xu/figure3_readability_preview_20260924/output/pdf/figure3_readability_preview.pdf`。名字虽有 preview，但与本地稿正式资产相同。

**数值权威指针**：`final_xu/cross_platform_trainability_v2_20260919/CURRENT_FIGURE3_RESULTS.json` → `releases/figure3_bc_update_20260922/`。根 README 开头旧 formal suite 不足以识别当前结果。

- A：训练/测试工作流示意。
- B：跨平台 cell-type annotation，七项 benchmark、34 个方向；三个 model families，10 种 preprocessing/alignment pipelines。
- C：缺失 marker 预测，六项 benchmark、79 个 benchmark–marker 组合；marker 必须在预处理和重拟合前被移除。原清单的方向总数没有明确来源，已删除，不与 B 的34个方向混用。

计算链位于 `src/`、`src/trainability/`、`config/formal_v0_5/`。当前 B 是原 suite 加 `releases/j1_scope_update_20260921_r3/` 指定的 scope 更新；其实际 producer 保留在多个 `exploration_*_20260921` 目录。C 包括 `exploration_lung_new140k_j2_20260922/` 的新版 lung。它们虽然叫 exploration，已成为当前结果依赖，不能统一排除。

出图链：`Figure/build_panels_bc.py` → `publication_cleanup_20260922/build_display_assets.py` 的 Figure 3 处理 → `figure3_readability_preview_20260924/draw_preview.py`。最后一项读的是 panel 数据表，不重新训练模型。

当前 paper 不含旧 J3/J4、旧 Figure 3 D–F。原三平台 melanoma 未纳入严格 J2，与两平台 melanoma story 不是一回事。各更新的标签范围及选择历史需随 manifest 发布。

## Figure 4 — CD19 completion（PDF p.12）

**最终资产**：`final_xu/story_cd19_prediction/Exploration0909/figure_layout_v31_DAPI_pair/cd19_story_three_row_v31_DAPI_pair.pdf`。

**实际大图入口**：同目录 `build_figure.py`，继续调用 `Exploration0909/scripts/assemble_cd19_layout_v30_polish.py`，后者仍使用更早 layout/helper 与资产。仅复制 v31 文件夹不能独立运行。

| Panel | 发表内容 | 对应链 |
|---|---|---|
| A | H14 同一 ROI 的真实 DAPI 与失败 CD19 | `Exploration0909/real_spatial0913/scripts/render_real_roi.py` 及冻结图像资产 |
| B | label-free QC、观察到的 B/T CD19 分布 | `prediction/`、`Exploration0909/` 的 QC 表和 panel 数据；B/T 只用于外部核对，不是训练标签 |
| C | completion 工作流 | `prediction/run_prediction_and_validation.py`、`masked_marker_model.py`；**KMEANSLOW → masked NN completion → completed-panel RAMP** |
| D | CD3 永久移除后预测检查与 13-marker joint UMAP | `Exploration0909/scripts/cd3_nn_joint_validation.py` |
| E | 14 个永久 held-out marker×12 CODEX donors | `Exploration0909/scripts/all_marker_nn_validation.py`；不是旧临时 mask 后沿用原 alignment 的 validation |
| F | observed/predicted CD19 ridge | `Exploration0909` 的 validation/completion 输出与 v30/v31 绘图依赖 |
| G | H14 全 atlas 投影后同 ROI 的 B cell/预测 CD19 | `real_spatial0913/scripts/project_h14.py` → `select_roi.py`/render 脚本；与 panel A 同一图像范围 |
| H | 14 markers vs 14+predicted CD19 joint UMAP | `alignment/run_alignment.py`、`umap/plot_umap.py`，并沿用最终 layout 中的 panel 资产 |

冻结输入为 88,031 cells：CODEX 12 donors/30,000，CyTOF 2 donors/30,000，AbSeq 3 donors/28,031。与四 benchmark 中 BM 的冻结抽样相关，但训练/完成/重对齐步骤必须单列。原始失败 CD19 不参与预测特征。

核对锚点：held-out marker donor-mean 的中位数 AUROC=.772、AP=.567、Spearman=.516、Pearson=.522；H14 全 atlas 83,384 cells；ROI 251 cells、43 B；B/NK UMAP silhouette .056→.724。本轮只用这些识别版本，不重算数值。

## Figure 5 — Melanoma spatial CD8 state（PDF p.14）

**最终资产**：`final_xu/figure_melanoma/figure_layout_v8/melanoma_three_row_redesign_v8.png`。

**入口**：本节未展开的路径均相对于 `final_xu/figure_melanoma/`。`figure_layout_v8/prepare_visual_data.py` 从已有 held-out predictions 重算 Panel F 展示指标；`figure_layout_v8/build_figure.py` 组装最终图。前者**不生成 alignment UMAP cache**。`analysis/run_all.sh` 只做到原始组图，不能代替 v8 末端绘图。

v8 在顶层实际读取三件 cache：`cache/ramp230_clean15_display_umap.npy`、`cache/ramp230_clean15_display_metadata.csv.gz`、`cache/ramp230_clean15_display_manifest.json`，并核对 manifest 中的 alignment/metadata/display-label 哈希。可定位的 cache 生成函数是 `01_build_three_row_figure.py::prepare_alignment_display()`（约第474–526行）。已检查的旧 layout v2–v5 同样**读取**这份 cache；不能把它们误写成已确认的 cache writer。

此外，v8直接读取 `figure_layout_v7/assets/ramp_sign_cross_platform_alignment.png`，以及 `figure_layout_v7/data/panel_H_group_means_by_patient.csv`。不能只迁移v8目录。四件 cache/logo 资产已在第二次审计补充索引 `evidence/audit2_dependency_inputs.json` 中单独冻结哈希；这不代表全部绘图依赖已经闭合。

| Panel | 发表内容 | 主要代码 |
|---|---|---|
| A | Ki67-positive、距 tumor≤30 μm 的 spatial CD8 定义 | `01_build_three_row_figure.py` + 冻结定义示意图 |
| B | 原始空间状态与 OS：30 patients | `inputs/observed_patient_level_features.csv`、`observed_survival_results.csv`；当前作图读冻结表，上游原始生存特征构建入口尚需补全追踪 |
| C | prediction 工作流 | `analysis/all_cd8_prediction/run_all_cd8_patient_heldout.py` |
| D | 15-marker 全量 alignment 的两张 UMAP | `analysis/alignment/run_alignment.py` → `01_build_three_row_figure.py::prepare_alignment_display()` → `cache/ramp230_clean15_display_*` → v8绘图；展示子集不等于用于 fitting 的全量数据 |
| E | 22RD representative spatial ROI，ground truth vs prediction | `scan_visual_rois.py`、`make_all_cd8_spatial_figure.py` |
| F | 14 patients 的 held-out AUROC、top-20% enrichment | `run_all_cd8_patient_heldout.py`；当前中位数 .707、**1.98×** |
| G | CITE-seq CD8 subtype 与 projected score UMAP | `build_all_cd8_citeseq_projection.py`、`plot_all_cd8_citeseq_umap.py` |
| H | CD11a/CD18/CD2/PD1 与 projected state 的关系 | `analyze_all_cd8_biology.py` |

上述未写全路径的 prediction 脚本均在 `analysis/all_cd8_prediction/`。alignment 输入 112,497 IMC＋53,189 CITE-seq，共 165,686 cells；总 UMAP 每平台展示 20,000。目标 CITE-seq CD8=5,768；源 CD8=5,649。patient-held-out 指分类器训练/标准化，RAMP alignment 已事先冻结，不能称为整套 pipeline 每次完整留出患者。

## Figure 6 — Lung HLA-ABC / vCAF / DFS（PDF p.16）

**最终资产**：`final_xu/story_lung_HLA_ABC/Figure/Combined ABCDEF/panels_ABCDEF_preview.png`。

**分析入口**：`story_lung_HLA_ABC/CURRENT_ANALYSIS.json` → `releases/senosain140k_20260921/run.sh`。运行链为 align → project(validation/cords) → analyze → survival.R → mechanism → UMAP → verify/report。

**绘图入口**：`Figure/Panel A` 至 `Panel F/scripts/draw_panel_X.py`，最后 `Figure/Combined ABCDEF/scripts/assemble_ABCDEF.py`。

| Panel | 内容 | 输入/关键限制 |
|---|---|---|
| A | 三平台 reference 与 transfer 示意 | Senosain CyTOF、Cords IMC、Zheng mIF；第三平台不是技术 benchmark 的 Monkman |
| B | 三平台 alignment | 28 units/platform×1,000=84,000 cells；9 shared markers，HLA-ABC/HLA-A 不用于 joint alignment |
| C | CAF/vCAF 与 source HLA-ABC | 5,206 CAF，646 CD146+CD34− vCAF |
| D | Zheng 的 held-out HLA-A 空间/患者 validation | 693,693 cells、119 cores、28 patients；图像是坐标衍生的 **synthetic cells/DAPI**，不是 provider acquired fluorescence |
| E | Cords patient-level predicted vCAF score 与 immune composition | 全量 2,849,385 cells /527 patients 投影，进入 stage I–II clinical cohort 的 280 patients |
| F | DFS | 280 patients、96 events；连续 score adjusted HR=1.36，P=.0104 |

新 release 仍读取旧 `data_umap/`、`data_cords/`、`validation/` 的冻结输入，并读取 trainability 目录中的 Senosain candidate-ID 表。它们是上游依赖，不能因 release 新而删除。旧 HR=1.43、477 人 OS、旧 `story_test_lung` two-platform 分析不应替换本图。

## Figure 7 — HNSCC niche-guided CD39/TIGIT phenotype（PDF p.18）

**最终资产**：`final_xu/figure_HNSCC/big_figure/polish_20260913/CEF_D_colorbar_aligned_20260914/hnscc_CEF_D_colorbar_aligned.pdf`。

**大图入口**：同目录 `build_figure.py`。依赖旧 layout 的 `LAYOUT_AUDIT.json`、中间 panel PDF、donor values 表、ROI 图；不是仅有单文件绘图。

| Panel | 内容 | 代码与数据链 |
|---|---|---|
| A | niche→CITE-seq→CD39/TIGIT→spatial 工作流 | `big_figure` 的 schematic/panel assets |
| B | CODEX CAF/macrophage niche score 与 OS | `survival_motivation/01_build_spatial_niche_scores.py` → `02_run_survival_and_plot.R` → `03_validate_outputs.py` |
| C | 全细胞 alignment UMAP | `alignment/01_prepare_alignment_input.py` → `02_run_ramp_ver2.py` → `03_plot_umap.py`，94,488 cells、22 markers |
| D | CITE-seq 独有 protein 的 niche-score association、三 donor consistency | `biomarker_discovery/01_project_spatial_state.py` → `02_screen_biomarkers.py` → `03_validate_discovery.py`；107 proteins，四 expression views，选 CD39/TIGIT |
| E | high/low state 的独立 protein/RNA characterization | `big_figure/panel_E/group_fraction_scan_20260910/scan_group_fractions.py`，bottom/top 30%；仍读 test 与 story_test_hnscc 的上游表 |
| F | returned score 对 held-out spatial LAG3 的全患者及代表 ROI validation | `spatial_return_survival/01_return_fixed_program.py` → `spatial_score_review_20260913/check_codex_phenotype.py` → `roi_selection_20260913/` → `selected_spatial_panel/` |

B 是 74 patients/24 deaths 的**原 niche** OS，HR≈1.34/SD，P=.029。F 是 74 patients 的 score–LAG3 validation，mean within-patient Spearman=.522；不能把 B 写成返回双 marker state 的独立生存关联。E 用 4,788 tumour CD8 中的 1,439 low、1,437 high，中间 1,912 不用于 high/low 对比。

## Supplementary Figures S1–S13

S1–S7 最终资产均位于 `final_xu/Supplementary Figures/ridge_compact_preview_20260924/Sn/output/pdf/`，每项自己的 `build_figure.py` 配合公共 `ridge_common.py`。各脚本第6行的 `HERE` **硬编码为不同的绝对路径**，第24行附近从对应目录读取 `RIDGE_CONFIG.json`；并非共享一个上游目录。映射如下（下表完整路径从项目根开始）：

| 附图 | `HERE` 指向的上游配置目录 |
|---|---|
| S1 | `final_xu/Supplementary Figures/Bone marrow/ridge_full_range_20260922/` |
| S2 | `final_xu/Supplementary Figures/TNBC/ridge_full_range_20260922/` |
| S3 | `final_xu/Supplementary Figures/Melanoma three-platform/ridge_full_range_20260922/` |
| S4 | `final_xu/Supplementary Figures/Lung cancer benchmark/ridge_full_range_20260922/` |
| S5 | `final_xu/Supplementary Figures/Lung HLA-ABC story/ridge_full_range_20260922/` |
| S6 | `final_xu/Supplementary Figures/Melanoma story/ridge_full_range_20260922/` |
| S7 | `final_xu/Supplementary Figures/HNSCC story/ridge_full_range_20260922/` |

这七处都不同于 Figure 1 使用的 `final_xu/Main Figure/ridge_full_range_20260922/`。显示窗口压缩不等于重算/截断 density 输入。

| 附图 | 内容 | 最终文件名 / producer |
|---|---|---|
| S1 | Bone marrow integration、ridge | `bone_marrow_supplement.pdf` / `S1/build_figure.py` |
| S2 | TNBC integration、ridge | `tnbc_benchmark_supplement.pdf` / `S2/build_figure.py` |
| S3 | 三平台 melanoma | `melanoma_three_platform_supplement.pdf` / `S3/build_figure.py` |
| S4 | Lung technical benchmark | `lung_cancer_benchmark_supplement.pdf` / `S4/build_figure.py` |
| S5 | Lung HLA-ABC application | `lung_hla_abc_story_supplement.pdf` / `S5/build_figure.py` |
| S6 | 两平台 melanoma application | `melanoma_story_supplement.pdf` / `S6/build_figure.py` |
| S7 | HNSCC application | `hnscc_story_supplement.pdf` / `S7/build_figure.py` |
| S8 | integration structure/transfer/UMAP 指标明细 | `Figure 2 complete benchmark/preview_v1/build_figures.py` |
| S9 | platform/donor mixing 指标明细 | 同 S8 |
| S10 | 四个技术 benchmark 的分类，按 model 分解 | `Figure 3 model benchmark breakdown/final_20260922/paired_annotation.py` |
| S11 | 三个 application benchmark 的分类 | 同 S10 |
| S12 | missing-marker Spearman 明细 | `Figure 3 model benchmark breakdown/final_20260922/missing_marker.py` |
| S13 | missing-marker MAE 明细 | 同 S12 |

S8–S13 表中 producer 路径相对于 `final_xu/Supplementary Figures/`。最终经过 `final_xu/publication_cleanup_20260922/build_display_assets.py` 的 Harmony 显示标签处理，资产在 `Supplementary Figures/harmony_display_20260922/pdf/`。这些发布脚本读历史 manuscript manifest；需区分数值重算、重画、一次性发布三个层次。

## Supplementary data

`bioarxiv_Ver2/supplementary_data/README.md` 及其目录是图表背后的随稿数据索引，不应仅发布图像而遗漏。其 file list 已单独索引；本轮没有把每个表的每一列都回溯到 producer，也没有复制 patient-level 数据。后续根据上述各模块逐项补表级 lineage，并核对允许公开的内容。
