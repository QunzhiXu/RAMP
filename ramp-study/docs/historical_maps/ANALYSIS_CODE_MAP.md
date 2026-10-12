# 分析模块、输入与运行链

简写路径均相对于 `/Users/qxu6/Documents/New project/`。这里列的是应整理的模块，不是可直接执行的一键命令；本次未运行分析。

## 1. 软件核心与研究代码分开交付

- 面向用户的软件包：`ramp-proteomics/`，README、Python API/CLI、tests、`reproducibility/requirements-paper.txt`、`PROVENANCE.md`、`docs/reproducibility.md`。
- 研究中实际使用的核心：`final_xu/algorithm/ramp/ramp_ver2.py`、`ramp.py`。当前核心版本 2.3.0，不应整理时替换为旧 `ramp_ver1` 或直接改 imports 后声称结果一样。
- Lung 新 release 有自己的 vendor 副本；其来源应随运行 manifest/hash 一起保留。
- 本次源码 census/curated index 保存哈希，但没有重新运行软件包的 113 项测试。用户给出的独立软件审计结论不自动覆盖所有研究脚本。
- License、作者确认、发布状态属于交付管理；本次没有查询或修改远端仓库，也没有代为批准许可证。

## 2. 四个技术 benchmark 的数据与 relabel

共同根：`final_xu/benchmark_data/`。每项先 `build_benchmark.py`，再 `relabel/build_relabel.py`；必须同时带 selected cell IDs、donor/unit metadata、marker manifest、provider-label mapping 和抽样说明。不是只发一个 numpy matrix。

| 数据 | 冻结规模 | shared markers | selection/seed | 当前构建性质 |
|---|---:|---:|---|---|
| Bone marrow | 88,031 cells；17 donor units | 14；CD19 单独保留为第15个 | CODEX 全12 donor×2,500；CyTOF H1/H2×15,000；AbSeq BM1/BM2/BM3 全部28,031；seed20260820 | 直接读取 provider 原始/导出数据 |
| TNBC | 62,701；11 donor units | 11 | Tietscher BB165 20,000；Meyer 5 patients全22,701；Shiao 5×4,000 | 从 test 中已冻结三平台 input 重新包装 |
| Melanoma | 60,000；46 patient units | 12 | Hickey6、Hoch17、Ptacek23；每平台20k，seed20260827 | 从 test 中已冻结 input 重新包装；raw builder 已找到 |
| Lung | 60,000；30 units | 8 | 每平台10×2,000，Cords TMA分层；donor seeds20260828/29/30；cells20260838/39/40 | 从 test0827 已冻结 input 包装；更早 raw sampler 链尚未完全恢复 |

总计 104 个 dataset-specific sample units，不能不加解释地都称为独立患者；Monkman 标识是 patient core。BM 是健康 bone marrow，不应简单把四项都叫 cancer cohorts。

### BM

主要入口：`benchmark_data/bone_marrow/build_benchmark.py`、`export_codex_30k.R`。

直接输入：

- `raw_data/bone_marrow/BM_Codex/Normal_Bone_Marrow_CODEX_Atlas_Seurat_v2.rds`。
- `raw_data/bone_marrow/BM_CyTOF/levine_CyTOF_Annotated_Celltypes_fcs/` 及 sample/channel 文件。
- `raw_data/bone_marrow/Triana_2021/exported/` 的 ADT matrix、features/barcodes、metadata。

2026-10-10 迁移核对补正：CODEX 使用 R `RNGversion("4.0.0")`，按排序 donor 的 1-based 序号 i 设置 `set.seed(20260820 + 1009*i)`，每组不放回抽 2,500。CyTOF 才使用 NumPy `default_rng(20260821 + 104729*(sorted_group_index+1))`，每组不放回抽 15,000。此前把两者概括为同一个 NumPy 规则不准确；没有改变实际代码或细胞选择。固定 group order、输入行序和 ID 类型也是复现条件。AbSeq 不再抽样。RDS 导出需要 R/SeuratObject/Matrix，Python 需要 numpy/pandas/scipy；FCS 读取器已包含在 builder 中。

### TNBC

`build_benchmark.py` 的默认来源：`final_xu/test/tnbc_meyer5_coverage_guided_seed20260960/input`。

Meyer 上游：`final_xu/test/tnbc_meyer5_resample_seed20260902/` 内 `scripts/build_test_input.py`、`screen_coverage_seeds.py`、`export_meyer.R`。读 `raw_data/tnbc/Meyer_2025/processed_r_objects/sce_ALL.rds`。Meyer 是扫描 20260903–20261002 共100个候选 seed 后的 **coverage-informed selection**：总 cell 数18–27k约束下，按 B/endothelial coverage 规则选择 20260960，保留五 donor 全部 cells。不得在发布文档简写为未经选择的随机抽取。

Shiao：full panel＋provider treatment code1 的十 donor 池；donor seed20260827、cell seed20260828；selected h05/h09/h12/h17/h30。Tietscher 仅 strict-TNBC BB165。

目前 Meyer builder 复制已存在 benchmark_data 中的 Tietscher/Shiao block，存在 bootstrap 问题：若把 benchmark_data 当成可从零构建的唯一入口，会反过来需要它本身。要么保留明确许可的 frozen input，要么把 Tietscher/Shiao 原始冻结流程独立出来；本次不改。

### Melanoma benchmark（非 Figure 5）

上游 `final_xu/test/melanoma_metadata_matched_20k_20260827/01_build_inputs.py` 直接读取 raw provider 文件。Hickey pretreatment、Hoch stageIII untreated、Ptacek stageIII；在 donor/image/FOV 层容量平衡，不是全 cell pool 均匀抽20k。SOX10/S100 proxy mapping 必须保留。内部 Ptacek_2021 名称与后来发表题录不同，需用 provenance 识别，不要按文件名另换数据。

### Lung benchmark（非 Figure 6）

当前 `benchmark_data/lung_cancer/build_benchmark.py` 默认读取 `test0827/lung/alignment_10_donors/`，provider Cords labels 从本地 `source_inputs/` 读取。

`test0827` 的 preparation 进一步引用旧 `story_lung_caf_bidirectional`。目前这条链没有形成已验证的独立 raw rebuild 入口；冻结 selected IDs、raw matrix、labels 仍在，不等于数据丢失，也不等于原始抽样流程已完整整理。

Senosain project annotation 相关源码位于 **raw_data 内部**：
`raw_data/lung_cancer/Senosain_2025_CyTOF_and_omics/annotation/01_sample_140k.py`、`02_kmeanslow_pca_umap.py`、`03_leiden_cluster_profiles.py`、`rough_annotation/build_final_annotations.py` 等。这些是研究代码，应提取纳入代码清单；不是复制整个 raw_data。Benchmark 中 C06/S01 的358 cells 标为 Mix，不能遗漏这个锁定 relabel。

## 3. 六种 comparison method 与四 benchmark runners

根：`final_xu/algorithm/`。方法适配代码和 benchmark 预处理配置分开，不能整理时把所有方法强行统一成同一个 arcsinh 或 KMEANSLOW 输入。

| 方法 | 主要入口 | 环境/版本证据 | 重要 contract |
|---|---|---|---|
| RAMP | `ramp/ramp_ver2.py`、`ramp.py` | requirements、运行 manifest、源码 hash | raw/source input按benchmark定义；内部自身预处理；不使用label拟合 |
| MaxFuse | `maxfuse/run_maxfuse_pairwise.py`、`run_maxfuse_threeway.py`、`run_maxfuse_three_view_extension.py`、`maxfuse_repro/` | upstream_lock、requirements-lock、bootstrap_venv；upstream commit `2900b260b697deec678060ea653015291c4dc5cb` | 三平台 extension 必须准确命名；full-active-panel integration 与 shared-only prediction 实验不是同一个配置 |
| cyCombine | `cycombine/run_cycombine.R`、`cycombine_repro.R` | upstream_lock、bootstrap_library、vendor publication-renv.lock；0.2.5 | rank norm/SOM/ComBat；保留实际输入预处理，不注入 RAMP init |
| Harmony | `harmony/run_harmony_direct.R`、`harmony_direct_repro.R` | upstream/runtime locks、bootstrap；1.2.3 | direct feature-space版本；source_scale主分析与fixed-arcsinh敏感性分开 |
| fastMNN | `fastmnn/run_fastmnn.R`、`fastmnn_repro.R` | batchelor1.18.1、runtime/upstream locks | corrected PCs用于integration；reconstructed markers仅数值诊断，两类输出不可互换 |
| CytoNorm | `CytoNorm/run_cytonorm.R`、`cytonorm_repro.R`、`benchmark.py` | upstream_lock、bootstrap.py/R；CytoNorm2.0.6/FlowSOM2.10.0 | proxy controls和门控规则需明确记录，不能写成具有真实跨批次共享control |

`algorithm/cytofruv/` 及 applicability summary 也存在，但不是论文六方法主比较。保留排除/适用性说明，不能把未适用项当成数值零分。KMEANSLOW 是 RAMP 内部步骤，不另列为 comparison algorithm。

四项结果入口位于 `final_xu/benchmark_results/<case>/`：

- `ramp/alignment/run_alignment.py` → `ramp/umap/plot_umap.py`，对应 manifests/输入与 alignment 输出。
- 当前 MaxFuse 结果是 **`maxfuse_archive/`**，不是同级旧 `maxfuse/`。TNBC经 `maxfuse_archive/build_inputs.py` → 根 `_maxfuse_benchmark.py`，使用该目录config；其余三项经 `_maxfuse_archive_replay.py` 重用锁定的 `maxfuse_v2` pairwise triplets，再做 Archive iterative GCCA。最终metrics直接读取 `maxfuse_archive/hub_*/alignment/data/maxfuse_embedding_float64.npy`。旧 `maxfuse/run_alignment.py` 只是历史入口，不能作为当前论文的默认运行入口。
- `cycombine/preprocessing/` → `alignment/run_alignment.R`；各 case Python/R 入口不同。
- `harmony/run_all.py` → 根 `harmony_benchmark_runner.py`。
- `fastmnn/run_all.py` → 根 `fastmnn_benchmark_runner.py`。
- `cytonorm_benchmark_runner.py` 驱动 case 的 CytoNorm 输出。

**保留 lung 的既有参数例外**：`benchmark_results/lung_cancer/ramp/alignment/run_alignment.py` 使用 pair-fit `(minimum, per-unit, maximum)=(7000,0,7000)`；其余三项为 `(4000,100,8000)`。这是既有配置，当前不更改。其 manifest/coupling audit 和说明应随代码发布；可复现整理不是重新调参。另有 BM with_cd19_15markers 对照，不替代主14-marker benchmark。

MaxFuse具体应保留 `MAXFUSE_ARCHIVE_CURRENT_AUDIT.md` 与各hub manifest。其余三benchmark的当前Archive replay不是从零重做pairwise，因此冻结triplets和其生成代码同样属于上游依赖。TNBC当前采用provider全active panels（40/36/39 markers），asinh(x/1)后dataset内sample-SD z-score，shared11为对应切片；不是旧 `maxfuse/config.json` 中65,211 cells、shared-only的历史设置。不要将历史配置误认为当前TNBC数据不一致。

## 4. 评估、artifacts 与 gating

### Integration metrics

`benchmark_results/comparison_evaluation/scripts/evaluate_benchmark.py`、`summarize_all.py` 计算/汇总31指标。三 application 的 method preparation/evaluation 分别在：

- `figure_melanoma/Comparison/`、`Comparison/maxfuse_full_panel/`（实际子路径见 curated index）。
- `figure_HNSCC/Comparison/` 与其 full-panel MaxFuse 目录。
- `story_lung_HLA_ABC/Comparison_senosain140k_20260921/` 与其 full-panel MaxFuse 目录。

必须携带这些 application comparator runners，否则只有四 benchmark 不足以重建 Figure2的七任务评估。

20 fixed resampling seeds：
`20220524, 20230817, 20200502, 20230801, 20190826, 20220525, 20230818, 20200503, 20230829, 20190827, 20137910, 20349283, 24918503, 38298493, 150, 2940, 10, 302934245, 3428, 9123784`。

每平台5,000；六方法共享相同 cells；具体embedding与UMAP参数由各eval/config决定，不是只比较任意现成PNG。

### Values / artifacts

早期发现代码：`benchmark_results/exact_value_artifacts/scripts/run_exact_value_audit.py`、`plot_representative_artifacts.py`，其中仍有 `test0827/lineage_marker_artifacts` 引用。

当前论文图/统计以 `Main Figure/celltype_percentile_shift_20260914/`、`pair_inversion_boxplot_20260914/`、`fastmnn_reconstructed_audit_20260914/` 和最终选中示例脚本为准。每数据集 pooled celltype mean 的旧 CD68 检索不等于当前 unit-level audit。

Gating入口在 `Main Figure/Figure 2/value_evaluation_20260914/`，再由 fastMNN reconstruction audit 加入 diagnostic 分支；不要把MaxFuse latent输出当成原marker值去做固定阈值转移。

## 5. Trainability：分类与缺失 marker prediction

根 `final_xu/cross_platform_trainability_v2_20260919/`。需要打包的不止 Figure 绘图脚本，还包括：

- `src/prepare_real_inputs.py` → `build_formal_suite.py`/`release_formal_suite.py` → `run_formal_suite.py`/`run_experiment.py` → `summarize_formal_suite.py`。
- `src/trainability/` 内 inputs、adapters（含RAMP/MaxFuse等）、sampling、heads、targets、scoring、aggregation等。
- `config/formal_v0_5/SUITE.json`、`PROFILE.json`、输入 SUMMARY/校准 cells/held-out target 清单。
- 当前 J1 release 加六个 scope update 的 `exploration_*` producer；J2 的 `exploration_lung_new140k_j2_20260922/`。
- `releases/figure3_bc_update_20260922/`、`releases/j1_scope_update_20260921_r3/` 的 manifests、primary_means、REPORT 和 jobs list。

10 pipelines＝六主方法＋unaligned assay-ready、clipped-z、minmax、percentile。三个模型族＝linear、kNN、MLP。主要 RNG seeds20260919/20/21；分类 source train500、target evaluation512；source calibration64/unit。当前 J1共3,060 primary jobs；当前 J2汇总29,310 primary jobs。原 suite更多任务数不能直接写成当前 Figure3样本数。

J2 缺失marker要在预处理/重对齐前移除。跨平台方法只在允许的calibration信息上拟合的限制由adapter/runtime manifest体现；整理不能为省事复用完整marker alignment。

## 6. 四个生物学应用的可执行链

具体图层映射见 FIGURE_CODE_MAP；以下列输入与依赖问题。

### CD19

`story_cd19_prediction/sampled_data/` 冻结BM输入 → `prediction/run_prediction_and_validation.py`/`masked_marker_model.py` → `alignment/run_alignment.py` → `umap/plot_umap.py`。

正式marker holdout的独立入口在 `Exploration0909/scripts/cd3_nn_joint_validation.py`、`all_marker_nn_validation.py`。真实H14图像/ROI投影在 `Exploration0909/real_spatial0913/scripts/`。最终v31通过旧v30/更早helpers组图，不能只迁移名字最新的目录。

### Melanoma

`figure_melanoma/inputs/cohort_package/build_story_sample.py` → `analysis/alignment/run_alignment.py` → `analysis/all_cd8_prediction/` 的patient-held-out model、CITE projection、biology → `figure_layout_v8/`。

展示alignment UMAP还有独立分支：`figure_melanoma/01_build_three_row_figure.py::prepare_alignment_display()` 生成 `cache/ramp230_clean15_display_umap.npy`、`ramp230_clean15_display_metadata.csv.gz`、`ramp230_clean15_display_manifest.json`，由v8直接读取。`figure_layout_v8/prepare_visual_data.py`只处理已有held-out预测对应的Panel F指标，不生成该cache。v8还读取v7的`assets/ramp_sign_cross_platform_alignment.png`及`data/panel_H_group_means_by_patient.csv`。旧layout v2–v5中的已检查入口是cache读取者，不是已确认的cache生成者。

全量alignment15markers；分类器balanced logistic regression，C=.5、liblinear、max_iter5000，seed20260829。Source definition：provider CD8、Ki67 probability≥.5、tumor centroid距离≤30μm。UMAP展示sampling与alignmentfit分开。

`inputs/observed_patient_level_features.csv`、`observed_survival_results.csv` 有冻结表，但本轮尚未定位其完整原始特征/临床合并 producer。这是需要补的 lineage，不否认现有表可用于重画。

### Lung HLA-ABC

当前release `story_lung_HLA_ABC/releases/senosain140k_20260921/` 包含 runtime、vendor、align/project/analyze/survival/mechanism/umaps/verify/report。

上游三层均要保留：

1. `data_umap/scripts/` 的 raw来源、84k reference/selected IDs、baseline/reference数据。
2. `data_cords/scripts/` 生成全Cords数据及clinical join；`validation/scripts/` 生成全Zheng数据及验证输入。
3. `raw_data/.../Senosain.../annotation/` 的140k抽样/annotation，及 trainability 的 `reports/senosain_new_annotation_review_20260921/CANDIDATE_REFERENCE_IDS_REVIEW_ONLY.csv.gz`。

release/run.sh不重建以上所有冻结输入，也不做最终Figure/Combined。必须把“冻结输入重算release”和“从provider重建全流程”作为两个明确入口，不能混称一键复现。

### HNSCC

`figure_HNSCC/sampled_data/` 完整 Haist/Rahim输入 → `survival_motivation/run_all.sh`、`alignment/run_all.sh` → `biomarker_discovery/` → `spatial_return_survival/01_return_fixed_program.py` → molecular/spatial validation → big figure。

Alignment每个74CODEXpatient抽1,000＋全部20,488Rahim cells，sampling seed20260823；不是早期41LNmet患者筛选。State projection用full PT CD8，22joint markers，k30，先各patient预测再取median；query tumourCD8=4,788。

E仍读 `test/hnscc_tigit_cd39_molecular_spatial_refinement_20260904/citeseq_state_validation/results/tumour_cd8_molecular_scores.csv.gz` 和 `story_test_hnscc/ending_exploration_20260910/individual_marker_donor_contrasts.csv`。

F的check脚本读 `story_test_hnscc/final_return_exploration_20260910/target_all_metadata.csv.gz`、`target_all_aligned.npy`；必须追入这些输入的producer。`check_codex_phenotype.py` import `analyze_spatial_scores` 的helpers不等于执行后者main，所以不能把该helper所有main输入都误列为当前F必需。

## 7. 环境、输入数据与发布边界

- RAMP包有固定依赖和测试；研究环境仍分Python/R、MaxFuse独立venv、各R方法包库。保留requirements/upstream/runtime locks、真实run manifests，不能仅复制本机 `.venv` 或 `.Rlib`。
- Figure脚本依赖matplotlib、numpy/pandas、PyMuPDF/PIL、部分TIFF工具、字体等；部分调用macOS Arial路径。完整统一绘图lock尚未验证。
- 用路径配置指向provider原始数据，保留下载accession/文件名/hash/许可说明；不复制大规模raw data进Git。
- 生成的numpy矩阵、临床表、图像cache、selectedID表是否可发布要分开检查。Patient IDs即便为provider编码，也不能默认全部公开。
- 附图配置、绘图数据表、原始图像crop/source、schematic/logo资产也是重画依赖；不能只发`.py/.R`后声称所有图可复现。
- 两个层次分别验收：先已许可冻结输入→数字/图，再provider输入→冻结输入；目前本清单只作定位，不宣称任何模块已端到端重跑通过。
