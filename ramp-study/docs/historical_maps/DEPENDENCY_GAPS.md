# 尚需处理的依赖与版本问题

这是**研究代码整理清单**，不是新一轮算法性能/统计学审稿，也不是要求立即改分析。优先级表示对代码交付的影响。第二次审计后仅修订本盘点文档/检查工具并补输入哈希；下列研究代码迁移、上游恢复和端到端复现工作仍未执行。

## 优先处理：决定是否能交付可复现研究代码

### G01 — 发表 PDF 与本地 manuscript 源稿未精确对上

发表 PDF61页，本地 `bioarxiv_Ver2/main.pdf`64页；标题和文字不同。20张本地图稿资产都找到精确同源文件，但不能因此证明整个本地TeX就是发表源稿。

处理：找回提交时的Overleaf export/最终TeX；若暂时没有，保留发表PDF作为结果范围锚点，明确本地稿是候选源稿。不要为消除差异重写/覆盖原论文。

证据：`evidence/paper_identity.json`、`manuscript_identity_comparison.json`、`publication_assets.json`。

### G02 — Benchmark builder 不全部从 raw 开始

- TNBC builder读取 `test/tnbc_meyer5_coverage_guided_seed20260960/input`；Meyer的raw重建脚本又复制已存在benchmark_data中的另外两平台，存在bootstrap依赖。
- Melanoma builder读取test输入，但对应raw builder已找到，需要一并纳入。
- Lung builder读取test0827；该准备链引用旧story_lung_caf_bidirectional。当前冻结数据和selectedIDs存在，不代表已找齐原始抽样入口。

处理：逐项明确两个运行目标：冻结输入复现、provider数据从零重建。保留历史donor/cell ID，不重新抽样“凑成一样”。不能把20260960写成无筛选的随机draw。

### G03 — HNSCC正式目录仍依赖外部test结果

Panel E 的group_fraction_scan使用 `test/.../tumour_cd8_molecular_scores.csv.gz`、`story_test_hnscc/.../individual_marker_donor_contrasts.csv`。Panel F的`check_codex_phenotype.py`读取 `story_test_hnscc/final_return_exploration_20260910/target_all_*`。

处理：四个精确输入已在 `evidence/audit2_dependency_inputs.json` 的 `critical_upstream` 组冻结hash；时间从该补充快照开始，不能据此追溯证明第一次盘点前未变化。仍需追踪producer，再迁入正式研究模块；不要因带test字样删除。必须区分脚本main与import的helper，避免夸大依赖。

### G04 — Melanoma原始state-survival输入的producer仍待确认

`figure_melanoma/inputs/observed_patient_level_features.csv`、`observed_survival_results.csv`用于重画B，但本轮未找齐其从原始细胞/临床表合并生成的完整入口。现有manifest不能完全替代这一段代码。

处理：补上来源或明确这两个表为冻结analysis input及生成说明；无需重跑整套alignment来解决文档问题。

### G05 — Lung新release不独立包含其全部上游

`releases/senosain140k_20260921`继续使用`data_umap`、`data_cords`、`validation`的冻结矩阵和metadata，另外读取trainability目录中的candidateIDs。raw_data中的Senosain annotation脚本也是必需的研究代码。

处理：为这些输入单独建manifest和provider-to-frozen入口；保留旧目录中仍被调用的部分。新release名称不代表所有旧目录都可删。

进展：candidate-ID表已作为第五个 `critical_upstream` 输入补入 `evidence/audit2_dependency_inputs.json`。其他矩阵的上游重建仍是后续工作；此处补hash不等于补完生成流程。

### G06 — 只整理六方法和四story不够覆盖论文

Figure2还需要三application的comparison输出；Figure3有独立训练框架及scope updates；Figure1、2有values/gating/artifact程序；S1–S13和supplementary_data也需纳入。部分`exploration_*`是当前Figure3结果的实际producer。

处理：按图/任务映射筛选，不按文件夹名字“test/exploration/preview/old”机械过滤。

特别是四benchmark的MaxFuse当前输出叫 `maxfuse_archive`。`maxfuse/`、`maxfuse_v2/`不能按名字自动视为正式/旧版：当前三项Archive流程依赖v2的冻结pairwise triplets，TNBC则从新版62,701-cell输入重新拟合。参考 `benchmark_results/MAXFUSE_ARCHIVE_CURRENT_AUDIT.md` 及evaluation脚本第402行附近；旧TNBC `maxfuse/config.json`仍是65,211-cell历史配置。

## 其次处理：可移植性与入口安全

### G07 — 最终绘图依赖旧layout和中间资产

CD19 v31 import v30/更早helpers，HNSCC最终assembler读取旧layout JSON和中间panel PDF。Melanoma v8读取三件 `cache/ramp230_clean15_display_*` 文件及v7 logo/Panel H数据；cache生成函数在 `figure_melanoma/01_build_three_row_figure.py::prepare_alignment_display()`，不是v8的`prepare_visual_data.py`。Supplement S1–S7分别硬编码到七个benchmark-specific `ridge_full_range_20260922`目录，详见FIGURE_CODE_MAP中的逐项映射；它们不是同名的Main Figure目录。没有完整import/asset闭包前，不应只复制一个最终build脚本。

处理：先建立每张图的绘图输入清单，然后提取必要helpers/数据表/图像/字体。保持数值和布局不变。

### G08 — 一次性发布脚本不是可重复分析入口

`figure_raw_labels_20260924/update_labels.py`有备份存在断言；`deploy.py`、`finalize.py`、`install_manuscript.py`等会写回图稿/发布位置。部分绘图脚本在import时就创建目录或生成输出。

处理：不要为“检查能不能跑”直接执行/import这些文件。后续将pure rendering与安装/发布步骤分离，先在隔离输出路径测试。

### G09 — 环境证据分散，不能保证跨机直接运行

算法包、MaxFuse、R方法、story、绘图各有不同依赖记录和本地绝对路径；macOS字体、R库、provider RDS/H5AD、TIFF decoder也是依赖。当前未安装或验证完整study环境。

处理：每模块带运行manifest和最小environment；不要打包venv/Rlib；先做小输入/重画级检查，再决定是否重跑大型分析。源码census包含第三方deps/历史副本，不是发布白名单。

### G10 — 数据许可与发布内容需单独审查

本次只索引文件，不复制raw或临床表，也不确认允许再分发。Code Availability中的study scripts不能仅靠软件包兑现，但这不授权把所有patient-level数据上传。

处理：代码、参数、source accession、hash、selectedIDs和可公开衍生表分层；需用户批准的许可证/作者/数据发布事项留到发布阶段。

## 保留既有分析，不在整理时“修正”的项目

- Lung技术benchmark的pair-fit7000/0/7000：保留实际配置，附manifest和解释；不把它改成默认后冒称原结果。
- Meyer coverage-informed seed20260960：保留选择规则和selectedIDs；不替换为“更自然”的日期seed。
- Figure2主23指标、Figure3scope更新、representative ROI选择：保留选择过程，不能包装成全部预先设定。
- Figure6的synthetic DAPI背景：明确是合成示意，不误标为真实成像。
- HNSCC原niche survival与returned state validation：是两个分析，不合并叙述成新的独立生存发现。
- fastMNN corrected PCs与reconstructed values、MaxFuse integration与prediction配置：保留各自用途，不因统一目录而变成一套输出。

## 仍未穷尽的检查

没有逐一审查所有旧脚本、每个supplementary表、每条动态import、每个外部下载URL及每个数据文件hash。没有执行研究代码或回归测试。以上未找到的producer只能记作“当前追踪未闭合”，不能断言已不存在或结果无法重现。
