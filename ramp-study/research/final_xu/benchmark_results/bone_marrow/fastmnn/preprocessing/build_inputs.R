#!/usr/bin/env Rscript

# Build the bone-marrow fastMNN input directly from frozen benchmark identities
# and original provider sources. No other comparison-method output is read.

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_argument)) stop("Cannot determine script location.", call. = FALSE)
script_path <- sub("^--file=", "", script_argument[[1L]])
script_path <- gsub("~+~", " ", script_path, fixed = TRUE)
script_path <- normalizePath(script_path, mustWork = TRUE)
preprocessing_dir <- dirname(script_path)
method_dir <- dirname(preprocessing_dir)
alignment_dir <- file.path(method_dir, "alignment")
final_xu <- normalizePath(
  file.path(preprocessing_dir, "..", "..", "..", ".."), mustWork = TRUE
)
benchmark_dir <- file.path(final_xu, "benchmark_data", "bone_marrow")
raw_dir <- file.path(final_xu, "raw_data", "bone_marrow")
algorithm_library <- file.path(final_xu, "algorithm", "fastmnn", ".Rlib")
.libPaths(c(algorithm_library, .libPaths()))

suppressPackageStartupMessages({
  library(Matrix)
  library(SeuratObject)
  library(digest)
  library(jsonlite)
  library(readr)
})

markers <- c(
  "CD117", "CD11B", "CD11C", "CD123", "CD14", "CD3", "CD33",
  "CD34", "CD38", "CD4", "CD45", "CD45RA", "CD61", "CD8"
)
id_columns <- c("cell_id", "dataset", "donor", "unit_key")
dataset_order <- c("BM CODEX", "BM CyTOF", "Triana AbSeq")
expected_counts <- c(
  "BM CODEX" = 30000L,
  "BM CyTOF" = 30000L,
  "Triana AbSeq" = 28031L
)
expected_total <- 88031L
marker_file <- file.path(preprocessing_dir, "markers.txt")
harmonization_file <- file.path(
  preprocessing_dir, "marker_harmonization_manifest.csv"
)
selection_file <- file.path(benchmark_dir, "selected_cells.csv.gz")
data_dir <- file.path(preprocessing_dir, "data")
manifest_path <- file.path(preprocessing_dir, "preprocessing_manifest.json")
build_manifest_path <- file.path(
  preprocessing_dir, "preprocessing_build_manifest.json"
)
complete_path <- file.path(preprocessing_dir, "COMPLETE.json")
scale_audit_path <- file.path(preprocessing_dir, "input_scale_audit.csv")
algorithm_config_path <- file.path(alignment_dir, "config.json")

sha256 <- function(path) {
  digest::digest(
    file = normalizePath(path, mustWork = TRUE),
    algo = "sha256",
    serialize = FALSE
  )
}

write_json <- function(value, path) {
  jsonlite::write_json(
    value, path, pretty = TRUE, auto_unbox = TRUE,
    null = "null", digits = 17
  )
  write("", file = path, append = TRUE)
}

write_csv_gz <- function(frame, path) {
  readr::write_csv(frame, path, na = "")
}

read_fixed <- function(path) {
  suppressMessages(readr::read_csv(
    path,
    col_types = readr::cols(
      .default = readr::col_double(),
      cell_id = readr::col_character(),
      dataset = readr::col_character(),
      donor = readr::col_character(),
      unit_key = readr::col_character()
    ),
    progress = FALSE,
    show_col_types = FALSE,
    name_repair = "minimal"
  ))
}

assert_fixed <- function(frame, dataset) {
  if (!identical(colnames(frame), c(id_columns, markers))) {
    stop(sprintf("%s frozen columns/order changed.", dataset), call. = FALSE)
  }
  if (nrow(frame) != expected_counts[[dataset]] || any(frame$dataset != dataset)) {
    stop(sprintf("%s frozen identities/count changed.", dataset), call. = FALSE)
  }
  if (anyNA(frame$cell_id) || any(!nzchar(frame$cell_id)) ||
      anyDuplicated(frame$cell_id)) {
    stop(sprintf("%s has incomplete or duplicated cell IDs.", dataset), call. = FALSE)
  }
}

replace_expression <- function(frozen, values) {
  values <- as.matrix(values)
  if (!identical(dim(values), c(nrow(frozen), length(markers)))) {
    stop("Replacement expression has the wrong dimensions.", call. = FALSE)
  }
  if (!all(is.finite(values))) {
    stop("Replacement expression contains NA/NaN/Inf.", call. = FALSE)
  }
  result <- as.data.frame(frozen[, id_columns, drop = FALSE], check.names = FALSE)
  for (index in seq_along(markers)) {
    result[[markers[[index]]]] <- as.numeric(values[, index])
  }
  result
}

generated_targets <- c(
  manifest_path, build_manifest_path, complete_path, scale_audit_path,
  algorithm_config_path,
  file.path(data_dir, "bm_codex_analysis_ready.csv.gz"),
  file.path(data_dir, "bm_cytof_analysis_ready.csv.gz"),
  file.path(data_dir, "triana_abseq_analysis_ready.csv.gz"),
  file.path(data_dir, "combined_analysis_ready.csv.gz")
)
existing_targets <- generated_targets[file.exists(generated_targets)]
if (length(existing_targets)) {
  stop(
    sprintf(
      "Refusing to overwrite existing preprocessing artifacts: %s",
      paste(existing_targets, collapse = ", ")
    ),
    call. = FALSE
  )
}
if (!file.exists(marker_file) || !file.exists(harmonization_file)) {
  stop("Static marker contracts are absent.", call. = FALSE)
}
if (!identical(readLines(marker_file, warn = FALSE), markers)) {
  stop("markers.txt order changed.", call. = FALSE)
}

stage <- tempfile(pattern = ".fastmnn-preprocessing-stage-", tmpdir = preprocessing_dir)
if (!dir.create(stage, recursive = FALSE)) {
  stop("Could not create preprocessing staging directory.", call. = FALSE)
}
on.exit(if (dir.exists(stage)) unlink(stage, recursive = TRUE, force = TRUE), add = TRUE)
stage_data <- file.path(stage, "data")
dir.create(stage_data)

# Freeze identities and row order using label-free benchmark matrices only.
codex_frozen_path <- file.path(
  benchmark_dir, "bm_codex", "alignment_input_14.csv.gz"
)
cytof_frozen_path <- file.path(
  benchmark_dir, "bm_cytof", "alignment_input_14.csv.gz"
)
triana_frozen_path <- file.path(
  benchmark_dir, "triana_abseq", "alignment_input_14.csv.gz"
)
codex_frozen <- read_fixed(codex_frozen_path)
cytof_frozen <- read_fixed(cytof_frozen_path)
triana_frozen <- read_fixed(triana_frozen_path)
assert_fixed(codex_frozen, "BM CODEX")
assert_fixed(cytof_frozen, "BM CyTOF")
assert_fixed(triana_frozen, "Triana AbSeq")

# BM CODEX: use the provider's source-study CLR analysis layer. The layer was
# computed per complete sample before benchmark sampling; no second transform.
codex_rds <- file.path(
  raw_dir, "BM_Codex", "Normal_Bone_Marrow_CODEX_Atlas_Seurat_v2.rds"
)
codex_object <- readRDS(codex_rds)
if (!("CODEX" %in% names(codex_object@assays))) {
  stop("CODEX assay is absent from the source RDS.", call. = FALSE)
}
codex_layer <- codex_object[["CODEX"]]@data
codex_source_markers <- c(
  "CD117", "CD11B", "CD11C", "CD123", "CD14", "CD3e", "CD33",
  "CD34", "CD38", "CD4", "CD45", "CD45RA", "CD61", "CD8"
)
if (!all(codex_source_markers %in% rownames(codex_layer))) {
  stop("A required CODEX source marker is absent.", call. = FALSE)
}
codex_index <- match(codex_frozen$cell_id, colnames(codex_layer))
if (anyNA(codex_index)) {
  stop("Frozen CODEX cells are absent from source RDS.", call. = FALSE)
}
codex_values <- t(as.matrix(
  codex_layer[codex_source_markers, codex_index, drop = FALSE]
))
colnames(codex_values) <- markers
codex_ready <- replace_expression(codex_frozen, codex_values)
rm(codex_object, codex_layer, codex_values)
gc(verbose = FALSE)

# BM CyTOF: the frozen values are continuous CCT-subtracted/bead-normalized
# intensities. Apply the source paper's asinh cofactor 5 exactly once.
cytof_values <- as.matrix(cytof_frozen[, markers, drop = FALSE])
cytof_ready <- replace_expression(cytof_frozen, asinh(cytof_values / 5))
rm(cytof_values)

# Triana AbSeq: calculate a log-like Seurat CLR scale within each cell over all
# 97 raw antibody features, then select the shared marker intersection.
triana_export <- file.path(raw_dir, "Triana_2021", "exported")
triana_count_path <- file.path(triana_export, "ab_counts.mtx")
triana_feature_path <- file.path(triana_export, "ab_features.tsv")
triana_barcode_path <- file.path(triana_export, "ab_barcodes.tsv")
triana_features <- readLines(triana_feature_path, warn = FALSE)
triana_barcodes <- readLines(triana_barcode_path, warn = FALSE)
triana_counts <- Matrix::readMM(triana_count_path)
if (nrow(triana_counts) == length(triana_barcodes) &&
    ncol(triana_counts) == length(triana_features)) {
  triana_counts <- Matrix::t(triana_counts)
} else if (!(nrow(triana_counts) == length(triana_features) &&
             ncol(triana_counts) == length(triana_barcodes))) {
  stop("Triana matrix dimensions do not match features/barcodes.", call. = FALSE)
}
triana_index <- match(triana_frozen$cell_id, triana_barcodes)
if (anyNA(triana_index)) {
  stop("Frozen Triana cells are absent from raw ADT counts.", call. = FALSE)
}
triana_source_markers <- c(
  "CD117-AB", "CD11b-AB", "CD11c-AB", "CD123-AB", "CD14-AB",
  "CD3-AB", "CD33-AB", "CD34-AB", "CD38-AB", "CD4-AB", "CD45-AB",
  "CD45RA-AB", "CD61-AB", "CD8-AB"
)
triana_marker_index <- match(triana_source_markers, triana_features)
if (anyNA(triana_marker_index)) {
  stop("A required Triana source marker is absent.", call. = FALSE)
}
triana_all_counts <- as.matrix(triana_counts[, triana_index, drop = FALSE])
clr_denominator <- exp(colSums(log1p(triana_all_counts)) / nrow(triana_all_counts))
triana_clr <- log1p(sweep(triana_all_counts, 2L, clr_denominator, FUN = "/"))
triana_values <- t(triana_clr[triana_marker_index, , drop = FALSE])
colnames(triana_values) <- markers
triana_ready <- replace_expression(triana_frozen, triana_values)
rm(triana_counts, triana_all_counts, triana_clr, triana_values)
gc(verbose = FALSE)

ready <- list(
  "BM CODEX" = codex_ready,
  "BM CyTOF" = cytof_ready,
  "Triana AbSeq" = triana_ready
)
combined <- do.call(rbind, ready)
rownames(combined) <- NULL
if (nrow(combined) != expected_total || anyDuplicated(combined$cell_id)) {
  stop("Combined fastMNN input differs from frozen cohort.", call. = FALSE)
}
if (!identical(unique(combined$dataset), dataset_order)) {
  stop("Combined dataset/progressive-merge order changed.", call. = FALSE)
}
selected <- suppressMessages(readr::read_csv(
  selection_file,
  col_types = readr::cols(.default = readr::col_character()),
  progress = FALSE,
  show_col_types = FALSE
))
if (!identical(as.character(selected$cell_id), combined$cell_id) ||
    !identical(as.character(selected$dataset), combined$dataset) ||
    !identical(as.character(selected$donor), combined$donor)) {
  stop("Combined identities/order differ from selected_cells.csv.gz.", call. = FALSE)
}
numeric_values <- as.matrix(combined[, markers, drop = FALSE])
if (!all(is.finite(numeric_values))) {
  stop("Combined matrix contains non-finite values.", call. = FALSE)
}
l2_norm <- sqrt(rowSums(numeric_values ^ 2))
zero_l2_count <- sum(l2_norm == 0)
zero_l2_by_dataset <- vapply(dataset_order, function(dataset) {
  sum(l2_norm[combined$dataset == dataset] == 0)
}, integer(1))
if (any(apply(numeric_values, 2L, stats::sd) <= 1e-12)) {
  stop("Combined matrix contains a constant marker.", call. = FALSE)
}

stage_outputs <- c(
  "BM CODEX" = file.path(stage_data, "bm_codex_analysis_ready.csv.gz"),
  "BM CyTOF" = file.path(stage_data, "bm_cytof_analysis_ready.csv.gz"),
  "Triana AbSeq" = file.path(stage_data, "triana_abseq_analysis_ready.csv.gz")
)
for (dataset in dataset_order) {
  write_csv_gz(ready[[dataset]], stage_outputs[[dataset]])
}
combined_stage <- file.path(stage_data, "combined_analysis_ready.csv.gz")
write_csv_gz(combined, combined_stage)

audit_rows <- lapply(dataset_order, function(dataset) {
  frame <- ready[[dataset]]
  do.call(rbind, lapply(markers, function(marker) {
    values <- frame[[marker]]
    data.frame(
      dataset = dataset,
      marker = marker,
      cells = length(values),
      minimum = min(values),
      median = median(values),
      mean = mean(values),
      maximum = max(values),
      stringsAsFactors = FALSE
    )
  }))
})
scale_audit_stage <- file.path(stage, "input_scale_audit.csv")
readr::write_csv(do.call(rbind, audit_rows), scale_audit_stage)

script_hash <- sha256(script_path)
dataset_entries <- unname(list(
  list(
    dataset = "BM CODEX",
    technology = "CODEX multiplexed fluorescence imaging",
    analysis_ready_matrix_file = "data/bm_codex_analysis_ready.csv.gz",
    analysis_ready_matrix_sha256 = sha256(stage_outputs[["BM CODEX"]]),
    source_file = codex_rds,
    source_file_sha256 = sha256(codex_rds),
    frozen_alignment_input_file = codex_frozen_path,
    frozen_alignment_input_sha256 = sha256(codex_frozen_path),
    source_state = "provider_analysis_scale",
    assay_or_layer = "Normal_Bone_Marrow_CODEX_Atlas_Seurat_v2.rds: CODEX assay data slot",
    processing_steps = list(
      "Retain the exact 30,000 frozen benchmark cell IDs and row order.",
      "Extract the provider's per-sample CLR-normalized CODEX data layer for the 14 shared markers.",
      "Rename source CD3e to canonical CD3; apply no second transform or feature scaling."
    ),
    transform_formula = "Provider NormalizeData(normalization.method='CLR', margin=1); no additional transform",
    transform_scope = "Computed by the source study within each complete CODEX sample before frozen-cell selection.",
    processing_reference = "Bandyopadhyay et al. 2024 Methods and the immutable provider CODEX data slot.",
    processing_code_file = "build_inputs.R",
    processing_code_sha256 = script_hash,
    universal_positive_median_arcsinh_not_used = TRUE,
    marker_wise_zscore_not_used = TRUE,
    double_transform_checked = TRUE
  ),
  list(
    dataset = "BM CyTOF",
    technology = "CyTOF",
    analysis_ready_matrix_file = "data/bm_cytof_analysis_ready.csv.gz",
    analysis_ready_matrix_sha256 = sha256(stage_outputs[["BM CyTOF"]]),
    source_file = cytof_frozen_path,
    source_file_sha256 = sha256(cytof_frozen_path),
    source_state = "technically_normalized_continuous",
    assay_or_layer = "Frozen CCT-subtracted/bead-normalized annotated FCS event values",
    processing_steps = list(
      "Retain the exact 30,000 frozen H1/H2 benchmark events and row order.",
      "Preserve continuous CCT-subtracted/bead-normalized values, including small negative background values.",
      "Apply asinh(x/5) exactly once; do not z-score or estimate a replacement cofactor."
    ),
    transform_formula = "asinh(x / 5)",
    transform_scope = "Element-wise over every frozen cell and each of the 14 shared markers.",
    processing_reference = "Levine et al. 2015 Methods: bead-normalized values followed by arcsinh cofactor 5.",
    processing_code_file = "build_inputs.R",
    processing_code_sha256 = script_hash,
    universal_positive_median_arcsinh_not_used = TRUE,
    marker_wise_zscore_not_used = TRUE,
    double_transform_checked = TRUE
  ),
  list(
    dataset = "Triana AbSeq",
    technology = "AbSeq antibody-derived tags",
    analysis_ready_matrix_file = "data/triana_abseq_analysis_ready.csv.gz",
    analysis_ready_matrix_sha256 = sha256(stage_outputs[["Triana AbSeq"]]),
    source_file = triana_count_path,
    source_file_sha256 = sha256(triana_count_path),
    source_files = list(
      list(path = triana_count_path, sha256 = sha256(triana_count_path)),
      list(path = triana_feature_path, sha256 = sha256(triana_feature_path)),
      list(path = triana_barcode_path, sha256 = sha256(triana_barcode_path))
    ),
    frozen_alignment_input_file = triana_frozen_path,
    frozen_alignment_input_sha256 = sha256(triana_frozen_path),
    source_state = "raw_counts_converted_to_log_like_clr",
    assay_or_layer = "Raw antibody UMI counts across all 97 antibody features",
    processing_steps = list(
      "Retain the exact 28,031 frozen BM1/BM2/BM3 cells and row order.",
      "Compute Seurat-style CLR margin=2 independently within each cell over all 97 antibody features.",
      "Select the 14 shared proteins only after CLR; do not use scale.data or integrated values."
    ),
    transform_formula = "log1p(x / exp(sum(log1p(x_all_97_features)) / 97)) within each cell",
    transform_scope = "Independently within each frozen cell over all 97 antibody features before marker selection.",
    processing_reference = "Triana et al. 2021 raw antibody counts and the standard Seurat CLR convention for antibody-count data.",
    processing_code_file = "build_inputs.R",
    processing_code_sha256 = script_hash,
    universal_positive_median_arcsinh_not_used = TRUE,
    marker_wise_zscore_not_used = TRUE,
    double_transform_checked = TRUE
  )
))

preprocessing_manifest <- list(
  schema_version = "final_xu.fastmnn_preprocessing.v1",
  combined_input_file = "data/combined_analysis_ready.csv.gz",
  combined_input_sha256 = sha256(combined_stage),
  shared_marker_policy = "Exact ordered intersection of 14 harmonized proteins frozen before method fitting; CD19 is held out.",
  shared_marker_selection_used_celltype_or_outcome_labels = FALSE,
  merge_order_selected_using_posthoc_labels_or_metrics = FALSE,
  canonical_parameters_selected_using_posthoc_labels_or_metrics = FALSE,
  correction_marker_file = "markers.txt",
  correction_marker_sha256 = sha256(marker_file),
  marker_harmonization_manifest_file = "marker_harmonization_manifest.csv",
  marker_harmonization_manifest_sha256 = sha256(harmonization_file),
  cell_selection_manifest_file = "../../../../benchmark_data/bone_marrow/selected_cells.csv.gz",
  cell_selection_manifest_sha256 = sha256(selection_file),
  cell_selection_id_column = "cell_id",
  dataset_order_is_progressive_merge_order = TRUE,
  labels_or_outcomes_read_during_preprocessing = FALSE,
  other_comparison_method_outputs_read = FALSE,
  algorithm_internal_cosine_normalization = TRUE,
  zero_l2_cell_policy = "Retain unchanged; official batchelor cosineNorm uses pmax(1e-8, L2). No deletion or epsilon is applied upstream.",
  zero_l2_cell_count = as.integer(zero_l2_count),
  zero_l2_cell_count_by_dataset = as.list(stats::setNames(
    as.integer(zero_l2_by_dataset), names(zero_l2_by_dataset)
  )),
  datasets = dataset_entries
)
manifest_stage <- file.path(stage, "preprocessing_manifest.json")
write_json(preprocessing_manifest, manifest_stage)

algorithm_config <- list(
  schema_version = "final_xu.fastmnn_repro.v1",
  method_label = "fastMNN (batchelor 1.18.1; corrected embedding)",
  cell_id_column = "cell_id",
  batch_column = "dataset",
  sample_column = "unit_key",
  batch_levels = as.list(dataset_order),
  input_scale = list(
    status = "analysis_ready",
    name = "BM platform-supported CLR/asinh shared-protein scale",
    description = paste(
      "BM CODEX provider CLR, BM CyTOF asinh(x/5), and Triana AbSeq",
      "97-feature per-cell CLR; no universal transform or marker-wise z-score."
    ),
    preprocessing_record_file = "../preprocessing/preprocessing_manifest.json",
    preprocessing_record_sha256 = sha256(manifest_stage),
    input_sha256 = sha256(combined_stage),
    marker_file_sha256 = sha256(marker_file),
    algorithm_internal_cosine_normalization = TRUE,
    algorithm_applies_other_expression_transform = FALSE
  ),
  algorithm_parameters = list(
    seed = 42L,
    rng_kind = "Mersenne-Twister",
    normal_kind = "Inversion",
    sample_kind = "Rejection",
    k = 20L,
    prop_k = NULL,
    cos_norm = TRUE,
    ndist = 3,
    requested_d = 50L,
    d_policy = "request 50; batchelor realizes min(50, n_markers - 1, n_cells - 1)",
    deferred = TRUE,
    weights = NULL,
    get_variance = TRUE,
    merge_order = NULL,
    auto_merge = FALSE,
    min_batch_skip = 0,
    subset_row = NULL,
    correct_all = FALSE,
    pca_solver = "IrlbaParam",
    neighbor_solver = "KmknnParam",
    parallel_solver = "SerialParam"
  )
)
config_stage <- file.path(stage, "config.json")
write_json(algorithm_config, config_stage)

source_files <- c(
  codex_rds, codex_frozen_path, cytof_frozen_path, triana_frozen_path,
  triana_count_path, triana_feature_path, triana_barcode_path, selection_file
)
build_manifest <- list(
  schema_version = "final_xu.bone_marrow.fastmnn_input_build.v1",
  status = "complete",
  method = "fastMNN (batchelor 1.18.1; corrected embedding)",
  labels_or_outcomes_read = FALSE,
  comparison_method_outputs_read = FALSE,
  locked_cells = expected_total,
  dataset_order = as.list(dataset_order),
  dataset_cell_counts = as.list(stats::setNames(as.integer(expected_counts), names(expected_counts))),
  markers = as.list(markers),
  cd19_in_fit = FALSE,
  zero_l2_cell_count = as.integer(zero_l2_count),
  zero_l2_cell_count_by_dataset = as.list(stats::setNames(
    as.integer(zero_l2_by_dataset), names(zero_l2_by_dataset)
  )),
  source_files = lapply(source_files, function(path) {
    list(path = normalizePath(path), sha256 = sha256(path))
  }),
  outputs = list(
    bm_codex = list(file = "data/bm_codex_analysis_ready.csv.gz", sha256 = sha256(stage_outputs[["BM CODEX"]])),
    bm_cytof = list(file = "data/bm_cytof_analysis_ready.csv.gz", sha256 = sha256(stage_outputs[["BM CyTOF"]])),
    triana_abseq = list(file = "data/triana_abseq_analysis_ready.csv.gz", sha256 = sha256(stage_outputs[["Triana AbSeq"]])),
    combined = list(file = "data/combined_analysis_ready.csv.gz", sha256 = sha256(combined_stage)),
    input_scale_audit = list(file = "input_scale_audit.csv", sha256 = sha256(scale_audit_stage))
  ),
  preprocessing_manifest_sha256 = sha256(manifest_stage),
  algorithm_config_sha256 = sha256(config_stage),
  script = list(path = script_path, sha256 = script_hash),
  runtime = list(
    R = as.character(getRversion()),
    Matrix = as.character(packageVersion("Matrix")),
    SeuratObject = as.character(packageVersion("SeuratObject")),
    readr = as.character(packageVersion("readr")),
    jsonlite = as.character(packageVersion("jsonlite")),
    digest = as.character(packageVersion("digest"))
  )
)
build_manifest_stage <- file.path(stage, "preprocessing_build_manifest.json")
write_json(build_manifest, build_manifest_stage)
complete <- list(
  schema_version = "final_xu.bone_marrow.fastmnn_preprocessing_complete.v1",
  status = "complete",
  preprocessing_manifest_sha256 = sha256(manifest_stage),
  preprocessing_build_manifest_sha256 = sha256(build_manifest_stage),
  algorithm_config_sha256 = sha256(config_stage)
)
complete_stage <- file.path(stage, "COMPLETE.json")
write_json(complete, complete_stage)

# Publish generated artifacts only after every validation and manifest succeeds.
if (dir.exists(data_dir)) {
  if (length(list.files(data_dir, all.files = TRUE, no.. = TRUE))) {
    stop("Preprocessing data directory became non-empty during build.", call. = FALSE)
  }
  unlink(data_dir, recursive = TRUE, force = TRUE)
}
if (!file.rename(stage_data, data_dir)) {
  stop("Could not publish preprocessing data directory.", call. = FALSE)
}
publish <- list(
  scale_audit_stage = scale_audit_path,
  manifest_stage = manifest_path,
  build_manifest_stage = build_manifest_path,
  complete_stage = complete_path
)
for (source_name in names(publish)) {
  source <- get(source_name)
  target <- publish[[source_name]]
  if (!file.rename(source, target)) {
    stop(sprintf("Could not publish %s.", target), call. = FALSE)
  }
}
dir.create(alignment_dir, recursive = TRUE, showWarnings = FALSE)
if (!file.rename(config_stage, algorithm_config_path)) {
  stop("Could not publish alignment config.", call. = FALSE)
}

cat(sprintf(
  "Built bone-marrow fastMNN input: %s cells x %s markers; %s\n",
  format(expected_total, big.mark = ","), length(markers),
  file.path(data_dir, "combined_analysis_ready.csv.gz")
))
