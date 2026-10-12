#!/usr/bin/env Rscript

# Build method-specific, analysis-ready inputs for the frozen TNBC cyCombine
# benchmark. This script reads only frozen identities/order, marker matrices,
# and provider expression objects. It never reads cell types, outcomes, spatial
# labels, RAMP results, MaxFuse results, or any other corrected matrix.

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_argument)) stop("Cannot determine script location.", call. = FALSE)
script_path <- sub("^--file=", "", script_argument[[1L]])
script_path <- gsub("~+~", " ", script_path, fixed = TRUE)
script_path <- normalizePath(script_path, mustWork = TRUE)
preprocessing_dir <- dirname(script_path)
method_dir <- dirname(preprocessing_dir)
final_xu <- normalizePath(file.path(method_dir, "..", "..", ".."), mustWork = TRUE)
benchmark_dir <- file.path(final_xu, "benchmark_data", "tnbc")
raw_dir <- file.path(final_xu, "raw_data", "tnbc")
algorithm_dir <- file.path(final_xu, "algorithm", "cycombine")
.libPaths(c(file.path(algorithm_dir, ".Rlib"), .libPaths()))

suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(SummarizedExperiment)
  library(digest)
  library(jsonlite)
  library(readr)
})

markers <- c(
  "CD11C", "CD20", "CD3", "CD31", "CD4", "CD68",
  "CD8", "HLA_DR", "KI67", "PANCK", "SMA"
)
id_columns <- c("cell_id", "dataset", "donor", "unit_key")
dataset_order <- c("Tietscher IMC", "Meyer IMC", "Shiao CODEX")
expected_counts <- c(
  "Tietscher IMC" = 20000L,
  "Meyer IMC" = 22701L,
  "Shiao CODEX" = 20000L
)
data_dir <- file.path(preprocessing_dir, "data")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

sha256 <- function(path) {
  digest::digest(
    file = normalizePath(path, mustWork = TRUE),
    algo = "sha256",
    serialize = FALSE
  )
}

read_fixed <- function(path) {
  header <- names(suppressMessages(readr::read_csv(
    path,
    n_max = 0L,
    progress = FALSE,
    show_col_types = FALSE,
    name_repair = "minimal"
  )))
  types <- if ("source_cell_id" %in% header) {
    readr::cols(
      .default = readr::col_double(),
      cell_id = readr::col_character(),
      source_cell_id = readr::col_character(),
      dataset = readr::col_character(),
      donor = readr::col_character(),
      unit_key = readr::col_character()
    )
  } else {
    readr::cols(
      .default = readr::col_double(),
      cell_id = readr::col_character(),
      dataset = readr::col_character(),
      donor = readr::col_character(),
      unit_key = readr::col_character()
    )
  }
  suppressMessages(readr::read_csv(
    path,
    col_types = types,
    progress = FALSE,
    show_col_types = FALSE,
    name_repair = "minimal"
  ))
}

write_csv_gz_atomic <- function(frame, path) {
  temporary <- file.path(dirname(path), paste0(".", basename(path), ".tmp.csv.gz"))
  if (file.exists(temporary)) unlink(temporary)
  readr::write_csv(frame, temporary, na = "")
  if (!file.rename(temporary, path)) {
    unlink(temporary)
    stop(sprintf("Could not publish %s", path), call. = FALSE)
  }
}

assert_fixed_contract <- function(frame, dataset) {
  expected_columns <- c("cell_id", "source_cell_id", "dataset", "donor", "unit_key", markers)
  if (!identical(colnames(frame), expected_columns)) {
    stop(sprintf("%s frozen input columns changed.", dataset), call. = FALSE)
  }
  if (nrow(frame) != expected_counts[[dataset]] || any(frame$dataset != dataset)) {
    stop(sprintf("%s frozen cell count or dataset identity changed.", dataset), call. = FALSE)
  }
  if (anyNA(frame$cell_id) || any(!nzchar(frame$cell_id)) || anyDuplicated(frame$cell_id)) {
    stop(sprintf("%s cell IDs are incomplete or duplicated.", dataset), call. = FALSE)
  }
  values <- as.matrix(frame[, markers, drop = FALSE])
  if (!all(is.finite(values)) || any(values < 0)) {
    stop(sprintf("%s frozen marker matrix must be finite and nonnegative.", dataset), call. = FALSE)
  }
}

replace_expression <- function(frozen, values) {
  if (!identical(dim(values), c(nrow(frozen), length(markers)))) {
    stop("Replacement expression matrix has the wrong dimensions.", call. = FALSE)
  }
  if (!all(is.finite(values))) stop("Replacement expression contains NA/NaN/Inf.", call. = FALSE)
  result <- as.data.frame(frozen[, id_columns, drop = FALSE], check.names = FALSE)
  for (index in seq_along(markers)) result[[markers[[index]]]] <- as.numeric(values[, index])
  result
}

selected <- suppressMessages(readr::read_csv(
  file.path(benchmark_dir, "selected_cells.csv.gz"),
  col_types = readr::cols(
    .default = readr::col_character(),
    cell_order = readr::col_integer(),
    source_index_zero_based = readr::col_integer()
  ),
  progress = FALSE,
  show_col_types = FALSE
))
if (nrow(selected) != 62701L ||
    !identical(selected$cell_order, 0:62700) ||
    anyDuplicated(selected$cell_id) ||
    !identical(unique(selected$dataset), dataset_order)) {
  stop("Frozen TNBC cell selection/order changed.", call. = FALSE)
}

# Read only label-free, frozen benchmark matrices.
tietscher_frozen <- read_fixed(file.path(benchmark_dir, "tietscher_imc", "alignment_input_11.csv.gz"))
meyer_frozen <- read_fixed(file.path(benchmark_dir, "meyer_imc", "alignment_input_11.csv.gz"))
shiao_frozen <- read_fixed(file.path(benchmark_dir, "shiao_codex", "alignment_input_11.csv.gz"))
assert_fixed_contract(tietscher_frozen, "Tietscher IMC")
assert_fixed_contract(meyer_frozen, "Meyer IMC")
assert_fixed_contract(shiao_frozen, "Shiao CODEX")

extract_provider_exprs <- function(
    frozen,
    dataset,
    rds_path,
    source_markers) {
  local_selection <- selected[selected$dataset == dataset, , drop = FALSE]
  if (!identical(as.character(local_selection$cell_id), as.character(frozen$cell_id))) {
    stop(sprintf("%s frozen IDs/order differ from selected_cells.csv.gz.", dataset), call. = FALSE)
  }
  object <- readRDS(rds_path)
  if (!inherits(object, "SingleCellExperiment")) {
    stop(sprintf("%s provider object is not a SingleCellExperiment.", dataset), call. = FALSE)
  }
  if (!all(c("counts", "exprs") %in% SummarizedExperiment::assayNames(object))) {
    stop(sprintf("%s provider object lacks counts/exprs assays.", dataset), call. = FALSE)
  }
  if (!all(source_markers %in% rownames(object))) {
    stop(sprintf("%s provider object lacks required source markers.", dataset), call. = FALSE)
  }
  positions <- as.integer(local_selection$source_index_zero_based) + 1L
  if (anyNA(positions) || any(positions < 1L) || any(positions > ncol(object)) || anyDuplicated(positions)) {
    stop(sprintf("%s frozen provider positions are invalid.", dataset), call. = FALSE)
  }
  observed_ids <- as.character(colnames(object)[positions])
  if (!identical(observed_ids, as.character(local_selection$source_cell_id))) {
    stop(sprintf("%s source index/cell ID lock failed.", dataset), call. = FALSE)
  }
  provider_counts <- t(as.matrix(
    SummarizedExperiment::assay(object, "counts")[source_markers, positions, drop = FALSE]
  ))
  provider_exprs <- t(as.matrix(
    SummarizedExperiment::assay(object, "exprs")[source_markers, positions, drop = FALSE]
  ))
  colnames(provider_counts) <- markers
  colnames(provider_exprs) <- markers
  frozen_values <- as.matrix(frozen[, markers, drop = FALSE])
  count_roundtrip_error <- max(abs(provider_counts - frozen_values))
  formula_error <- max(abs(provider_exprs - asinh(provider_counts)))
  if (!is.finite(count_roundtrip_error) || count_roundtrip_error > 1e-10) {
    stop(sprintf("%s frozen/source count mismatch: %.17g", dataset, count_roundtrip_error), call. = FALSE)
  }
  if (!is.finite(formula_error) || formula_error > 1e-12) {
    stop(sprintf("%s exprs != asinh(counts): %.17g", dataset, formula_error), call. = FALSE)
  }
  result <- list(
    ready = replace_expression(frozen, provider_exprs),
    audit = list(
      cells = nrow(frozen),
      source_index_and_id_exact = TRUE,
      frozen_counts_maximum_absolute_error = count_roundtrip_error,
      exprs_asinh_counts_maximum_absolute_error = formula_error,
      provider_assay = "exprs"
    )
  )
  rm(object, provider_counts, provider_exprs)
  gc(verbose = FALSE)
  result
}

tietscher_rds <- file.path(
  raw_dir, "Tietscher_2023", "Protein_panel_SingleCellExperiment_object.rds"
)
meyer_rds <- file.path(
  raw_dir, "Meyer_2025", "processed_r_objects", "sce_ALL.rds"
)
tietscher_source_markers <- c(
  "CD11c", "CD20", "CD3", "CD31_vWF", "CD4", "CD68",
  "CD8a", "HLA-DR", "Ki-67", "panCK", "SMA"
)
meyer_source_markers <- c(
  "CD11c", "CD20", "CD3", "CD31_vWF", "CD4", "CD68",
  "CD8a", "HLADR", "Ki67", "panCK", "SMA"
)
tietscher_extraction <- extract_provider_exprs(
  tietscher_frozen, "Tietscher IMC", tietscher_rds, tietscher_source_markers
)
tietscher_ready <- tietscher_extraction$ready
meyer_extraction <- extract_provider_exprs(
  meyer_frozen, "Meyer IMC", meyer_rds, meyer_source_markers
)
meyer_ready <- meyer_extraction$ready

# Shiao CODEX: the frozen matrix is the benchmark-locked arithmetic mean of
# nuclei_mean and membrane_mean. The source paper does not define a nonlinear
# transform for these averages. cofactor-1 asinh is a predeclared comparator
# adaptation matching the two IMC expression scales; it is not provider scale.
shiao_values <- as.matrix(shiao_frozen[, markers, drop = FALSE])
shiao_ready <- replace_expression(shiao_frozen, asinh(shiao_values / 1))
rm(shiao_values)

ready_tables <- list(tietscher_ready, meyer_ready, shiao_ready)
names(ready_tables) <- dataset_order
combined <- do.call(rbind, ready_tables)
rownames(combined) <- NULL
if (nrow(combined) != 62701L || anyDuplicated(combined$cell_id)) {
  stop("Combined cyCombine input is not the frozen 62,701-cell benchmark.", call. = FALSE)
}
if (!identical(unique(combined$dataset), dataset_order) ||
    !identical(as.character(selected$cell_id), as.character(combined$cell_id)) ||
    !identical(as.character(selected$dataset), as.character(combined$dataset)) ||
    !identical(as.character(selected$donor), as.character(combined$donor)) ||
    !identical(as.character(selected$unit_key), as.character(combined$unit_key))) {
  stop("Combined identities/order differ from selected_cells.csv.gz.", call. = FALSE)
}

output_paths <- c(
  "Tietscher IMC" = file.path(data_dir, "tietscher_imc_analysis_ready.csv.gz"),
  "Meyer IMC" = file.path(data_dir, "meyer_imc_analysis_ready.csv.gz"),
  "Shiao CODEX" = file.path(data_dir, "shiao_codex_analysis_ready.csv.gz")
)
for (dataset in dataset_order) write_csv_gz_atomic(ready_tables[[dataset]], output_paths[[dataset]])
combined_path <- file.path(data_dir, "combined_analysis_ready.csv.gz")
write_csv_gz_atomic(combined, combined_path)

# Re-read every persisted table before publishing provenance. This catches CSV
# type/order drift and proves that the per-dataset row-bind recreates combined.
persisted <- lapply(output_paths, read_fixed)
persisted_combined <- read_fixed(combined_path)
reconstructed <- do.call(rbind, persisted)
rownames(reconstructed) <- NULL
for (column in c(id_columns, markers)) {
  equal <- if (column %in% markers) {
    identical(as.numeric(reconstructed[[column]]), as.numeric(persisted_combined[[column]]))
  } else {
    identical(as.character(reconstructed[[column]]), as.character(persisted_combined[[column]]))
  }
  if (!equal) stop(sprintf("Persisted row-bind mismatch in %s.", column), call. = FALSE)
}

audit_rows <- lapply(dataset_order, function(dataset) {
  frame <- ready_tables[[dataset]]
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
      sum = sum(values),
      variance = var(values),
      exact_zero_cells = sum(values == 0),
      stringsAsFactors = FALSE
    )
  }))
})
scale_audit <- do.call(rbind, audit_rows)
if (any(!is.finite(as.matrix(scale_audit[, c("minimum", "median", "mean", "maximum", "sum", "variance")]))) ||
    any(scale_audit$sum == 0) || any(scale_audit$variance <= 0)) {
  stop("Analysis-ready input has a non-finite, zero-sum, or constant batch-marker.", call. = FALSE)
}
scale_audit_path <- file.path(preprocessing_dir, "input_scale_audit.csv")
readr::write_csv(scale_audit, scale_audit_path)

marker_file <- file.path(preprocessing_dir, "markers.txt")
harmonization_file <- file.path(preprocessing_dir, "marker_harmonization_manifest.csv")
selection_file <- file.path(benchmark_dir, "selected_cells.csv.gz")
benchmark_marker_file <- file.path(benchmark_dir, "marker_manifest.csv")
if (!identical(readLines(marker_file, warn = FALSE), markers)) stop("markers.txt drifted.", call. = FALSE)
if (!identical(
    as.data.frame(readr::read_csv(harmonization_file, show_col_types = FALSE)),
    as.data.frame(readr::read_csv(benchmark_marker_file, show_col_types = FALSE)))) {
  stop("Local marker harmonization no longer mirrors benchmark_data.", call. = FALSE)
}
script_hash <- sha256(script_path)

dataset_entries <- unname(list(
  list(
    dataset = "Tietscher IMC",
    technology = "Imaging mass cytometry (IMC)",
    input_matrix_file = "data/tietscher_imc_analysis_ready.csv.gz",
    input_matrix_sha256 = sha256(output_paths[["Tietscher IMC"]]),
    source_state = "normalized_and_nonlinearly_transformed",
    assay_or_layer = "Protein_panel_SingleCellExperiment_object.rds: exprs assay for frozen cells and source channels",
    processing_steps = list(
      "Retain the exact 20,000 frozen BB165 cell IDs and row order.",
      "Verify source indices and cell IDs against the provider SingleCellExperiment.",
      "Extract the provider exprs assay and verify exprs equals asinh(counts) with cofactor 1; apply no second transform, scaling, centering, or 0-1 normalization."
    ),
    cycombine_input_scale = "Provider IMC expression assay: asinh mean pixel intensity, cofactor 1",
    transform_formula = "Provider exprs = asinh(counts / 1); no additional transform",
    transform_scope = "Element-wise source-study transform over the provider protein-panel counts assay before frozen-cell selection.",
    processing_reference = "Tietscher et al. 2023 Methods, IMC analysis workflow and clustering; provider exprs/counts assay identity verified by build_inputs.R.",
    processing_code_file = "build_inputs.R",
    processing_code_sha256 = script_hash,
    double_transform_checked = TRUE
  ),
  list(
    dataset = "Meyer IMC",
    technology = "Imaging mass cytometry (IMC)",
    input_matrix_file = "data/meyer_imc_analysis_ready.csv.gz",
    input_matrix_sha256 = sha256(output_paths[["Meyer IMC"]]),
    source_state = "normalized_and_nonlinearly_transformed",
    assay_or_layer = "processed_r_objects/sce_ALL.rds: exprs assay for frozen cells and source channels",
    processing_steps = list(
      "Retain the exact 22,701 frozen cell IDs from five coverage-guided donors (selection seed 20260960) and their row order.",
      "Verify source indices and cell IDs against the provider SingleCellExperiment.",
      "Extract the provider exprs assay and verify exprs equals asinh(counts) with cofactor 1; apply no min-max scaling, z-scoring, or second transform."
    ),
    cycombine_input_scale = "Provider IMC expression assay: asinh mean pixel intensity, cofactor 1",
    transform_formula = "Provider exprs = asinh(counts / 1); no additional transform",
    transform_scope = "Element-wise source-study transform over raw per-cell mean pixel intensities before frozen-cell selection.",
    processing_reference = "Meyer et al. 2025 Methods, Data processing (mean pixel intensities arsinh-transformed with cofactor 1); provider exprs/counts identity verified by build_inputs.R.",
    processing_code_file = "build_inputs.R",
    processing_code_sha256 = script_hash,
    double_transform_checked = TRUE
  ),
  list(
    dataset = "Shiao CODEX",
    technology = "CODEX multiplexed fluorescence imaging",
    input_matrix_file = "data/shiao_codex_analysis_ready.csv.gz",
    input_matrix_sha256 = sha256(output_paths[["Shiao CODEX"]]),
    source_state = "nonlinearly_transformed",
    assay_or_layer = "Frozen arithmetic mean of provider nuclei_mean and membrane_mean fields from TNBC_immune_CODEX_full_panel_counts.csv",
    processing_steps = list(
      "Retain the exact 20,000 frozen cells from five donors and their row order.",
      "Use the benchmark-locked arithmetic mean of each marker's nuclei_mean and membrane_mean fields without antibody-specific compartment reselection.",
      "Apply asinh(x/1) exactly once as a predeclared cyCombine comparator adaptation; apply no marker z-score or data-derived cofactor."
    ),
    cycombine_input_scale = "Cofactor-1 asinh of frozen CODEX segmented-cell compartment averages",
    transform_formula = "asinh(mean(nuclei_mean, membrane_mean) / 1)",
    transform_scope = "Element-wise on all eleven frozen marker averages for every frozen Shiao cell after provider image correction, segmentation, and intensity extraction.",
    processing_reference = "Shiao et al. 2024 Methods documents provider image correction, segmentation, and cell-level channel averages but no nonlinear transform; cofactor-1 asinh is an explicit comparator adaptation matching the two TNBC IMC expression scales, not a claimed provider step.",
    processing_code_file = "build_inputs.R",
    processing_code_sha256 = script_hash,
    double_transform_checked = TRUE
  )
))

preprocessing_manifest <- list(
  schema_version = "final_xu.cycombine_preprocessing.v2",
  combined_input_sha256 = sha256(combined_path),
  shared_marker_policy = "Exact ordered intersection of eleven harmonized protein markers frozen before method fitting; no provider label, outcome, spatial field, or method output enters preprocessing.",
  correction_marker_file = "markers.txt",
  correction_marker_sha256 = sha256(marker_file),
  marker_harmonization_manifest_file = "marker_harmonization_manifest.csv",
  marker_harmonization_manifest_sha256 = sha256(harmonization_file),
  cell_selection_manifest_file = "../../../../benchmark_data/tnbc/selected_cells.csv.gz",
  cell_selection_manifest_sha256 = sha256(selection_file),
  cell_selection_id_column = "cell_id",
  datasets = dataset_entries
)
manifest_path <- file.path(preprocessing_dir, "preprocessing_manifest.json")
jsonlite::write_json(
  preprocessing_manifest,
  manifest_path,
  pretty = TRUE,
  auto_unbox = TRUE,
  null = "null",
  digits = NA
)

shiao_raw_counts <- file.path(
  raw_dir, "Shiao_2024", "TNBC_immune_CODEX_full_panel_counts.csv"
)
tietscher_paper <- file.path(raw_dir, "Tietscher_2023", "Tietscher_2023.pdf")
meyer_paper <- file.path(raw_dir, "Meyer_2025", "Meyer_2025.pdf")
shiao_paper <- file.path(raw_dir, "Shiao_2024", "Shiao_2024.pdf")
source_files <- c(
  tietscher_rds,
  meyer_rds,
  shiao_raw_counts,
  tietscher_paper,
  meyer_paper,
  shiao_paper,
  file.path(benchmark_dir, "tietscher_imc", "alignment_input_11.csv.gz"),
  file.path(benchmark_dir, "meyer_imc", "alignment_input_11.csv.gz"),
  file.path(benchmark_dir, "shiao_codex", "alignment_input_11.csv.gz"),
  selection_file,
  benchmark_marker_file
)
build_manifest <- list(
  schema_version = "final_xu.tnbc.cycombine_input_build.v1",
  method = "cyCombine v0.2.5 canonical comparator",
  labels_read_during_preprocessing = FALSE,
  locked_cells = nrow(combined),
  dataset_order = as.list(dataset_order),
  dataset_cell_counts = as.list(stats::setNames(as.integer(expected_counts), names(expected_counts))),
  markers = as.list(markers),
  provider_expression_checks = list(
    "Tietscher IMC" = tietscher_extraction$audit,
    "Meyer IMC" = meyer_extraction$audit,
    "Shiao CODEX" = list(
      cells = nrow(shiao_ready),
      frozen_compartment_mean_used = TRUE,
      transform = "asinh(x / 1)",
      transform_is_comparator_adaptation = TRUE,
      provider_transform_claimed = FALSE
    )
  ),
  source_files = lapply(source_files, function(path) {
    list(path = normalizePath(path), sha256 = sha256(path))
  }),
  outputs = lapply(
    c(output_paths, combined = combined_path, scale_audit = scale_audit_path),
    function(path) list(path = normalizePath(path), sha256 = sha256(path))
  ),
  preprocessing_manifest_sha256 = sha256(manifest_path),
  script = list(path = script_path, sha256 = script_hash),
  runtime = list(
    R = as.character(getRversion()),
    SingleCellExperiment = as.character(packageVersion("SingleCellExperiment")),
    SummarizedExperiment = as.character(packageVersion("SummarizedExperiment")),
    readr = as.character(packageVersion("readr"))
  )
)
jsonlite::write_json(
  build_manifest,
  file.path(preprocessing_dir, "preprocessing_build_manifest.json"),
  pretty = TRUE,
  auto_unbox = TRUE,
  null = "null",
  digits = NA
)

cat(sprintf(
  "Built TNBC cyCombine input: %s cells x %s markers; output: %s\n",
  format(nrow(combined), big.mark = ","), length(markers), combined_path
))
