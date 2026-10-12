#!/usr/bin/env Rscript

# Build the method-specific, analysis-ready input for the frozen bone-marrow
# cyCombine benchmark. This script reads only benchmark cell identities/order,
# label-free marker matrices, and provider source matrices. It never reads cell
# type annotations or benchmark results.

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_argument)) stop("Cannot determine script location.", call. = FALSE)
script_path <- sub("^--file=", "", script_argument[[1L]])
script_path <- gsub("~+~", " ", script_path, fixed = TRUE)
script_path <- normalizePath(script_path, mustWork = TRUE)
preprocessing_dir <- dirname(script_path)
method_dir <- dirname(preprocessing_dir)
final_xu <- normalizePath(file.path(method_dir, "..", "..", ".."), mustWork = TRUE)
benchmark_dir <- file.path(final_xu, "benchmark_data", "bone_marrow")
raw_dir <- file.path(final_xu, "raw_data", "bone_marrow")
algorithm_dir <- file.path(final_xu, "algorithm", "cycombine")
.libPaths(c(file.path(algorithm_dir, ".Rlib"), .libPaths()))

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
expected_counts <- c("BM CODEX" = 30000L, "BM CyTOF" = 30000L, "Triana AbSeq" = 28031L)
data_dir <- file.path(preprocessing_dir, "data")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

sha256 <- function(path) {
  digest::digest(file = normalizePath(path, mustWork = TRUE), algo = "sha256", serialize = FALSE)
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
  expected_columns <- c(id_columns, markers)
  if (!identical(colnames(frame), expected_columns)) {
    stop(sprintf("%s frozen input columns changed.", dataset), call. = FALSE)
  }
  if (nrow(frame) != expected_counts[[dataset]] || any(frame$dataset != dataset)) {
    stop(sprintf("%s frozen cell count or dataset identity changed.", dataset), call. = FALSE)
  }
  if (anyNA(frame$cell_id) || any(!nzchar(frame$cell_id)) || anyDuplicated(frame$cell_id)) {
    stop(sprintf("%s cell IDs are incomplete or duplicated.", dataset), call. = FALSE)
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

# Freeze cell identities and ordering from benchmark_data. No labels are read.
codex_frozen <- read_fixed(file.path(benchmark_dir, "bm_codex", "alignment_input_14.csv.gz"))
cytof_frozen <- read_fixed(file.path(benchmark_dir, "bm_cytof", "alignment_input_14.csv.gz"))
triana_frozen <- read_fixed(file.path(benchmark_dir, "triana_abseq", "alignment_input_14.csv.gz"))
assert_fixed_contract(codex_frozen, "BM CODEX")
assert_fixed_contract(cytof_frozen, "BM CyTOF")
assert_fixed_contract(triana_frozen, "Triana AbSeq")

# BM CODEX: extract the source-study per-sample CLR layer. CD3e is renamed CD3.
codex_rds <- file.path(raw_dir, "BM_Codex", "Normal_Bone_Marrow_CODEX_Atlas_Seurat_v2.rds")
codex_object <- readRDS(codex_rds)
if (!("CODEX" %in% names(codex_object@assays))) stop("CODEX assay is absent.", call. = FALSE)
codex_layer <- codex_object[["CODEX"]]@data
codex_source_markers <- c(
  "CD117", "CD11B", "CD11C", "CD123", "CD14", "CD3e", "CD33",
  "CD34", "CD38", "CD4", "CD45", "CD45RA", "CD61", "CD8"
)
if (!all(codex_source_markers %in% rownames(codex_layer))) {
  stop("A required CODEX source marker is absent.", call. = FALSE)
}
codex_cell_index <- match(codex_frozen$cell_id, colnames(codex_layer))
if (anyNA(codex_cell_index)) stop("Frozen CODEX cells are absent from the source RDS.", call. = FALSE)
codex_values <- t(as.matrix(codex_layer[codex_source_markers, codex_cell_index, drop = FALSE]))
colnames(codex_values) <- markers
codex_ready <- replace_expression(codex_frozen, codex_values)
rm(codex_values, codex_layer, codex_object)
gc(verbose = FALSE)

# BM CyTOF: values are CCT-subtracted/bead-normalized linear intensities.
# Apply the source-paper and official cyCombine CyTOF transform exactly once.
cytof_values <- as.matrix(cytof_frozen[, markers, drop = FALSE])
cytof_ready <- replace_expression(cytof_frozen, asinh(cytof_values / 5))
rm(cytof_values)

# Triana AbSeq: reproduce the official cyCombine CITE-seq input convention.
# CLR margin=2 is calculated per cell over all 97 antibody features, before
# selecting the 14 shared markers. It is therefore not recomputed on a 14-marker
# subset and does not use AB@data, integrated, or scale.data.
triana_export <- file.path(raw_dir, "Triana_2021", "exported")
triana_features <- readLines(file.path(triana_export, "ab_features.tsv"), warn = FALSE)
triana_barcodes <- readLines(file.path(triana_export, "ab_barcodes.tsv"), warn = FALSE)
triana_counts <- Matrix::readMM(file.path(triana_export, "ab_counts.mtx"))
if (nrow(triana_counts) == length(triana_barcodes) && ncol(triana_counts) == length(triana_features)) {
  triana_counts <- Matrix::t(triana_counts)
} else if (!(nrow(triana_counts) == length(triana_features) && ncol(triana_counts) == length(triana_barcodes))) {
  stop("Triana count matrix dimensions do not match features/barcodes.", call. = FALSE)
}
triana_cell_index <- match(triana_frozen$cell_id, triana_barcodes)
if (anyNA(triana_cell_index)) stop("Frozen Triana cells are absent from raw AB counts.", call. = FALSE)
triana_source_markers <- c(
  "CD117-AB", "CD11b-AB", "CD11c-AB", "CD123-AB", "CD14-AB", "CD3-AB", "CD33-AB",
  "CD34-AB", "CD38-AB", "CD4-AB", "CD45-AB", "CD45RA-AB", "CD61-AB", "CD8-AB"
)
triana_marker_index <- match(triana_source_markers, triana_features)
if (anyNA(triana_marker_index)) stop("A required Triana source marker is absent.", call. = FALSE)
triana_all_counts <- as.matrix(triana_counts[, triana_cell_index, drop = FALSE])
clr_denominator <- exp(colSums(log1p(triana_all_counts)) / nrow(triana_all_counts))
triana_clr_all <- log1p(sweep(triana_all_counts, 2L, clr_denominator, FUN = "/"))
triana_values <- t(triana_clr_all[triana_marker_index, , drop = FALSE])
colnames(triana_values) <- markers
triana_ready <- replace_expression(triana_frozen, triana_values)
rm(triana_counts, triana_all_counts, triana_clr_all, triana_values)
gc(verbose = FALSE)

ready_tables <- list(codex_ready, cytof_ready, triana_ready)
names(ready_tables) <- dataset_order
combined <- do.call(rbind, ready_tables)
rownames(combined) <- NULL
if (nrow(combined) != 88031L || anyDuplicated(combined$cell_id)) {
  stop("Combined cyCombine input is not the frozen 88,031-cell benchmark.", call. = FALSE)
}
if (!identical(unique(combined$dataset), dataset_order)) {
  stop("Combined dataset order changed.", call. = FALSE)
}
selected <- suppressMessages(readr::read_csv(
  file.path(benchmark_dir, "selected_cells.csv.gz"),
  col_types = readr::cols(.default = readr::col_character()),
  progress = FALSE,
  show_col_types = FALSE
))
if (!identical(as.character(selected$cell_id), combined$cell_id) ||
    !identical(as.character(selected$dataset), combined$dataset) ||
    !identical(as.character(selected$donor), combined$donor)) {
  stop("Combined identities/order differ from selected_cells.csv.gz.", call. = FALSE)
}

output_paths <- c(
  "BM CODEX" = file.path(data_dir, "bm_codex_analysis_ready.csv.gz"),
  "BM CyTOF" = file.path(data_dir, "bm_cytof_analysis_ready.csv.gz"),
  "Triana AbSeq" = file.path(data_dir, "triana_abseq_analysis_ready.csv.gz")
)
for (dataset in dataset_order) write_csv_gz_atomic(ready_tables[[dataset]], output_paths[[dataset]])
combined_path <- file.path(data_dir, "combined_analysis_ready.csv.gz")
write_csv_gz_atomic(combined, combined_path)

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
      stringsAsFactors = FALSE
    )
  }))
})
scale_audit_path <- file.path(preprocessing_dir, "input_scale_audit.csv")
readr::write_csv(do.call(rbind, audit_rows), scale_audit_path)

marker_file <- file.path(preprocessing_dir, "markers.txt")
harmonization_file <- file.path(preprocessing_dir, "marker_harmonization_manifest.csv")
selection_file <- file.path(benchmark_dir, "selected_cells.csv.gz")
if (!identical(readLines(marker_file, warn = FALSE), markers)) stop("markers.txt drifted.", call. = FALSE)
script_hash <- sha256(script_path)

dataset_entries <- unname(list(
  list(
    dataset = "BM CODEX",
    technology = "CODEX multiplexed fluorescence imaging",
    input_matrix_file = "data/bm_codex_analysis_ready.csv.gz",
    input_matrix_sha256 = sha256(output_paths[["BM CODEX"]]),
    source_state = "normalized_and_nonlinearly_transformed",
    assay_or_layer = "Normal_Bone_Marrow_CODEX_Atlas_Seurat_v2.rds: CODEX assay data slot",
    processing_steps = list(
      "Retain the exact 30,000 frozen benchmark cell IDs and row order.",
      "Extract the provider's per-sample Seurat CLR-normalized CODEX data layer for the 14 shared markers.",
      "Rename source CD3e to canonical CD3; apply no second CLR, arcsinh, scaling, or integration."
    ),
    cycombine_input_scale = "Provider per-sample Seurat CLR expression (margin=1)",
    transform_formula = "Provider NormalizeData(normalization.method='CLR', margin=1); no additional transform",
    transform_scope = "Computed by the source study within each complete CODEX sample before frozen-cell selection.",
    processing_reference = "Shovik et al. 2024 Methods (sample-internal Seurat CLR, margin=1) and the provider CODEX data slot.",
    processing_code_file = "build_inputs.R",
    processing_code_sha256 = script_hash,
    double_transform_checked = TRUE
  ),
  list(
    dataset = "BM CyTOF",
    technology = "CyTOF",
    input_matrix_file = "data/bm_cytof_analysis_ready.csv.gz",
    input_matrix_sha256 = sha256(output_paths[["BM CyTOF"]]),
    source_state = "technically_normalized_untransformed",
    assay_or_layer = "Frozen cct_subtracted_normalized annotated FCS event values",
    processing_steps = list(
      "Retain the exact 30,000 frozen benchmark events and row order.",
      "Preserve continuous CCT-subtracted/bead-normalized intensities, including small negative background values.",
      "Apply asinh(x/5) exactly once; do not derandomize, z-score, or use a data-derived cofactor."
    ),
    cycombine_input_scale = "asinh protein intensity with cofactor 5",
    transform_formula = "asinh(x / 5)",
    transform_scope = "Applied element-wise to every one of the 14 markers in every frozen CyTOF event.",
    processing_reference = "Levine et al. 2015 Methods (bead normalization followed by asinh cofactor 5) and official cyCombine cross-platform vignette.",
    processing_code_file = "build_inputs.R",
    processing_code_sha256 = script_hash,
    double_transform_checked = TRUE
  ),
  list(
    dataset = "Triana AbSeq",
    technology = "AbSeq antibody-derived tags",
    input_matrix_file = "data/triana_abseq_analysis_ready.csv.gz",
    input_matrix_sha256 = sha256(output_paths[["Triana AbSeq"]]),
    source_state = "normalized_and_nonlinearly_transformed",
    assay_or_layer = "Raw AB UMI counts from all 97 antibody features, converted to Seurat-style CLR margin=2",
    processing_steps = list(
      "Retain the exact 28,031 frozen BM1/BM2/BM3 cells and row order.",
      "Compute Seurat-style CLR margin=2 within each cell using all 97 raw antibody features.",
      "Subset the already normalized matrix to the 14 shared markers; do not use AB scale.data or integrated values."
    ),
    cycombine_input_scale = "Seurat-style CLR-normalized antibody UMI counts (margin=2)",
    transform_formula = "log1p(x / exp(sum(log1p(x_all_97_features)) / 97)) within each cell",
    transform_scope = "Calculated independently for each frozen cell over all 97 antibody features before selecting shared markers.",
    processing_reference = "Official cyCombine CITE-seq/spectral-flow/CyTOF vignette (Seurat CLR margin=2); Triana et al. 2021 raw antibody UMI counts.",
    processing_code_file = "build_inputs.R",
    processing_code_sha256 = script_hash,
    double_transform_checked = TRUE
  )
))

preprocessing_manifest <- list(
  schema_version = "final_xu.cycombine_preprocessing.v2",
  combined_input_sha256 = sha256(combined_path),
  shared_marker_policy = "Exact ordered intersection of 14 harmonized protein markers frozen before method fitting; CD19 is held out and excluded.",
  correction_marker_file = "markers.txt",
  correction_marker_sha256 = sha256(marker_file),
  marker_harmonization_manifest_file = "marker_harmonization_manifest.csv",
  marker_harmonization_manifest_sha256 = sha256(harmonization_file),
  cell_selection_manifest_file = "../../../../benchmark_data/bone_marrow/selected_cells.csv.gz",
  cell_selection_manifest_sha256 = sha256(selection_file),
  cell_selection_id_column = "cell_id",
  datasets = dataset_entries
)
manifest_path <- file.path(preprocessing_dir, "preprocessing_manifest.json")
jsonlite::write_json(
  preprocessing_manifest, manifest_path, pretty = TRUE, auto_unbox = TRUE,
  null = "null", digits = NA
)

source_files <- c(
  codex_rds,
  file.path(benchmark_dir, "bm_codex", "alignment_input_14.csv.gz"),
  file.path(benchmark_dir, "bm_cytof", "alignment_input_14.csv.gz"),
  file.path(benchmark_dir, "triana_abseq", "alignment_input_14.csv.gz"),
  file.path(triana_export, "ab_counts.mtx"),
  file.path(triana_export, "ab_features.tsv"),
  file.path(triana_export, "ab_barcodes.tsv"),
  selection_file
)
build_manifest <- list(
  schema_version = "final_xu.bone_marrow.cycombine_input_build.v1",
  method = "cyCombine v0.2.5 canonical comparator",
  labels_read_during_preprocessing = FALSE,
  locked_cells = nrow(combined),
  dataset_order = as.list(dataset_order),
  dataset_cell_counts = as.list(stats::setNames(as.integer(expected_counts), names(expected_counts))),
  markers = as.list(markers),
  cd19_in_fit = FALSE,
  source_files = lapply(source_files, function(path) list(path = normalizePath(path), sha256 = sha256(path))),
  outputs = lapply(c(output_paths, combined = combined_path, scale_audit = scale_audit_path), function(path) {
    list(path = normalizePath(path), sha256 = sha256(path))
  }),
  preprocessing_manifest_sha256 = sha256(manifest_path),
  script = list(path = script_path, sha256 = script_hash),
  runtime = list(R = as.character(getRversion()), Matrix = as.character(packageVersion("Matrix")), SeuratObject = as.character(packageVersion("SeuratObject")))
)
jsonlite::write_json(
  build_manifest, file.path(preprocessing_dir, "preprocessing_build_manifest.json"),
  pretty = TRUE, auto_unbox = TRUE, null = "null", digits = NA
)

cat(sprintf(
  "Built bone-marrow cyCombine input: %s cells x %s markers; output: %s\n",
  format(nrow(combined), big.mark = ","), length(markers), combined_path
))
