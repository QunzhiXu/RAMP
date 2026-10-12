#!/usr/bin/env Rscript

# Build the method-specific, analysis-ready input for the frozen melanoma
# cyCombine benchmark. All three provider matrices are already transformed or
# standardized analysis layers, so this script validates and passes them
# through without adding arcsinh, CLR, shifting, clipping, or rescaling.

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_argument)) stop("Cannot determine script location.", call. = FALSE)
script_path <- sub("^--file=", "", script_argument[[1L]])
script_path <- gsub("~+~", " ", script_path, fixed = TRUE)
script_path <- normalizePath(script_path, mustWork = TRUE)
preprocessing_dir <- dirname(script_path)
method_dir <- dirname(preprocessing_dir)
final_xu <- normalizePath(file.path(method_dir, "..", "..", ".."), mustWork = TRUE)
benchmark_dir <- file.path(final_xu, "benchmark_data", "melanoma")
raw_dir <- file.path(final_xu, "raw_data", "melanoma")
algorithm_dir <- file.path(final_xu, "algorithm", "cycombine")
.libPaths(c(file.path(algorithm_dir, ".Rlib"), .libPaths()))

suppressPackageStartupMessages({
  library(digest)
  library(jsonlite)
  library(readr)
  library(SingleCellExperiment)
})

markers <- c(
  "CD163", "CD3", "CD31", "CD68", "CD8", "HLA_DR",
  "LAG3", "PD1", "PDL1", "SMA", "SOX10", "VIMENTIN"
)
id_columns <- c("cell_id", "dataset", "donor", "unit_key")
dataset_order <- c(
  "MEL_Hickey2023_CODEX", "MEL_Hoch2021_IMC", "MEL_Ptacek2021_MIBI"
)
dataset_slugs <- c(
  "MEL_Hickey2023_CODEX" = "hickey_codex",
  "MEL_Hoch2021_IMC" = "hoch_imc",
  "MEL_Ptacek2021_MIBI" = "ptacek_mibi"
)
expected_counts <- stats::setNames(rep(20000L, length(dataset_order)), dataset_order)
expected_row_starts <- c(
  "MEL_Hickey2023_CODEX" = 0L,
  "MEL_Hoch2021_IMC" = 20000L,
  "MEL_Ptacek2021_MIBI" = 40000L
)
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

write_text_atomic <- function(lines, path) {
  temporary <- file.path(dirname(path), paste0(".", basename(path), ".tmp"))
  if (file.exists(temporary)) unlink(temporary)
  writeLines(lines, temporary, useBytes = TRUE)
  if (!file.rename(temporary, path)) {
    unlink(temporary)
    stop(sprintf("Could not publish %s", path), call. = FALSE)
  }
}

assert_fixed_contract <- function(frame, dataset) {
  expected_columns <- c("row_id", id_columns, markers)
  if (!identical(colnames(frame), expected_columns)) {
    stop(sprintf("%s frozen input columns changed.", dataset), call. = FALSE)
  }
  if (nrow(frame) != expected_counts[[dataset]] || any(frame$dataset != dataset)) {
    stop(sprintf("%s frozen cell count or dataset identity changed.", dataset), call. = FALSE)
  }
  expected_rows <- seq.int(
    expected_row_starts[[dataset]],
    expected_row_starts[[dataset]] + expected_counts[[dataset]] - 1L
  )
  if (!identical(as.integer(frame$row_id), expected_rows)) {
    stop(sprintf("%s row lock changed.", dataset), call. = FALSE)
  }
  if (anyNA(frame$cell_id) || any(!nzchar(frame$cell_id)) || anyDuplicated(frame$cell_id)) {
    stop(sprintf("%s cell IDs are incomplete or duplicated.", dataset), call. = FALSE)
  }
  values <- as.matrix(frame[, markers, drop = FALSE])
  if (!all(is.finite(values))) stop(sprintf("%s contains non-finite values.", dataset), call. = FALSE)
  if (any(apply(values, 2L, stats::sd) < 1e-10)) {
    stop(sprintf("%s contains a constant correction marker.", dataset), call. = FALSE)
  }
}

input_paths <- c(
  "MEL_Hickey2023_CODEX" = file.path(benchmark_dir, "hickey_codex", "alignment_input_12.csv.gz"),
  "MEL_Hoch2021_IMC" = file.path(benchmark_dir, "hoch_imc", "alignment_input_12.csv.gz"),
  "MEL_Ptacek2021_MIBI" = file.path(benchmark_dir, "ptacek_mibi", "alignment_input_12.csv.gz")
)
fixed_tables <- lapply(dataset_order, function(dataset) read_fixed(input_paths[[dataset]]))
names(fixed_tables) <- dataset_order
for (dataset in dataset_order) assert_fixed_contract(fixed_tables[[dataset]], dataset)

selected_path <- file.path(benchmark_dir, "selected_cells.csv.gz")
selected <- suppressMessages(readr::read_csv(
  selected_path,
  col_types = readr::cols(.default = readr::col_character()),
  progress = FALSE,
  show_col_types = FALSE
))
if (nrow(selected) != 60000L || anyDuplicated(selected$cell_id)) {
  stop("Frozen melanoma cell-selection manifest changed.", call. = FALSE)
}
if (!identical(as.integer(selected$row_id), 0:59999)) {
  stop("Frozen melanoma row IDs changed.", call. = FALSE)
}

# The exact source assay relationship is checked for Hoch, where the provider
# retained counts and exprs together. This protects against accidentally using
# counts or scaled_exprs while avoiding any new transform of the method input.
hoch_rds <- file.path(raw_dir, "hoch_imc", "HochSchulz_2022_Melanoma_sce.rds")
hoch_sce <- readRDS(hoch_rds)
if (!all(c("counts", "exprs", "scaled_counts", "scaled_exprs") %in% assayNames(hoch_sce))) {
  stop("Hoch source assay contract changed.", call. = FALSE)
}
hoch_selected <- selected[selected$dataset == "MEL_Hoch2021_IMC", , drop = FALSE]
hoch_indices <- as.integer(hoch_selected$source_index_zero_based) + 1L
hoch_source_markers <- c(
  "CD163", "CD3e", "CD31", "CD68", "CD8a", "HLA_DR",
  "LAG3", "PD_1", "PD_L1", "SMA", "SOX10", "VIM"
)
if (!all(hoch_source_markers %in% rownames(hoch_sce))) {
  stop("Hoch source marker mapping changed.", call. = FALSE)
}
hoch_exprs <- t(as.matrix(assay(hoch_sce, "exprs")[hoch_source_markers, hoch_indices, drop = FALSE]))
hoch_counts <- t(as.matrix(assay(hoch_sce, "counts")[hoch_source_markers, hoch_indices, drop = FALSE]))
hoch_formula_error <- max(abs(hoch_exprs - asinh(hoch_counts)))
hoch_reconstruction_error <- max(abs(
  hoch_exprs - as.matrix(fixed_tables[["MEL_Hoch2021_IMC"]][, markers, drop = FALSE])
))
hoch_source_ids <- paste0("hoch::", as.character(colData(hoch_sce)$cell_id[hoch_indices]))
if (hoch_formula_error > 1e-12 || hoch_reconstruction_error > 1e-6 ||
    !identical(hoch_source_ids, hoch_selected$cell_id)) {
  stop("Hoch exprs/source identity audit failed.", call. = FALSE)
}
rm(hoch_counts, hoch_exprs, hoch_sce)
gc(verbose = FALSE)

# Ptacek is small enough to reconstruct directly from the distributed processed
# table. Values are already on the provider analysis scale and remain unchanged.
ptacek_source_path <- file.path(raw_dir, "ptacek_mibi", "scaled_mibi_data_w_cell_gating.csv")
ptacek_source_markers <- c(
  "CD163", "CD3", "CD31", "CD68", "CD8", "HLA DPDQDR",
  "LAG3", "PD-1", "PD-L1", "SMA", "SOX10", "Vimentin"
)
ptacek_source <- suppressMessages(readr::read_csv(
  ptacek_source_path,
  col_types = readr::cols(.default = readr::col_double(), fov = readr::col_character(), label = readr::col_character()),
  col_select = c("fov", "label", tidyselect::all_of(ptacek_source_markers)),
  progress = FALSE,
  show_col_types = FALSE,
  name_repair = "minimal"
))
ptacek_selected <- selected[selected$dataset == "MEL_Ptacek2021_MIBI", , drop = FALSE]
ptacek_indices <- as.integer(ptacek_selected$source_index_zero_based) + 1L
ptacek_rows <- ptacek_source[ptacek_indices, , drop = FALSE]
ptacek_source_ids <- paste0(ptacek_rows$fov, "::", ptacek_rows$label)
ptacek_source_ids <- paste0("ptacek::", ptacek_source_ids)
ptacek_reconstruction_error <- max(abs(
  as.matrix(ptacek_rows[, ptacek_source_markers, drop = FALSE]) -
    as.matrix(fixed_tables[["MEL_Ptacek2021_MIBI"]][, markers, drop = FALSE])
))
if (!identical(ptacek_source_ids, ptacek_selected$cell_id) || ptacek_reconstruction_error > 1e-5) {
  stop("Ptacek processed-source identity/value audit failed.", call. = FALSE)
}
rm(ptacek_rows, ptacek_source)
gc(verbose = FALSE)

ready_tables <- lapply(dataset_order, function(dataset) {
  as.data.frame(fixed_tables[[dataset]][, c(id_columns, markers), drop = FALSE], check.names = FALSE)
})
names(ready_tables) <- dataset_order
combined <- do.call(rbind, ready_tables)
rownames(combined) <- NULL
if (nrow(combined) != 60000L || anyDuplicated(combined$cell_id)) {
  stop("Combined cyCombine input is not the frozen 60,000-cell benchmark.", call. = FALSE)
}
if (!identical(unique(combined$dataset), dataset_order)) {
  stop("Combined dataset block order changed.", call. = FALSE)
}
for (column in id_columns) {
  if (!identical(as.character(selected[[column]]), as.character(combined[[column]]))) {
    stop(sprintf("Combined %s/order differs from selected_cells.csv.gz.", column), call. = FALSE)
  }
}

output_paths <- c(
  "MEL_Hickey2023_CODEX" = file.path(data_dir, "hickey_codex_analysis_ready.csv.gz"),
  "MEL_Hoch2021_IMC" = file.path(data_dir, "hoch_imc_analysis_ready.csv.gz"),
  "MEL_Ptacek2021_MIBI" = file.path(data_dir, "ptacek_mibi_analysis_ready.csv.gz")
)
for (dataset in dataset_order) write_csv_gz_atomic(ready_tables[[dataset]], output_paths[[dataset]])
combined_path <- file.path(data_dir, "combined_analysis_ready.csv.gz")
write_csv_gz_atomic(combined, combined_path)

marker_file <- file.path(preprocessing_dir, "markers.txt")
write_text_atomic(markers, marker_file)
harmonization_source <- file.path(benchmark_dir, "marker_manifest.csv")
harmonization_file <- file.path(preprocessing_dir, "marker_harmonization_manifest.csv")
if (!file.copy(harmonization_source, harmonization_file, overwrite = TRUE, copy.mode = TRUE)) {
  stop("Could not copy the frozen marker harmonization manifest.", call. = FALSE)
}
if (!identical(sha256(harmonization_source), sha256(harmonization_file))) {
  stop("Copied marker harmonization manifest differs from benchmark_data.", call. = FALSE)
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
      negative_fraction = mean(values < 0),
      zero_fraction = mean(values == 0),
      stringsAsFactors = FALSE
    )
  }))
})
scale_audit_path <- file.path(preprocessing_dir, "input_scale_audit.csv")
readr::write_csv(do.call(rbind, audit_rows), scale_audit_path)

script_hash <- sha256(script_path)
dataset_entries <- unname(list(
  list(
    dataset = "MEL_Hickey2023_CODEX",
    technology = "CODEX multiplexed fluorescence imaging",
    input_matrix_file = "data/hickey_codex_analysis_ready.csv.gz",
    input_matrix_sha256 = sha256(output_paths[["MEL_Hickey2023_CODEX"]]),
    source_state = "standardized",
    assay_or_layer = "Dryad human melanoma per-cell area-averaged fluorescence table after marker-column z-normalization within tissue",
    processing_steps = list(
      "Retain the exact 20,000 frozen pretreatment cells, dataset identity, and row order.",
      "Use the provider-distributed tissue-wise marker-z-normalized CODEX measurements for the 12 shared markers.",
      "Map provider SOX10/S100 to the documented canonical SOX10 proxy and apply no second arcsinh, CLR, shift, clipping, or scaling."
    ),
    cycombine_input_scale = "Provider tissue-wise marker-column z-normalized CODEX fluorescence",
    transform_formula = "Provider z = (area-averaged fluorescence - tissue-marker mean) / tissue-marker SD; no additional transform",
    transform_scope = "Applied by the provider separately within tissue and marker column before frozen-cell selection.",
    processing_reference = "Hickey et al. 2023 CODEX Methods and Dryad DOI 10.5061/dryad.k0p2ngfcc README.",
    processing_code_file = "build_inputs.R",
    processing_code_sha256 = script_hash,
    double_transform_checked = TRUE
  ),
  list(
    dataset = "MEL_Hoch2021_IMC",
    technology = "Imaging mass cytometry",
    input_matrix_file = "data/hoch_imc_analysis_ready.csv.gz",
    input_matrix_sha256 = sha256(output_paths[["MEL_Hoch2021_IMC"]]),
    source_state = "normalized_and_nonlinearly_transformed",
    assay_or_layer = "HochSchulz_2022_Melanoma_sce.rds exprs assay / exported exprs cell-by-marker table",
    processing_steps = list(
      "Retain the exact 20,000 frozen Stage III treatment-naive cells and row order.",
      "Use provider single-cell mean marker measurements after non-negative least-squares channel spillover correction.",
      "Select the provider exprs assay, verified as exactly asinh(counts/1), and apply no second nonlinear transform or scaled_exprs standardization."
    ),
    cycombine_input_scale = "Provider IMC exprs assay: asinh spillover-corrected mean intensity with cofactor 1",
    transform_formula = "Provider exprs = asinh(counts / 1); no additional transform",
    transform_scope = "Applied by the provider to the complete SingleCellExperiment before frozen-cell selection.",
    processing_reference = "Hoch et al. 2021/2022 Methods and provider SingleCellExperiment counts/exprs assays.",
    processing_code_file = "build_inputs.R",
    processing_code_sha256 = script_hash,
    double_transform_checked = TRUE
  ),
  list(
    dataset = "MEL_Ptacek2021_MIBI",
    technology = "Multiplexed ion beam imaging",
    input_matrix_file = "data/ptacek_mibi_analysis_ready.csv.gz",
    input_matrix_sha256 = sha256(output_paths[["MEL_Ptacek2021_MIBI"]]),
    source_state = "provider_analysis_scale",
    assay_or_layer = "Mendeley Data V2 scaled_mibi_data_w_cell_gating.csv processed protein measurements",
    processing_steps = list(
      "Retain the exact 20,000 frozen gross Stage III cells and row order.",
      "Use the provider-distributed processed and already arcsinh-transformed MIBI protein measurements.",
      "Because the provider cofactor is not reported in the retained record, preserve the distributed analysis scale without inversion, guessed cofactor, second arcsinh, CLR, or shift."
    ),
    cycombine_input_scale = "Provider processed MIBI analysis scale, documented as arcsinh-transformed; cofactor not reported",
    transform_formula = "Provider arcsinh transformation with unreported cofactor; distributed values passed through without additional transform",
    transform_scope = "Applied by the provider before release of the complete 54-patient cell table and before frozen-cell selection.",
    processing_reference = "Mendeley Data V2 DOI 10.17632/79y7bht7tf.2 and related Ptacek et al. 2021 abstract.",
    processing_code_file = "build_inputs.R",
    processing_code_sha256 = script_hash,
    double_transform_checked = TRUE
  )
))

preprocessing_manifest <- list(
  schema_version = "final_xu.cycombine_preprocessing.v2",
  combined_input_sha256 = sha256(combined_path),
  shared_marker_policy = "Exact ordered intersection of 12 harmonized protein measurements frozen before method fitting; Hickey SOX10/S100 is a documented SOX10 proxy.",
  correction_marker_file = "markers.txt",
  correction_marker_sha256 = sha256(marker_file),
  marker_harmonization_manifest_file = "marker_harmonization_manifest.csv",
  marker_harmonization_manifest_sha256 = sha256(harmonization_file),
  cell_selection_manifest_file = "../../../../benchmark_data/melanoma/selected_cells.csv.gz",
  cell_selection_manifest_sha256 = sha256(selected_path),
  cell_selection_id_column = "cell_id",
  datasets = dataset_entries
)
manifest_path <- file.path(preprocessing_dir, "preprocessing_manifest.json")
jsonlite::write_json(
  preprocessing_manifest, manifest_path, pretty = TRUE, auto_unbox = TRUE,
  null = "null", digits = NA
)

hickey_source <- file.path(raw_dir, "hickey_codex", "23_10_11_Melanoma_Marker_Cell_Neighborhood.csv")
hoch_export <- file.path(raw_dir, "hoch_imc", "HochSchulz_2022_Melanoma_exprs_cells_by_markers.csv")
source_files <- c(
  input_paths,
  selected_path,
  file.path(benchmark_dir, "input", "raw.npy"),
  harmonization_source,
  hickey_source,
  file.path(raw_dir, "hickey_codex", "DRYAD_README.md"),
  file.path(raw_dir, "hickey_codex", "Hickey_2023.pdf"),
  hoch_export,
  hoch_rds,
  file.path(raw_dir, "hoch_imc", "Hoch_2021.pdf"),
  ptacek_source_path,
  file.path(raw_dir, "ptacek_mibi", "SOURCE_MANIFEST.md"),
  file.path(raw_dir, "ptacek_mibi", "Ptacek_2021.pdf")
)
build_manifest <- list(
  schema_version = "final_xu.melanoma.cycombine_input_build.v1",
  method = "cyCombine v0.2.5 canonical comparator",
  labels_outcomes_or_coordinates_read_during_preprocessing = FALSE,
  expression_operation = "provider analysis-ready passthrough for all three datasets",
  second_arcsinh_clr_shift_clip_or_scale_applied = FALSE,
  locked_cells = nrow(combined),
  dataset_order = as.list(dataset_order),
  dataset_cell_counts = as.list(stats::setNames(as.integer(expected_counts), names(expected_counts))),
  markers = as.list(markers),
  source_validation = list(
    hoch_exprs_equals_asinh_counts_maximum_error = hoch_formula_error,
    hoch_frozen_reconstruction_maximum_error = hoch_reconstruction_error,
    ptacek_frozen_reconstruction_maximum_error = ptacek_reconstruction_error,
    hickey_source_value_reextraction = FALSE,
    hickey_source_value_reextraction_reason = "The 5-GB provider table is already frozen and hash-locked by benchmark_data; cyCombine uses the exact canonical provider-scale subset without modifying it."
  ),
  limitations = list(
    ptacek_arcsinh_cofactor = "not reported by the provider in the retained V2 record",
    hickey_sox10 = "provider combined SOX10/S100 proxy, not SOX10-only"
  ),
  source_files = lapply(source_files, function(path) list(path = normalizePath(path), sha256 = sha256(path))),
  outputs = lapply(c(output_paths, combined = combined_path, scale_audit = scale_audit_path), function(path) {
    list(path = normalizePath(path), sha256 = sha256(path))
  }),
  preprocessing_manifest_sha256 = sha256(manifest_path),
  script = list(path = script_path, sha256 = script_hash),
  runtime = list(
    R = as.character(getRversion()),
    SingleCellExperiment = as.character(packageVersion("SingleCellExperiment")),
    readr = as.character(packageVersion("readr"))
  )
)
jsonlite::write_json(
  build_manifest, file.path(preprocessing_dir, "preprocessing_build_manifest.json"),
  pretty = TRUE, auto_unbox = TRUE, null = "null", digits = NA
)

cat(sprintf(
  "Built melanoma cyCombine passthrough input: %s cells x %s markers; output: %s\n",
  format(nrow(combined), big.mark = ","), length(markers), combined_path
))
