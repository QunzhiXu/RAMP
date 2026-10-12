#!/usr/bin/env Rscript

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_argument)) stop("Cannot determine script location.", call. = FALSE)
script_path <- sub("^--file=", "", script_argument[[1L]])
script_path <- gsub("~+~", " ", script_path, fixed = TRUE)
script_path <- normalizePath(script_path, mustWork = TRUE)
preprocessing_dir <- dirname(script_path)
method_root <- dirname(preprocessing_dir)
final_xu <- normalizePath(file.path(method_root, "..", "..", ".."), mustWork = TRUE)

required_packages <- c("digest", "jsonlite", "readr")
for (package in required_packages) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stop(sprintf("Missing required R package: %s", package), call. = FALSE)
  }
}

stopf <- function(...) stop(sprintf(...), call. = FALSE)
assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}
sha256_file <- function(path) {
  digest::digest(file = path, algo = "sha256", serialize = FALSE)
}
write_json_atomic <- function(value, path) {
  temporary <- paste0(path, ".tmp-", Sys.getpid())
  jsonlite::write_json(
    value, temporary, pretty = TRUE, auto_unbox = TRUE,
    null = "null", digits = 17
  )
  if (!file.rename(temporary, path)) stopf("Could not publish %s", path)
}
write_csv_atomic <- function(frame, path) {
  temporary <- paste0(path, ".tmp-", Sys.getpid(), ".csv.gz")
  readr::write_csv(frame, temporary, na = "")
  if (!file.rename(temporary, path)) stopf("Could not publish %s", path)
}
write_plain_csv_atomic <- function(frame, path) {
  temporary <- paste0(path, ".tmp-", Sys.getpid(), ".csv")
  readr::write_csv(frame, temporary, na = "")
  if (!file.rename(temporary, path)) stopf("Could not publish %s", path)
}
copy_atomic <- function(source, target) {
  temporary <- paste0(target, ".tmp-", Sys.getpid())
  if (!file.copy(source, temporary, overwrite = TRUE, copy.mode = TRUE)) {
    stopf("Could not stage %s", source)
  }
  if (!file.rename(temporary, target)) stopf("Could not publish %s", target)
}

markers <- c(
  "CD163", "CD3", "CD31", "CD68", "CD8", "HLA_DR",
  "LAG3", "PD1", "PDL1", "SMA", "SOX10", "VIMENTIN"
)
dataset_names <- c(
  "MEL_Hickey2023_CODEX",
  "MEL_Hoch2021_IMC",
  "MEL_Ptacek2021_MIBI"
)
dataset_slugs <- c("hickey_codex", "hoch_imc", "ptacek_mibi")
expected_cells <- setNames(rep(20000L, 3L), dataset_names)

benchmark_root <- file.path(final_xu, "benchmark_data", "melanoma")
benchmark_manifest_path <- file.path(benchmark_root, "manifest.json")
canonical_raw_path <- file.path(benchmark_root, "input", "raw.npy")
canonical_marker_path <- file.path(benchmark_root, "input", "markers.txt")
fit_metadata_path <- file.path(benchmark_root, "input", "fit_metadata.csv.gz")
selected_cells_path <- file.path(benchmark_root, "selected_cells.csv.gz")
marker_manifest_path <- file.path(benchmark_root, "marker_manifest.csv")
source_paths <- c(
  file.path(benchmark_root, "hickey_codex", "alignment_input_12.csv.gz"),
  file.path(benchmark_root, "hoch_imc", "alignment_input_12.csv.gz"),
  file.path(benchmark_root, "ptacek_mibi", "alignment_input_12.csv.gz")
)

expected_hashes <- c(
  benchmark_manifest = "a52f3c1fd3ece3b8637bb2b48e628bc915dd2cf21709ec073fc83b42b5fba013",
  canonical_raw = "eedadfd47dfe09daf1760ec96cb0d8bd5eb0483d8ab0a5274fc108450877a9da",
  canonical_markers = "9314e9a8dfe93fa305cda18ddd0ad6e63a1c4c22693106f2c7ec5cef43d1b5ec",
  fit_metadata = "2f53f0d31bdc32928b91ba6371bc12c14854bdcf504575d90be2f2e7aef8713b",
  selected_cells = "a707e3f4d8d581ca0779b296a3c5916651fc0fbec56bf8d12324de67c5a9fc2c",
  marker_manifest = "1e51cd35c05d29ae7b227bdcdb1a204df43704eafaeae55db46ec1532ee2d869",
  hickey_input = "70ac486c77a1e33e42580879cfad7ca84228fa21c01e30dd7000e70f76c8e7c4",
  hoch_input = "8da251ccf84d358cfb80fa539114976d346e1314ed0c3dccb0daaae12a313d13",
  ptacek_input = "0cd8d3504cee7bdabb9dde0fe3f0ab68086bf9d9505348fdc75dbb472760ba53"
)
observed_hashes <- c(
  benchmark_manifest = sha256_file(benchmark_manifest_path),
  canonical_raw = sha256_file(canonical_raw_path),
  canonical_markers = sha256_file(canonical_marker_path),
  fit_metadata = sha256_file(fit_metadata_path),
  selected_cells = sha256_file(selected_cells_path),
  marker_manifest = sha256_file(marker_manifest_path),
  hickey_input = sha256_file(source_paths[[1L]]),
  hoch_input = sha256_file(source_paths[[2L]]),
  ptacek_input = sha256_file(source_paths[[3L]])
)
if (!identical(observed_hashes, expected_hashes)) {
  changed <- names(observed_hashes)[observed_hashes != expected_hashes]
  stopf("Frozen melanoma source hash changed: %s", paste(changed, collapse = ", "))
}

benchmark_manifest <- jsonlite::read_json(
  benchmark_manifest_path, simplifyVector = FALSE
)
assert_true(
  identical(benchmark_manifest$package, "melanoma_three_platform_benchmark_v2_60000"),
  "Unexpected melanoma benchmark package."
)
assert_true(
  identical(unlist(benchmark_manifest$dataset_order, use.names = FALSE), dataset_names),
  "Canonical melanoma dataset order changed."
)

marker_file_values <- trimws(readLines(canonical_marker_path, warn = FALSE))
marker_file_values <- marker_file_values[nzchar(marker_file_values)]
assert_true(identical(marker_file_values, markers), "Canonical marker order changed.")

source_column_types <- readr::cols(
  row_id = readr::col_integer(),
  cell_id = readr::col_character(),
  dataset = readr::col_character(),
  donor = readr::col_character(),
  unit_key = readr::col_character(),
  .default = readr::col_double()
)
source_frames <- vector("list", length(source_paths))
for (index in seq_along(source_paths)) {
  frame <- suppressMessages(readr::read_csv(
    source_paths[[index]], col_types = source_column_types,
    show_col_types = FALSE, progress = FALSE, name_repair = "minimal"
  ))
  expected_columns <- c("row_id", "cell_id", "dataset", "donor", "unit_key", markers)
  assert_true(
    identical(colnames(frame), expected_columns),
    sprintf("%s input columns changed.", dataset_names[[index]])
  )
  assert_true(nrow(frame) == expected_cells[[index]], "Dataset cell count changed.")
  assert_true(
    identical(unique(frame$dataset), dataset_names[[index]]),
    "Dataset identity/order changed."
  )
  marker_matrix <- as.matrix(frame[, markers, drop = FALSE])
  storage.mode(marker_matrix) <- "double"
  assert_true(all(is.finite(marker_matrix)), "Marker input contains NA/NaN/Inf.")
  source_frames[[index]] <- as.data.frame(frame, stringsAsFactors = FALSE)
}
combined_with_row <- do.call(rbind, source_frames)
rownames(combined_with_row) <- NULL
assert_true(
  identical(combined_with_row$row_id, 0:59999),
  "Combined melanoma row_id is not the frozen 0:59999 order."
)
assert_true(!anyDuplicated(combined_with_row$cell_id), "Cell IDs are not globally unique.")

selected_types <- readr::cols(
  row_id = readr::col_integer(),
  cell_id = readr::col_character(),
  source_cell_id = readr::col_character(),
  source_index_zero_based = readr::col_double(),
  dataset = readr::col_character(),
  technology = readr::col_character(),
  donor = readr::col_character(),
  patient_id = readr::col_character(),
  specimen_id = readr::col_character(),
  acquisition_id = readr::col_character(),
  unit_key = readr::col_character()
)
selected <- suppressMessages(readr::read_csv(
  selected_cells_path, col_types = selected_types,
  show_col_types = FALSE, progress = FALSE, name_repair = "minimal"
))
assert_true(nrow(selected) == 60000L, "Selected-cell manifest row count changed.")
for (column in c("row_id", "cell_id", "dataset", "donor", "unit_key")) {
  left <- as.character(combined_with_row[[column]])
  right <- as.character(selected[[column]])
  assert_true(identical(left, right), sprintf("Frozen identity mismatch: %s", column))
}

fit_metadata <- suppressMessages(readr::read_csv(
  fit_metadata_path,
  col_types = readr::cols(
    dataset = readr::col_character(), unit_key = readr::col_character()
  ),
  show_col_types = FALSE, progress = FALSE
))
assert_true(
  identical(as.character(fit_metadata$dataset), combined_with_row$dataset) &&
    identical(as.character(fit_metadata$unit_key), combined_with_row$unit_key),
  "Canonical fit metadata differs from the inspectable inputs."
)

combined <- combined_with_row[, c("cell_id", "dataset", "donor", "unit_key", markers)]
marker_matrix <- as.matrix(combined[, markers, drop = FALSE])
storage.mode(marker_matrix) <- "double"
zero_l2_rows <- combined_with_row$row_id[rowSums(marker_matrix^2) == 0]
zero_l2_by_dataset <- setNames(
  lapply(dataset_names, function(dataset) {
    sum(rowSums(marker_matrix[combined$dataset == dataset, , drop = FALSE]^2) == 0)
  }),
  dataset_names
)
assert_true(
  identical(zero_l2_rows, c(47529L, 57813L, 58167L)),
  "The audited three zero-L2 Ptacek cells changed."
)
assert_true(
  all(combined$dataset[zero_l2_rows + 1L] == "MEL_Ptacek2021_MIBI"),
  "Zero-L2 cells no longer belong exclusively to Ptacek."
)
for (marker in markers) {
  assert_true(stats::var(combined[[marker]]) > 0, sprintf("Zero-variance marker: %s", marker))
}

data_dir <- file.path(preprocessing_dir, "data")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
per_dataset_paths <- file.path(
  data_dir, paste0(dataset_slugs, "_analysis_ready.csv.gz")
)
for (index in seq_along(source_frames)) {
  frame <- source_frames[[index]][, c("cell_id", "dataset", "donor", "unit_key", markers)]
  write_csv_atomic(frame, per_dataset_paths[[index]])
}
combined_path <- file.path(data_dir, "combined_analysis_ready.csv.gz")
write_csv_atomic(combined, combined_path)

marker_output_path <- file.path(preprocessing_dir, "markers.txt")
marker_mapping_output_path <- file.path(
  preprocessing_dir, "marker_harmonization_manifest.csv"
)
cell_selection_output_path <- file.path(
  preprocessing_dir, "cell_selection_manifest.csv.gz"
)
copy_atomic(canonical_marker_path, marker_output_path)
copy_atomic(marker_manifest_path, marker_mapping_output_path)
copy_atomic(selected_cells_path, cell_selection_output_path)

audit_rows <- vector("list", length(dataset_names) * length(markers))
cursor <- 1L
for (dataset in dataset_names) {
  selected_dataset <- combined$dataset == dataset
  for (marker in markers) {
    values <- combined[[marker]][selected_dataset]
    audit_rows[[cursor]] <- data.frame(
      dataset = dataset,
      marker = marker,
      cells = length(values),
      minimum = min(values),
      median = stats::median(values),
      mean = mean(values),
      maximum = max(values),
      negative_fraction = mean(values < 0),
      zero_fraction = mean(values == 0),
      stringsAsFactors = FALSE
    )
    cursor <- cursor + 1L
  }
}
input_scale_audit_path <- file.path(preprocessing_dir, "input_scale_audit.csv")
write_plain_csv_atomic(do.call(rbind, audit_rows), input_scale_audit_path)

dataset_records <- list(
  list(
    dataset = dataset_names[[1L]],
    technology = "CODEX multiplexed fluorescence imaging",
    analysis_ready_matrix_file = "data/hickey_codex_analysis_ready.csv.gz",
    analysis_ready_matrix_sha256 = sha256_file(per_dataset_paths[[1L]]),
    source_file = source_paths[[1L]],
    source_file_sha256 = expected_hashes[["hickey_input"]],
    source_state = "standardized",
    assay_or_layer = paste0(
      "Provider-distributed per-cell area-averaged fluorescence after ",
      "marker-column z-normalization within tissue"
    ),
    processing_steps = list(
      "Retain the exact 20,000 frozen pretreatment cells and row order.",
      paste0(
        "Pass through provider tissue-wise marker-z-normalized CODEX values; ",
        "map SOX10/S100 to the documented SOX10 proxy."
      ),
      "Apply no second arcsinh, CLR, shift, clipping, or marker-wise scaling."
    ),
    transform_formula = paste0(
      "Provider z=(area-averaged fluorescence-tissue-marker mean)/",
      "tissue-marker SD; no additional transform"
    ),
    transform_scope = "Provider-applied separately within tissue and marker.",
    processing_reference = paste0(
      "Hickey et al. 2023 Methods and Dryad DOI 10.5061/dryad.k0p2ngfcc README."
    ),
    processing_code_file = "build_inputs.R",
    processing_code_sha256 = sha256_file(script_path),
    universal_positive_median_arcsinh_not_used = TRUE,
    marker_wise_zscore_not_used = TRUE,
    double_transform_checked = TRUE
  ),
  list(
    dataset = dataset_names[[2L]],
    technology = "Imaging mass cytometry",
    analysis_ready_matrix_file = "data/hoch_imc_analysis_ready.csv.gz",
    analysis_ready_matrix_sha256 = sha256_file(per_dataset_paths[[2L]]),
    source_file = source_paths[[2L]],
    source_file_sha256 = expected_hashes[["hoch_input"]],
    source_state = "normalized_and_nonlinearly_transformed",
    assay_or_layer = "Provider exprs assay / exported exprs cell-by-marker table",
    processing_steps = list(
      "Retain the exact 20,000 frozen Stage III treatment-naive cells and row order.",
      "Use provider spillover-corrected single-cell mean marker measurements.",
      "Pass through exprs=asinh(counts/1), with no second transform or scaling."
    ),
    transform_formula = "Provider exprs=asinh(counts/1); no additional transform",
    transform_scope = "Provider-applied before frozen-cell selection.",
    processing_reference = "Hoch et al. Methods and provider SingleCellExperiment assays.",
    processing_code_file = "build_inputs.R",
    processing_code_sha256 = sha256_file(script_path),
    universal_positive_median_arcsinh_not_used = TRUE,
    marker_wise_zscore_not_used = TRUE,
    double_transform_checked = TRUE
  ),
  list(
    dataset = dataset_names[[3L]],
    technology = "Multiplexed ion beam imaging",
    analysis_ready_matrix_file = "data/ptacek_mibi_analysis_ready.csv.gz",
    analysis_ready_matrix_sha256 = sha256_file(per_dataset_paths[[3L]]),
    source_file = source_paths[[3L]],
    source_file_sha256 = expected_hashes[["ptacek_input"]],
    source_state = "provider_analysis_scale",
    assay_or_layer = "Mendeley V2 scaled_mibi_data_w_cell_gating.csv processed proteins",
    processing_steps = list(
      "Retain the exact 20,000 frozen gross Stage III cells and row order.",
      "Pass through provider processed, already arcsinh-documented protein values.",
      "Do not guess/invert the unreported cofactor or apply a second transform."
    ),
    transform_formula = paste0(
      "Provider arcsinh transformation with unreported cofactor; ",
      "distributed values passed through"
    ),
    transform_scope = "Provider-applied before release and frozen-cell selection.",
    processing_reference = "Mendeley Data V2 DOI 10.17632/79y7bht7tf.2.",
    processing_code_file = "build_inputs.R",
    processing_code_sha256 = sha256_file(script_path),
    universal_positive_median_arcsinh_not_used = TRUE,
    marker_wise_zscore_not_used = TRUE,
    double_transform_checked = TRUE
  )
)

preprocessing_manifest_path <- file.path(
  preprocessing_dir, "preprocessing_manifest.json"
)
preprocessing_manifest <- list(
  schema_version = "final_xu.fastmnn_preprocessing.v1",
  method = "fastMNN (batchelor 1.18.1; corrected embedding)",
  combined_input_file = "data/combined_analysis_ready.csv.gz",
  combined_input_sha256 = sha256_file(combined_path),
  shared_marker_policy = paste0(
    "Exact ordered intersection of 12 harmonized proteins frozen before method fitting; ",
    "Hickey SOX10/S100 is a documented SOX10 proxy."
  ),
  shared_marker_selection_used_celltype_or_outcome_labels = FALSE,
  merge_order_selected_using_posthoc_labels_or_metrics = FALSE,
  canonical_parameters_selected_using_posthoc_labels_or_metrics = FALSE,
  correction_marker_file = "markers.txt",
  correction_marker_sha256 = sha256_file(marker_output_path),
  marker_harmonization_manifest_file = "marker_harmonization_manifest.csv",
  marker_harmonization_manifest_sha256 = sha256_file(marker_mapping_output_path),
  cell_selection_manifest_file = "cell_selection_manifest.csv.gz",
  cell_selection_manifest_sha256 = sha256_file(cell_selection_output_path),
  cell_selection_id_column = "cell_id",
  dataset_order_is_progressive_merge_order = TRUE,
  datasets = dataset_records,
  dataset_cells = as.list(expected_cells),
  cells = nrow(combined),
  markers = markers,
  algorithm_visible_columns = colnames(combined),
  input_scale = paste0(
    "Benchmark-specific provider analysis-ready/source scales; no additional ",
    "nonlinear transform, marker-wise z-score, clipping, or multiBatchNorm."
  ),
  algorithm_internal_cosine_normalization = TRUE,
  algorithm_applies_other_expression_transform = FALSE,
  zero_l2_cell_count = length(zero_l2_rows),
  zero_l2_cell_count_by_dataset = zero_l2_by_dataset,
  zero_l2_cells = list(
    count = length(zero_l2_rows),
    row_id = as.list(zero_l2_rows),
    policy = paste0(
      "Retain all cells. batchelor 1.18.1 cosineNorm uses pmax(1e-8,l2), ",
      "leaving an all-zero column unchanged."
    )
  ),
  labels_outcomes_survival_spatial_coordinates_or_umap_read = FALSE,
  canonical_sources = list(
    benchmark_manifest = list(
      path = benchmark_manifest_path,
      sha256 = expected_hashes[["benchmark_manifest"]]
    ),
    canonical_raw_reference = list(
      path = canonical_raw_path,
      sha256 = expected_hashes[["canonical_raw"]]
    ),
    fit_metadata = list(
      path = fit_metadata_path,
      sha256 = expected_hashes[["fit_metadata"]]
    )
  ),
  input_scale_audit = list(
    file = "input_scale_audit.csv",
    sha256 = sha256_file(input_scale_audit_path)
  ),
  builder = list(path = script_path, sha256 = sha256_file(script_path)),
  runtime = list(
    R = as.character(getRversion()),
    readr = as.character(utils::packageVersion("readr")),
    jsonlite = as.character(utils::packageVersion("jsonlite")),
    digest = as.character(utils::packageVersion("digest"))
  )
)
write_json_atomic(preprocessing_manifest, preprocessing_manifest_path)

algorithm_config <- list(
  schema_version = "final_xu.fastmnn_repro.v1",
  method_label = "fastMNN (batchelor 1.18.1; corrected embedding)",
  cell_id_column = "cell_id",
  batch_column = "dataset",
  sample_column = "unit_key",
  batch_levels = as.list(dataset_names),
  input_scale = list(
    status = "analysis_ready",
    name = "melanoma provider analysis-ready/source-scale shared proteins",
    description = paste0(
      "Dataset-specific provider-supported passthrough: Hickey tissue-wise z values, ",
      "Hoch exprs=asinh(counts/1), and Ptacek provider processed arcsinh scale."
    ),
    preprocessing_record_file = "preprocessing/preprocessing_manifest.json",
    preprocessing_record_sha256 = sha256_file(preprocessing_manifest_path),
    input_sha256 = sha256_file(combined_path),
    marker_file_sha256 = sha256_file(marker_output_path),
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
algorithm_config_path <- file.path(method_root, "algorithm_config.json")
write_json_atomic(algorithm_config, algorithm_config_path)

benchmark_config_path <- file.path(method_root, "config.json")
if (!file.exists(benchmark_config_path)) stopf("Missing benchmark config: %s", benchmark_config_path)
build_manifest_path <- file.path(
  preprocessing_dir, "preprocessing_build_manifest.json"
)
build_manifest <- list(
  schema_version = "final_xu.fastmnn_preprocessing_build.v1",
  status = "complete",
  benchmark = "melanoma",
  method = "fastMNN (batchelor 1.18.1; corrected embedding)",
  cells = nrow(combined),
  marker_count = length(markers),
  dataset_order = as.list(dataset_names),
  dataset_cells = as.list(expected_cells),
  zero_l2_cells_retained = length(zero_l2_rows),
  label_free_preprocessing = TRUE,
  builder = list(path = script_path, sha256 = sha256_file(script_path)),
  benchmark_config = list(
    path = benchmark_config_path,
    sha256 = sha256_file(benchmark_config_path)
  ),
  authenticated_sources = lapply(seq_along(observed_hashes), function(index) {
    list(name = names(observed_hashes)[[index]], sha256 = observed_hashes[[index]])
  }),
  outputs = list(
    combined_input = list(
      path = combined_path,
      sha256 = sha256_file(combined_path)
    ),
    marker_file = list(
      path = marker_output_path,
      sha256 = sha256_file(marker_output_path)
    ),
    preprocessing_manifest = list(
      path = preprocessing_manifest_path,
      sha256 = sha256_file(preprocessing_manifest_path)
    ),
    algorithm_config = list(
      path = algorithm_config_path,
      sha256 = sha256_file(algorithm_config_path)
    )
  )
)
write_json_atomic(build_manifest, build_manifest_path)

preprocessing_complete_path <- file.path(preprocessing_dir, "COMPLETE.json")
preprocessing_complete <- list(
  schema_version = "final_xu.fastmnn_preprocessing_complete.v1",
  status = "complete",
  preprocessing_manifest_sha256 = sha256_file(preprocessing_manifest_path),
  preprocessing_build_manifest_sha256 = sha256_file(build_manifest_path),
  algorithm_config_sha256 = sha256_file(algorithm_config_path)
)
write_json_atomic(preprocessing_complete, preprocessing_complete_path)

cat(sprintf(
  paste0(
    "PASS: melanoma fastMNN preprocessing; %d cells x %d markers; ",
    "zero-L2 retained=%d; combined SHA-256=%s\n"
  ),
  nrow(combined), length(markers), length(zero_l2_rows), sha256_file(combined_path)
))
