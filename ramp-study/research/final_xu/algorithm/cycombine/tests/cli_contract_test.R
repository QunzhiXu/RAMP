#!/usr/bin/env Rscript

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_argument)) stop("Cannot determine script location.", call. = FALSE)
test_path <- sub("^--file=", "", script_argument[[1L]])
test_path <- gsub("~+~", " ", test_path, fixed = TRUE)
test_path <- normalizePath(test_path, mustWork = TRUE)
algorithm_dir <- dirname(dirname(test_path))
local_library <- file.path(algorithm_dir, ".Rlib")
.libPaths(c(local_library, .libPaths()))

for (package in c("readr", "jsonlite", "digest")) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stop(sprintf("Missing CLI-test package: %s", package), call. = FALSE)
  }
}

hash_file <- function(path) {
  digest::digest(file = path, algo = "sha256", serialize = FALSE)
}

test_dir <- tempfile("cycombine-cli-contract-")
dir.create(test_dir, recursive = TRUE)
on.exit(unlink(test_dir, recursive = TRUE, force = TRUE), add = TRUE)

set.seed(20260830)
batch_levels <- c("platform_A", "platform_B", "platform_C")
markers <- c("CD3", "CD4", "CD8", "CD45", "HLA_DR")
parts <- lapply(seq_along(batch_levels), function(batch_index) {
  n <- 300L
  state <- sample.int(4L, n, replace = TRUE)
  shift <- c(-0.25, 0.10, 0.35)[batch_index]
  z1 <- stats::rnorm(n, state * 0.6, 0.45)
  z2 <- stats::rnorm(n, (5 - state) * 0.4, 0.5)
  data.frame(
    cell_id = sprintf("%s_cell_%04d", batch_levels[batch_index], seq_len(n)),
    dataset = batch_levels[batch_index],
    unit_key = sprintf("%s_donor_%02d", batch_levels[batch_index], 1L + (seq_len(n) - 1L) %% 6L),
    CD3 = 4 + z1 + shift + stats::rnorm(n, 0, 0.1),
    CD4 = 3.5 + z2 - shift / 2 + stats::rnorm(n, 0, 0.1),
    CD8 = 3 + 0.7 * z1 - 0.3 * z2 + shift + stats::rnorm(n, 0, 0.1),
    CD45 = 5 + 0.2 * z1 + 0.2 * z2 + stats::rnorm(n, 0, 0.1),
    HLA_DR = 2.5 - 0.3 * z1 + 0.6 * z2 - shift + stats::rnorm(n, 0, 0.1),
    provider_celltype = paste0("class_", state),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
})

dataset_paths <- file.path(test_dir, paste0("dataset_", seq_along(parts), ".csv.gz"))
for (index in seq_along(parts)) readr::write_csv(parts[[index]], dataset_paths[index])
combined <- do.call(rbind, parts)
rownames(combined) <- NULL
combined_path <- file.path(test_dir, "combined.csv.gz")
readr::write_csv(combined, combined_path)

cell_selection_path <- file.path(test_dir, "selected_cells.csv.gz")
readr::write_csv(combined[, c("cell_id", "dataset", "unit_key")], cell_selection_path)
marker_harmonization_path <- file.path(test_dir, "marker_manifest.csv")
readr::write_csv(data.frame(marker = markers, used_for_alignment = TRUE), marker_harmonization_path)
marker_path <- file.path(test_dir, "markers.txt")
writeLines(markers, marker_path)
processing_code_path <- file.path(test_dir, "preprocess_fixture.R")
writeLines("# Synthetic fixture is already on a finite analysis scale.", processing_code_path)

dataset_entries <- lapply(seq_along(parts), function(index) {
  list(
    dataset = batch_levels[index],
    technology = paste0("synthetic_platform_", index),
    input_matrix_file = dataset_paths[index],
    input_matrix_sha256 = hash_file(dataset_paths[index]),
    source_state = "provider_analysis_scale",
    assay_or_layer = "synthetic analysis-ready protein matrix",
    processing_steps = list("Generate the predeclared finite synthetic fixture without labels in fitting."),
    cycombine_input_scale = "synthetic continuous protein values",
    transform_formula = "none",
    transform_scope = "all frozen fixture cells",
    processing_reference = "algorithm/cycombine/tests/cli_contract_test.R",
    processing_code_file = processing_code_path,
    processing_code_sha256 = hash_file(processing_code_path),
    double_transform_checked = TRUE
  )
})

preprocessing <- list(
  schema_version = "final_xu.cycombine_preprocessing.v2",
  combined_input_sha256 = hash_file(combined_path),
  shared_marker_policy = "Five frozen synthetic shared markers.",
  correction_marker_file = marker_path,
  correction_marker_sha256 = hash_file(marker_path),
  marker_harmonization_manifest_file = marker_harmonization_path,
  marker_harmonization_manifest_sha256 = hash_file(marker_harmonization_path),
  cell_selection_manifest_file = cell_selection_path,
  cell_selection_manifest_sha256 = hash_file(cell_selection_path),
  cell_selection_id_column = "cell_id",
  datasets = dataset_entries
)
preprocessing_path <- file.path(test_dir, "preprocessing_manifest.json")
jsonlite::write_json(preprocessing, preprocessing_path, pretty = TRUE, auto_unbox = TRUE)

config <- list(
  schema_version = "final_xu.cycombine_repro.v1",
  cell_id_column = "cell_id",
  batch_column = "dataset",
  sample_column = "unit_key",
  batch_levels = batch_levels,
  input_scale = list(
    status = "analysis_ready",
    description = "Synthetic CLI contract fixture; no transform in the algorithm layer.",
    provenance_file = preprocessing_path,
    algorithm_applies_transformation = FALSE
  ),
  algorithm_parameters = list(
    xdim = 8L,
    ydim = 8L,
    rlen = 10L,
    seed = 840L,
    norm_method = "rank",
    ties_method = "average",
    parametric = TRUE,
    covariate = NULL,
    anchor = NULL
  )
)
config_path <- file.path(test_dir, "config.json")
jsonlite::write_json(config, config_path, pretty = TRUE, auto_unbox = TRUE, null = "null")

runner <- file.path(algorithm_dir, "run_cycombine.R")
output_path <- file.path(test_dir, "alignment")
run_cli <- function(config_file, output_directory) {
  command_output <- system2(
    file.path(R.home("bin"), "Rscript"),
    c(
      "--vanilla", shQuote(runner),
      "--input", shQuote(combined_path),
      "--markers", shQuote(marker_path),
      "--config", shQuote(config_file),
      "--output", shQuote(output_directory)
    ),
    stdout = TRUE,
    stderr = TRUE
  )
  list(output = command_output, status = attr(command_output, "status") %||% 0L)
}

`%||%` <- function(left, right) if (is.null(left)) right else left

good_run <- run_cli(config_path, output_path)
if (good_run$status != 0L) {
  stop(paste(c("CLI runner failed:", good_run$output), collapse = "\n"), call. = FALSE)
}
required_outputs <- c(
  "corrected_expression.csv.gz", "cell_metadata_label_free.csv.gz",
  "correction_som_assignments.csv.gz", "correction_som_node_batch_counts.csv",
  "correction_som_node_modes.csv", "resolved_preprocessing_manifest.json",
  "manifest.json", "COMPLETE.json"
)
if (!all(file.exists(file.path(output_path, required_outputs)))) {
  stop("CLI output contract is incomplete.", call. = FALSE)
}
complete <- jsonlite::read_json(file.path(output_path, "COMPLETE.json"), simplifyVector = TRUE)
if (!identical(complete$status, "complete") ||
    !identical(complete$manifest_sha256, hash_file(file.path(output_path, "manifest.json")))) {
  stop("COMPLETE.json does not authenticate the run manifest.", call. = FALSE)
}
corrected <- readr::read_csv(
  file.path(output_path, "corrected_expression.csv.gz"),
  show_col_types = FALSE
)
if (nrow(corrected) != nrow(combined) || !identical(as.character(corrected$cell_id), combined$cell_id)) {
  stop("CLI corrected output did not preserve rows and cell order.", call. = FALSE)
}

bad_preprocessing <- preprocessing
bad_preprocessing$combined_input_sha256 <- paste(rep("0", 64L), collapse = "")
bad_preprocessing_path <- file.path(test_dir, "bad_preprocessing_manifest.json")
jsonlite::write_json(bad_preprocessing, bad_preprocessing_path, pretty = TRUE, auto_unbox = TRUE)
bad_config <- config
bad_config$input_scale$provenance_file <- bad_preprocessing_path
bad_config_path <- file.path(test_dir, "bad_config.json")
jsonlite::write_json(bad_config, bad_config_path, pretty = TRUE, auto_unbox = TRUE, null = "null")
bad_output_path <- file.path(test_dir, "bad_alignment")
bad_run <- suppressWarnings(run_cli(bad_config_path, bad_output_path))
if (bad_run$status == 0L || dir.exists(bad_output_path)) {
  stop("Hash-negative CLI test failed or left a partial output directory.", call. = FALSE)
}

cat(sprintf(
  "PASS: CLI v2 contract; %d cells; exact dataset binding; atomic output; hash-negative failure clean\n",
  nrow(combined)
))
