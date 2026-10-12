#!/usr/bin/env Rscript

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_argument)) stop("Cannot determine test location.", call. = FALSE)
script_path <- normalizePath(sub("^--file=", "", script_argument[[1L]]), mustWork = TRUE)
algorithm_dir <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)
local_library <- file.path(algorithm_dir, ".Rlib")
.libPaths(c(local_library, .libPaths()))
source(file.path(algorithm_dir, "fastmnn_repro.R"), local = FALSE)

for (package in c("jsonlite", "readr", "digest")) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stop(sprintf("Missing test dependency: %s", package), call. = FALSE)
  }
}

assert_true <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
}

test_root <- tempfile("fastmnn-cli-contract-")
dir.create(test_root, recursive = TRUE)
on.exit(unlink(test_root, recursive = TRUE, force = TRUE), add = TRUE)

set.seed(9107L)
batch_levels <- c("source_one", "source_two", "source_three")
n_per_batch <- 45L
n_markers <- 6L
batches <- rep(batch_levels, each = n_per_batch)
states <- rep(rep(c(0, 1, 2), each = n_per_batch / 3L), 3L)
matrix_values <- matrix(rexp(length(batches) * n_markers, rate = 1.8), ncol = n_markers)
matrix_values[, 1L] <- matrix_values[, 1L] + states * 0.9
matrix_values[, 2L] <- matrix_values[, 2L] + (states == 2L) * 1.2
matrix_values <- matrix_values + rep(c(0.15, 0.35, 0.05), each = n_per_batch)
colnames(matrix_values) <- paste0("protein_", seq_len(n_markers))
input <- data.frame(
  cell_id = sprintf("cli_cell_%04d", seq_len(nrow(matrix_values))),
  dataset = batches,
  unit_key = paste0(batches, "_unit_", rep(seq_len(9L), each = 5L, times = 3L)),
  matrix_values,
  provider_celltype = paste0("type_", states),
  clinical_outcome = rep(c("yes", "no"), length.out = nrow(matrix_values)),
  stringsAsFactors = FALSE,
  check.names = FALSE
)
input <- input[sample(seq_len(nrow(input))), , drop = FALSE]
rownames(input) <- NULL

input_path <- file.path(test_root, "combined.csv.gz")
marker_path <- file.path(test_root, "markers.txt")
record_path <- file.path(test_root, "preprocessing_manifest.json")
config_path <- file.path(test_root, "config.json")
readr::write_csv(input, input_path)
writeLines(colnames(matrix_values), marker_path)
jsonlite::write_json(
  list(
    schema_version = "final_xu.fastmnn_preprocessing.v1",
    status = "analysis_ready",
    combined_input_file = basename(input_path),
    combined_input_sha256 = sha256_file(input_path),
    correction_marker_file = basename(marker_path),
    correction_marker_sha256 = sha256_file(marker_path),
    shared_marker_selection_used_celltype_or_outcome_labels = FALSE,
    merge_order_selected_using_posthoc_labels_or_metrics = FALSE,
    canonical_parameters_selected_using_posthoc_labels_or_metrics = FALSE,
    dataset_order_is_progressive_merge_order = TRUE,
    datasets = lapply(batch_levels, function(dataset) list(dataset = dataset)),
    algorithm_layer_expression_transform = "none",
    note = "Synthetic contract fixture."
  ),
  record_path, pretty = TRUE, auto_unbox = TRUE
)

make_config <- function(input_hash = sha256_file(input_path)) {
  list(
    schema_version = FASTMNN_REPRO_SCHEMA,
    method_label = FASTMNN_METHOD_LABEL,
    cell_id_column = "cell_id",
    batch_column = "dataset",
    sample_column = "unit_key",
    batch_levels = batch_levels,
    input_scale = list(
      status = "analysis_ready",
      name = "synthetic analysis-ready test scale",
      description = "Only for CLI contract testing.",
      preprocessing_record_file = basename(record_path),
      preprocessing_record_sha256 = sha256_file(record_path),
      input_sha256 = input_hash,
      marker_file_sha256 = sha256_file(marker_path),
      algorithm_internal_cosine_normalization = TRUE,
      algorithm_applies_other_expression_transform = FALSE
    ),
    algorithm_parameters = canonical_fastmnn_parameters()
  )
}

write_config <- function(config) {
  jsonlite::write_json(
    config, config_path, pretty = TRUE, auto_unbox = TRUE,
    null = "null", digits = 17
  )
}

runner <- file.path(algorithm_dir, "run_fastmnn.R")
rscript <- file.path(R.home("bin"), "Rscript")
run_cli <- function(output_path, environment = character()) {
  arguments <- c(
    runner,
    "--input", input_path,
    "--markers", marker_path,
    "--config", config_path,
    "--output", output_path
  )
  suppressWarnings(system2(
    rscript,
    args = shQuote(arguments),
    stdout = TRUE,
    stderr = TRUE,
    env = environment
  ))
}

write_config(make_config())
output_path <- file.path(test_root, "alignment")
successful_log <- run_cli(output_path)
successful_status <- attr(successful_log, "status")
assert_true(
  is.null(successful_status) || identical(successful_status, 0L),
  paste("Canonical CLI failed:", paste(successful_log, collapse = "\n"))
)

required_files <- c(
  "COMPLETE.json",
  "authenticated_preprocessing_manifest.json",
  "cell_metadata_label_free.csv.gz",
  "fastmnn_corrected_embedding.csv.gz",
  "fastmnn_corrected_embedding.rds",
  "fastmnn_lost_variance.csv",
  "fastmnn_merge_diagnostics.csv",
  "fastmnn_merge_info.rds",
  "fastmnn_pca_centers.csv",
  "fastmnn_pca_rotation.csv.gz",
  "fastmnn_pca_variance.csv",
  "fastmnn_runtime_messages.txt",
  "input_cell_l2_statistics.csv",
  "input_column_audit.json",
  "input_marker_batch_statistics.csv",
  "manifest.json",
  "method_semantics.json",
  "resolved_config.json",
  "sessionInfo.txt"
)
assert_true(
  all(file.exists(file.path(output_path, required_files))),
  "The successful CLI run did not emit its complete output contract."
)
assert_true(
  !any(grepl("reconstruct", list.files(output_path), ignore.case = TRUE)),
  "The canonical CLI exported a reconstructed marker assay."
)
assert_true(
  identical(
    sha256_file(file.path(output_path, "authenticated_preprocessing_manifest.json")),
    sha256_file(record_path)
  ),
  "The published preprocessing-manifest snapshot differs from its authenticated source."
)

embedding <- readr::read_csv(
  file.path(output_path, "fastmnn_corrected_embedding.csv.gz"),
  show_col_types = FALSE
)
metadata <- readr::read_csv(
  file.path(output_path, "cell_metadata_label_free.csv.gz"),
  show_col_types = FALSE
)
embedding_rds <- readRDS(file.path(output_path, "fastmnn_corrected_embedding.rds"))
assert_true(
  identical(as.character(embedding$cell_id), input$cell_id),
  "CLI embedding does not preserve input cell order."
)
assert_true(
  identical(as.character(metadata$cell_id), input$cell_id) &&
    identical(colnames(metadata), c("cell_id", "batch", "sample")),
  "CLI metadata is not label-free or not in input cell order."
)
assert_true(
  identical(rownames(embedding_rds), input$cell_id) &&
    identical(colnames(embedding_rds), colnames(embedding)[-1L]) &&
    isTRUE(all.equal(
      unname(embedding_rds),
      unname(as.matrix(embedding[, -1L, drop = FALSE])),
      tolerance = 1e-14,
      check.attributes = FALSE
    )),
  "CSV and RDS corrected embeddings are inconsistent."
)

audit <- jsonlite::read_json(file.path(output_path, "input_column_audit.json"))
assert_true(
  all(c("provider_celltype", "clinical_outcome") %in% unlist(audit$ignored_input_columns)),
  "Label/outcome columns were not explicitly audited as ignored."
)
semantics <- jsonlite::read_json(file.path(output_path, "method_semantics.json"))
assert_true(
  identical(semantics$primary_output, "reducedDim(result, 'corrected')") &&
    identical(semantics$reconstructed_assay_exported, FALSE) &&
    identical(semantics$marker_space_d_NA_used, FALSE),
  "Method semantics do not describe the canonical corrected latent representation."
)

manifest_path <- file.path(output_path, "manifest.json")
manifest <- jsonlite::read_json(manifest_path, simplifyVector = FALSE)
complete <- jsonlite::read_json(file.path(output_path, "COMPLETE.json"))
assert_true(
  identical(complete$manifest_sha256, sha256_file(manifest_path)),
  "COMPLETE.json does not authenticate manifest.json."
)
manifest_files <- sort(vapply(
  manifest$output_files, function(entry) entry$file, character(1)
))
expected_manifest_files <- sort(setdiff(
  required_files, c("manifest.json", "COMPLETE.json")
))
assert_true(
  identical(manifest_files, expected_manifest_files),
  "manifest.json does not cover the complete pre-manifest output set."
)
for (entry in manifest$output_files) {
  target <- file.path(output_path, entry$file)
  assert_true(file.exists(target), sprintf("Manifest output is missing: %s", entry$file))
  assert_true(
    identical(sha256_file(target), entry$sha256),
    sprintf("Manifest hash mismatch: %s", entry$file)
  )
}

# The preprocessing record must explicitly bind label-free choices.
valid_preprocessing_record <- jsonlite::read_json(record_path, simplifyVector = FALSE)
invalid_preprocessing_record <- valid_preprocessing_record
invalid_preprocessing_record$shared_marker_selection_used_celltype_or_outcome_labels <- TRUE
jsonlite::write_json(
  invalid_preprocessing_record,
  record_path,
  pretty = TRUE,
  auto_unbox = TRUE,
  null = "null"
)
write_config(make_config())
label_leak_output <- file.path(test_root, "label_leak_alignment")
label_leak_log <- run_cli(label_leak_output)
assert_true(
  !is.null(attr(label_leak_log, "status")) && attr(label_leak_log, "status") != 0L,
  "A preprocessing record declaring label-based marker selection was accepted."
)
assert_true(
  !file.exists(label_leak_output),
  "Label-leak declaration failure published an output directory."
)
jsonlite::write_json(
  valid_preprocessing_record,
  record_path,
  pretty = TRUE,
  auto_unbox = TRUE,
  null = "null"
)

# A false input provenance binding must fail before any output is staged.
write_config(make_config(strrep("0", 64L)))
bad_output <- file.path(test_root, "bad_hash_alignment")
bad_log <- run_cli(bad_output)
assert_true(
  !is.null(attr(bad_log, "status")) && attr(bad_log, "status") != 0L,
  "A bad input SHA-256 was accepted."
)
assert_true(!file.exists(bad_output), "Bad-hash run published an output directory.")
assert_true(
  !length(Sys.glob(file.path(test_root, ".bad_hash_alignment.staging-*"))),
  "Bad-hash run left a staging directory."
)

# A failure after the first staged file must still leave no partial result.
write_config(make_config())
fault_output <- file.path(test_root, "fault_alignment")
fault_log <- run_cli(
  fault_output,
  environment = "FINAL_XU_FASTMNN_TEST_FAULT=after_first_staged_output"
)
assert_true(
  !is.null(attr(fault_log, "status")) && attr(fault_log, "status") != 0L,
  "Injected post-stage fault did not fail."
)
assert_true(!file.exists(fault_output), "Faulted run published a partial output.")
assert_true(
  !length(Sys.glob(file.path(test_root, ".fault_alignment.staging-*"))),
  "Faulted run left a staging directory."
)

# A failure after every staged file is complete but before rename must also be clean.
late_fault_output <- file.path(test_root, "late_fault_alignment")
late_fault_log <- run_cli(
  late_fault_output,
  environment = "FINAL_XU_FASTMNN_TEST_FAULT=before_atomic_publish"
)
assert_true(
  !is.null(attr(late_fault_log, "status")) && attr(late_fault_log, "status") != 0L,
  "Injected pre-publication fault did not fail."
)
assert_true(
  !file.exists(late_fault_output),
  "Late-faulted run published a partial output."
)
assert_true(
  !length(Sys.glob(file.path(test_root, ".late_fault_alignment.staging-*"))),
  "Late-faulted run left a staging directory."
)

cat(sprintf(
  "PASS: CLI contract; %d cells, %d outputs, hashes and atomic failure paths verified.\n",
  nrow(input), length(required_files)
))
