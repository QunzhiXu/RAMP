#!/usr/bin/env Rscript

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_argument)) stop("Cannot determine script location.", call. = FALSE)
test_path <- sub("^--file=", "", script_argument[[1L]])
test_path <- gsub("~+~", " ", test_path, fixed = TRUE)
test_path <- normalizePath(test_path, mustWork = TRUE)
algorithm_dir <- dirname(dirname(test_path))
local_library <- file.path(algorithm_dir, ".Rlib")
.libPaths(c(local_library, .libPaths()))
source(file.path(algorithm_dir, "harmony_direct_repro.R"), local = FALSE)
for (package in c("readr", "jsonlite", "digest")) {
  if (!requireNamespace(package, quietly = TRUE)) stop(sprintf("Missing test package: %s", package), call. = FALSE)
}

test_dir <- tempfile("harmony-direct-cli-")
dir.create(test_dir, recursive = TRUE)
on.exit(unlink(test_dir, recursive = TRUE, force = TRUE), add = TRUE)
batch_levels <- c("platform_C", "platform_A", "platform_B")
markers <- c("CD3", "CD4", "CD8", "CD19", "CD45")
set.seed(831)
parts <- lapply(seq_along(batch_levels), function(index) {
  n <- 120L
  state <- sample.int(3L, n, replace = TRUE)
  shift <- c(-0.4, 0.1, 0.55)[index]
  z1 <- stats::rnorm(n, state * 0.5, 0.45)
  z2 <- stats::rnorm(n, (4 - state) * 0.45, 0.5)
  data.frame(
    cell_id = sprintf("%s_%04d", batch_levels[index], seq_len(n)),
    dataset = batch_levels[index],
    unit_key = sprintf("%s_donor_%02d", batch_levels[index], 1L + (seq_len(n) - 1L) %% 5L),
    CD3 = 3 + z1 + shift,
    CD4 = 3 + z2 - shift / 2,
    CD8 = 2.5 + 0.7 * z1 - 0.2 * z2 + shift,
    CD19 = 2 - 0.3 * z1 + 0.6 * z2 - shift,
    CD45 = 5 + 0.15 * z1 + 0.2 * z2,
    provider_celltype = paste0("type_", state),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
})
combined <- do.call(rbind, parts)
rownames(combined) <- NULL
input_path <- file.path(test_dir, "combined.csv.gz")
marker_path <- file.path(test_dir, "markers.txt")
record_path <- file.path(test_dir, "benchmark_preprocessing_record.txt")
readr::write_csv(combined, input_path)
writeLines(markers, marker_path)
writeLines(c(
  "Synthetic analysis-ready marker matrix.",
  "No transform is performed by algorithm/harmony."
), record_path)

config <- list(
  schema_version = HARMONY_DIRECT_SCHEMA,
  method_label = HARMONY_DIRECT_METHOD_LABEL,
  cell_id_column = "cell_id",
  batch_column = "dataset",
  sample_column = "unit_key",
  batch_levels = batch_levels,
  input_scale = list(
    status = "analysis_ready",
    name = "synthetic continuous protein scale",
    description = "Frozen CLI fixture; intentionally no transform in the algorithm layer.",
    preprocessing_record_file = record_path,
    preprocessing_record_sha256 = sha256_file(record_path),
    input_sha256 = sha256_file(input_path),
    marker_file_sha256 = sha256_file(marker_path),
    algorithm_applies_transformation = FALSE
  ),
  algorithm_parameters = canonical_harmony_direct_parameters()
)
config_path <- file.path(test_dir, "config.json")
jsonlite::write_json(config, config_path, pretty = TRUE, auto_unbox = TRUE, null = "null", digits = 17)

runner <- file.path(algorithm_dir, "run_harmony_direct.R")
run_cli <- function(config_file, output_directory, env = character()) {
  command_output <- system2(
    file.path(R.home("bin"), "Rscript"),
    c(
      "--vanilla", shQuote(runner),
      "--input", shQuote(input_path),
      "--markers", shQuote(marker_path),
      "--config", shQuote(config_file),
      "--output", shQuote(output_directory)
    ),
    env = env,
    stdout = TRUE,
    stderr = TRUE
  )
  list(output = command_output, status = attr(command_output, "status") %||% 0L)
}

output_path <- file.path(test_dir, "alignment")
good <- run_cli(config_path, output_path)
if (good$status != 0L) stop(paste(c("CLI runner failed:", good$output), collapse = "\n"), call. = FALSE)
required <- c(
  "harmony_adjusted_marker_coordinates.csv.gz",
  "harmony_adjusted_marker_coordinates.rds",
  "cell_metadata_label_free.csv.gz",
  "marker_rank_audit.csv",
  "marker_rank_audit_by_sample.csv",
  "marker_value_shift_audit.csv",
  "input_column_audit.json",
  "method_semantics.json",
  "resolved_config.json",
  "manifest.json",
  "COMPLETE.json"
)
if (!all(file.exists(file.path(output_path, required)))) stop("CLI output contract is incomplete.", call. = FALSE)
adjusted <- readRDS(file.path(output_path, "harmony_adjusted_marker_coordinates.rds"))
if (!identical(rownames(adjusted), combined$cell_id) || !identical(colnames(adjusted), markers)) {
  stop("CLI output did not preserve cell and marker order.", call. = FALSE)
}
roundtrip_input <- read_analysis_table(
  input_path,
  character_columns = c("cell_id", "dataset", "unit_key")
)
direct_fit <- fit_harmony_direct(
  input = roundtrip_input,
  markers = markers,
  cell_id_column = "cell_id",
  batch_column = "dataset",
  sample_column = "unit_key",
  batch_levels = batch_levels,
  parameters = canonical_harmony_direct_parameters(),
  required_library = local_library
)
if (!identical(adjusted, direct_fit$adjusted_marker_coordinates)) {
  stop("CLI RDS output differs from the direct audited wrapper on the same round-tripped input.", call. = FALSE)
}
column_audit <- jsonlite::read_json(file.path(output_path, "input_column_audit.json"), simplifyVector = TRUE)
if (!"provider_celltype" %in% column_audit$ignored_input_columns || "provider_celltype" %in% column_audit$fitting_input_columns) {
  stop("CLI label-leak audit failed.", call. = FALSE)
}
complete <- jsonlite::read_json(file.path(output_path, "COMPLETE.json"), simplifyVector = TRUE)
if (!identical(complete$status, "complete") ||
    !identical(complete$manifest_sha256, sha256_file(file.path(output_path, "manifest.json")))) {
  stop("COMPLETE.json does not authenticate manifest.json.", call. = FALSE)
}

bad_config <- config
bad_config$input_scale$input_sha256 <- paste(rep("0", 64L), collapse = "")
bad_config_path <- file.path(test_dir, "bad_config.json")
jsonlite::write_json(bad_config, bad_config_path, pretty = TRUE, auto_unbox = TRUE, null = "null", digits = 17)
bad_output_path <- file.path(test_dir, "bad_alignment")
bad <- suppressWarnings(run_cli(bad_config_path, bad_output_path))
if (bad$status == 0L || dir.exists(bad_output_path)) {
  stop("Hash-negative run succeeded or left a partial output directory.", call. = FALSE)
}

fault_output_path <- file.path(test_dir, "post_stage_failure")
fault <- suppressWarnings(run_cli(
  config_path,
  fault_output_path,
  env = "FINAL_XU_HARMONY_TEST_FAULT=after_first_staged_output"
))
staging_prefix <- sprintf(".%s.staging-", basename(fault_output_path))
staging_leftovers <- list.files(
  dirname(fault_output_path),
  all.files = TRUE,
  full.names = FALSE,
  no.. = TRUE
)
staging_leftovers <- staging_leftovers[startsWith(staging_leftovers, staging_prefix)]
if (fault$status == 0L || dir.exists(fault_output_path) || length(staging_leftovers)) {
  stop("Post-staging fault did not fail cleanly or left staged/final output behind.", call. = FALSE)
}

cat(sprintf(
  paste0(
    "PASS: Harmony-direct CLI contract; %d cells; label-free fit; exact hashes; ",
    "atomic publish; invalid hash and post-staging fault both fail cleanly\n"
  ),
  nrow(combined)
))
