#!/usr/bin/env Rscript

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_argument)) stop("Cannot determine script location.", call. = FALSE)
script_path <- sub("^--file=", "", script_argument[[1L]])
script_path <- gsub("~+~", " ", script_path, fixed = TRUE)
script_path <- normalizePath(script_path, mustWork = TRUE)
algorithm_dir <- dirname(script_path)
local_library <- file.path(algorithm_dir, ".Rlib")
.libPaths(c(local_library, .libPaths()))
source(file.path(algorithm_dir, "harmony_direct_repro.R"), local = FALSE)

usage <- function() {
  cat(paste0(
    "Usage:\n",
    "  Rscript run_harmony_direct.R --input INPUT.csv.gz --markers markers.txt \\\n",
    "    --config config.json --output OUTPUT_DIRECTORY\n"
  ))
}

parse_cli <- function(arguments) {
  if (!length(arguments) || any(arguments %in% c("-h", "--help"))) {
    usage()
    quit(status = if (length(arguments)) 0L else 2L)
  }
  if (length(arguments) %% 2L != 0L) stopf("Every CLI flag requires one value.")
  allowed <- c("--input", "--markers", "--config", "--output")
  keys <- arguments[seq(1L, length(arguments), by = 2L)]
  values <- arguments[seq(2L, length(arguments), by = 2L)]
  if (any(!keys %in% allowed) || anyDuplicated(keys)) {
    stopf("Unknown or repeated CLI flags: %s.", paste(keys, collapse = ", "))
  }
  missing <- setdiff(allowed, keys)
  if (length(missing)) stopf("Missing CLI flags: %s.", paste(missing, collapse = ", "))
  result <- as.list(values)
  names(result) <- sub("^--", "", keys)
  result
}

resolve_existing_path <- function(path, base_directory = getwd()) {
  if (!is.character(path) || length(path) != 1L || is.na(path) || !nzchar(path)) {
    stopf("Path must be one non-empty string.")
  }
  candidate <- if (grepl("^/", path)) path else file.path(base_directory, path)
  normalizePath(candidate, mustWork = TRUE)
}

validate_sha256_string <- function(value, field) {
  if (!is.character(value) || length(value) != 1L || is.na(value) || !grepl("^[0-9a-fA-F]{64}$", value)) {
    stopf("%s must be one 64-character hexadecimal SHA-256.", field)
  }
  tolower(value)
}

validate_config <- function(config, config_path, input_path, marker_path) {
  expected_fields <- c(
    "schema_version", "method_label", "cell_id_column", "batch_column",
    "sample_column", "batch_levels", "input_scale", "algorithm_parameters"
  )
  if (!is.list(config) || !setequal(names(config), expected_fields)) {
    stopf("Config fields do not exactly match the v1 schema.")
  }
  if (!identical(config$schema_version, HARMONY_DIRECT_SCHEMA)) stopf("Unsupported config schema_version.")
  if (!identical(config$method_label, HARMONY_DIRECT_METHOD_LABEL)) stopf("method_label must use the exact audited method name.")
  scalar_columns <- c(config$cell_id_column, config$batch_column, config$sample_column)
  if (length(scalar_columns) != 3L || any(!vapply(scalar_columns, function(x) is.character(x) && length(x) == 1L && !is.na(x) && nzchar(x), logical(1))) ||
      anyDuplicated(scalar_columns)) {
    stopf("cell_id_column, batch_column, and sample_column must be distinct non-empty strings.")
  }
  if (!is.character(config$batch_levels) || length(config$batch_levels) < 2L ||
      anyNA(config$batch_levels) || any(!nzchar(config$batch_levels)) || anyDuplicated(config$batch_levels)) {
    stopf("batch_levels must contain at least two unique non-empty names.")
  }

  scale_fields <- c(
    "status", "name", "description", "preprocessing_record_file",
    "preprocessing_record_sha256", "input_sha256", "marker_file_sha256",
    "algorithm_applies_transformation"
  )
  scale <- config$input_scale
  if (!is.list(scale) || !setequal(names(scale), scale_fields)) stopf("input_scale fields do not exactly match the v1 schema.")
  text_fields <- c(scale$status, scale$name, scale$description, scale$preprocessing_record_file)
  if (length(text_fields) != 4L || any(!vapply(text_fields, function(x) is.character(x) && length(x) == 1L && !is.na(x) && nzchar(x), logical(1)))) {
    stopf("input_scale text fields must be non-empty strings.")
  }
  if (!identical(scale$status, "analysis_ready")) stopf("input_scale.status must be 'analysis_ready'.")
  if (!identical(scale$algorithm_applies_transformation, FALSE)) {
    stopf("algorithm_applies_transformation must be false: this algorithm layer performs no CLR, arcsinh, log, scaling, clipping, or PCA.")
  }
  expected_input_hash <- validate_sha256_string(scale$input_sha256, "input_scale.input_sha256")
  expected_marker_hash <- validate_sha256_string(scale$marker_file_sha256, "input_scale.marker_file_sha256")
  expected_record_hash <- validate_sha256_string(scale$preprocessing_record_sha256, "input_scale.preprocessing_record_sha256")
  actual_input_hash <- sha256_file(input_path)
  actual_marker_hash <- sha256_file(marker_path)
  if (!identical(actual_input_hash, expected_input_hash)) stopf("Input matrix SHA-256 does not match the benchmark configuration.")
  if (!identical(actual_marker_hash, expected_marker_hash)) stopf("Marker file SHA-256 does not match the benchmark configuration.")
  record_path <- resolve_existing_path(scale$preprocessing_record_file, dirname(config_path))
  if (!identical(sha256_file(record_path), expected_record_hash)) stopf("Preprocessing record SHA-256 does not match the benchmark configuration.")

  parameters <- validate_canonical_parameters(config$algorithm_parameters)
  config$algorithm_parameters <- parameters
  config$input_scale$preprocessing_record_file <- record_path
  list(config = config, preprocessing_record_path = record_path)
}

write_json <- function(object, path) {
  jsonlite::write_json(object, path, pretty = TRUE, auto_unbox = TRUE, null = "null", digits = 17)
}

arguments <- parse_cli(commandArgs(trailingOnly = TRUE))
for (package in c("jsonlite", "digest", "readr")) {
  if (!requireNamespace(package, quietly = TRUE)) stopf("Missing runner package: %s.", package)
}
validate_harmony_install(local_library)

input_path <- resolve_existing_path(arguments$input)
marker_path <- resolve_existing_path(arguments$markers)
config_path <- resolve_existing_path(arguments$config)
output_parent <- normalizePath(dirname(arguments$output), mustWork = TRUE)
output_path <- file.path(output_parent, basename(arguments$output))
if (file.exists(output_path) || dir.exists(output_path)) stopf("Output already exists: %s.", output_path)

config_raw <- jsonlite::read_json(config_path, simplifyVector = TRUE)
validated <- validate_config(config_raw, config_path, input_path, marker_path)
config <- validated$config
markers <- read_marker_file(marker_path)
input <- read_analysis_table(
  input_path,
  character_columns = c(config$cell_id_column, config$batch_column, config$sample_column)
)

fit <- fit_harmony_direct(
  input = input,
  markers = markers,
  cell_id_column = config$cell_id_column,
  batch_column = config$batch_column,
  sample_column = config$sample_column,
  batch_levels = config$batch_levels,
  parameters = config$algorithm_parameters,
  required_library = local_library
)

publish_outputs <- function() {
  stage_path <- file.path(
    output_parent,
    sprintf(".%s.staging-%d-%s", basename(output_path), Sys.getpid(), format(Sys.time(), "%Y%m%d%H%M%S"))
  )
  if (!dir.create(stage_path, recursive = FALSE, showWarnings = FALSE)) stopf("Could not create staging directory: %s.", stage_path)
  published <- FALSE
  on.exit({
    if (!published && dir.exists(stage_path)) unlink(stage_path, recursive = TRUE, force = TRUE)
  }, add = TRUE)

adjusted_table <- data.frame(
  cell_id = rownames(fit$adjusted_marker_coordinates),
  fit$adjusted_marker_coordinates,
  stringsAsFactors = FALSE,
  check.names = FALSE
)
readr::write_csv(adjusted_table, file.path(stage_path, "harmony_adjusted_marker_coordinates.csv.gz"))
if (identical(
    Sys.getenv("FINAL_XU_HARMONY_TEST_FAULT", unset = ""),
    "after_first_staged_output")) {
  stopf("Injected internal test fault after the first staged output.")
}
saveRDS(fit$adjusted_marker_coordinates, file.path(stage_path, "harmony_adjusted_marker_coordinates.rds"), version = 3)
readr::write_csv(fit$cell_metadata, file.path(stage_path, "cell_metadata_label_free.csv.gz"))
readr::write_csv(fit$marker_rank_audit, file.path(stage_path, "marker_rank_audit.csv"))
readr::write_csv(fit$marker_rank_audit_by_sample, file.path(stage_path, "marker_rank_audit_by_sample.csv"))
readr::write_csv(fit$marker_value_shift_audit, file.path(stage_path, "marker_value_shift_audit.csv"))

column_audit <- list(
  fitting_input_columns = fit$fitting_input_columns,
  provenance_only_input_columns = fit$provenance_only_input_columns,
  ignored_input_columns = fit$ignored_input_columns
)
write_json(column_audit, file.path(stage_path, "input_column_audit.json"))
write_json(config, file.path(stage_path, "resolved_config.json"))
write_json(
  list(
    method_label = HARMONY_DIRECT_METHOD_LABEL,
    interpretation = paste0(
      "Each output column is a Harmony-adjusted coordinate corresponding to the same input marker axis. ",
      "It is not a raw count and not canonical PCA-Harmony gene-expression correction."
    ),
    pca_applied = FALSE,
    algorithm_layer_transform = "none",
    clipping_or_inverse_transform = "none",
    negative_values_are_allowed = TRUE,
    artifact_claim_scope = "ADTnorm-style direct-marker Harmony only"
  ),
  file.path(stage_path, "method_semantics.json")
)
writeLines(
  c(
    "[messages]",
    if (length(fit$messages)) fit$messages else "<none>",
    "[warnings]",
    if (length(fit$warnings)) fit$warnings else "<none>"
  ),
  file.path(stage_path, "harmony_runtime_messages.txt")
)
writeLines(capture.output(sessionInfo()), file.path(stage_path, "sessionInfo.txt"))

output_files <- sort(list.files(stage_path, full.names = FALSE))
output_hashes <- lapply(output_files, function(file) list(file = file, sha256 = sha256_file(file.path(stage_path, file))))
manifest <- list(
  schema_version = HARMONY_DIRECT_SCHEMA,
  status = "complete",
  created_at = format(Sys.time(), tz = "UTC", usetz = TRUE),
  method_label = HARMONY_DIRECT_METHOD_LABEL,
  input = list(
    path = input_path,
    sha256 = sha256_file(input_path),
    n_cells = nrow(input),
    marker_file = marker_path,
    marker_file_sha256 = sha256_file(marker_path),
    markers = markers,
    preprocessing_record = validated$preprocessing_record_path,
    preprocessing_record_sha256 = sha256_file(validated$preprocessing_record_path),
    input_scale_name = config$input_scale$name,
    algorithm_applies_transformation = FALSE
  ),
  algorithm = list(
    package = "harmony",
    version = fit$installation$package_version,
    source_sha256 = fit$installation$source_sha256,
    resolved_parameters = fit$resolved_parameters,
    pca_applied = FALSE,
    seed_is_reproducibility_addition_not_adtnorm_parameter = TRUE
  ),
  implementation = list(
    wrapper_file = list(path = file.path(algorithm_dir, "harmony_direct_repro.R"), sha256 = sha256_file(file.path(algorithm_dir, "harmony_direct_repro.R"))),
    runner_file = list(path = script_path, sha256 = sha256_file(script_path)),
    upstream_lock = list(path = file.path(algorithm_dir, "upstream_lock.json"), sha256 = sha256_file(file.path(algorithm_dir, "upstream_lock.json"))),
    runtime_lock = list(path = file.path(algorithm_dir, "runtime_lock.csv"), sha256 = sha256_file(file.path(algorithm_dir, "runtime_lock.csv")))
  ),
  column_audit = column_audit,
  output_files = output_hashes,
  config_file = list(path = config_path, sha256 = sha256_file(config_path))
)
write_json(manifest, file.path(stage_path, "manifest.json"))
complete <- list(
  status = "complete",
  schema_version = HARMONY_DIRECT_SCHEMA,
  manifest_sha256 = sha256_file(file.path(stage_path, "manifest.json"))
)
write_json(complete, file.path(stage_path, "COMPLETE.json"))

  if (!file.rename(stage_path, output_path)) stopf("Atomic publish failed from %s to %s.", stage_path, output_path)
  published <- TRUE
  invisible(output_path)
}

publish_outputs()
cat(sprintf(
  "PASS: %s; %d cells x %d markers; %d batches; resolved nclust=%d; output=%s\n",
  HARMONY_DIRECT_METHOD_LABEL,
  nrow(input),
  length(markers),
  length(config$batch_levels),
  fit$resolved_parameters$resolved_nclust,
  output_path
))
