#!/usr/bin/env Rscript

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_argument)) stop("Cannot determine script location.", call. = FALSE)
script_path <- sub("^--file=", "", script_argument[[1L]])
script_path <- gsub("~+~", " ", script_path, fixed = TRUE)
script_path <- normalizePath(script_path, mustWork = TRUE)
algorithm_dir <- dirname(script_path)
local_library <- file.path(algorithm_dir, ".Rlib")
.libPaths(c(local_library, .libPaths()))
source(file.path(algorithm_dir, "fastmnn_repro.R"), local = FALSE)

usage <- function() {
  cat(paste0(
    "Usage:\n",
    "  Rscript run_fastmnn.R --input INPUT.csv.gz --markers markers.txt \\\n",
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
  if (!is.character(value) || length(value) != 1L || is.na(value) ||
      !grepl("^[0-9a-fA-F]{64}$", value)) {
    stopf("%s must be one 64-character hexadecimal SHA-256.", field)
  }
  tolower(value)
}

validate_preprocessing_record <- function(
    record_path,
    expected_input_hash,
    expected_marker_hash,
    batch_levels) {
  record <- tryCatch(
    jsonlite::read_json(record_path, simplifyVector = FALSE),
    error = function(error) {
      stopf("Preprocessing record is not valid JSON: %s.", conditionMessage(error))
    }
  )
  required <- c(
    "schema_version",
    "combined_input_sha256",
    "correction_marker_sha256",
    "shared_marker_selection_used_celltype_or_outcome_labels",
    "merge_order_selected_using_posthoc_labels_or_metrics",
    "canonical_parameters_selected_using_posthoc_labels_or_metrics",
    "dataset_order_is_progressive_merge_order",
    "datasets"
  )
  missing <- setdiff(required, names(record))
  if (length(missing)) {
    stopf(
      "Preprocessing record is missing required fields: %s.",
      paste(missing, collapse = ", ")
    )
  }
  if (!identical(record$schema_version, "final_xu.fastmnn_preprocessing.v1")) {
    stopf("Unsupported preprocessing-record schema_version.")
  }
  internal_input_hash <- validate_sha256_string(
    record$combined_input_sha256,
    "preprocessing_record.combined_input_sha256"
  )
  internal_marker_hash <- validate_sha256_string(
    record$correction_marker_sha256,
    "preprocessing_record.correction_marker_sha256"
  )
  if (!identical(internal_input_hash, expected_input_hash) ||
      !identical(internal_marker_hash, expected_marker_hash)) {
    stopf("Preprocessing record does not bind the configured input and marker file.")
  }
  label_declarations <- c(
    record$shared_marker_selection_used_celltype_or_outcome_labels,
    record$merge_order_selected_using_posthoc_labels_or_metrics,
    record$canonical_parameters_selected_using_posthoc_labels_or_metrics
  )
  if (length(label_declarations) != 3L ||
      !all(vapply(label_declarations, identical, logical(1), FALSE))) {
    stopf(
      paste0(
        "Canonical fastMNN requires explicit false declarations for label-based ",
        "marker selection, merge-order selection, and parameter selection."
      )
    )
  }
  if (!identical(record$dataset_order_is_progressive_merge_order, TRUE)) {
    stopf("Preprocessing record must bind dataset order to progressive merge order.")
  }
  if (!is.list(record$datasets) || !length(record$datasets)) {
    stopf("Preprocessing record datasets must be a non-empty JSON array.")
  }
  recorded_datasets <- vapply(record$datasets, function(dataset) {
    value <- dataset$dataset
    if (!is.character(value) || length(value) != 1L || is.na(value) || !nzchar(value)) {
      stopf("Every preprocessing-record dataset requires one non-empty dataset name.")
    }
    value
  }, character(1))
  if (!identical(recorded_datasets, batch_levels)) {
    stopf("Preprocessing-record dataset order differs from config batch_levels.")
  }
  invisible(record)
}

validate_config <- function(config, config_path, input_path, marker_path) {
  expected_fields <- c(
    "schema_version", "method_label", "cell_id_column", "batch_column",
    "sample_column", "batch_levels", "input_scale", "algorithm_parameters"
  )
  if (!is.list(config) || !setequal(names(config), expected_fields)) {
    stopf("Config fields do not exactly match the v1 schema.")
  }
  if (!identical(config$schema_version, FASTMNN_REPRO_SCHEMA)) {
    stopf("Unsupported config schema_version.")
  }
  if (!identical(config$method_label, FASTMNN_METHOD_LABEL)) {
    stopf("method_label must use the exact audited fastMNN name.")
  }
  scalar_columns <- c(config$cell_id_column, config$batch_column, config$sample_column)
  if (length(scalar_columns) != 3L || any(!vapply(
      scalar_columns,
      function(value) is.character(value) && length(value) == 1L &&
        !is.na(value) && nzchar(value),
      logical(1))) || anyDuplicated(scalar_columns)) {
    stopf("cell_id_column, batch_column, and sample_column must be distinct non-empty strings.")
  }
  if (!is.character(config$batch_levels) || length(config$batch_levels) < 2L ||
      anyNA(config$batch_levels) || any(!nzchar(config$batch_levels)) ||
      anyDuplicated(config$batch_levels)) {
    stopf("batch_levels must contain at least two ordered unique names.")
  }

  scale_fields <- c(
    "status", "name", "description", "preprocessing_record_file",
    "preprocessing_record_sha256", "input_sha256", "marker_file_sha256",
    "algorithm_internal_cosine_normalization",
    "algorithm_applies_other_expression_transform"
  )
  scale <- config$input_scale
  if (!is.list(scale) || !setequal(names(scale), scale_fields)) {
    stopf("input_scale fields do not exactly match the v1 schema.")
  }
  text_fields <- c(scale$status, scale$name, scale$description, scale$preprocessing_record_file)
  if (length(text_fields) != 4L || any(!vapply(
      text_fields,
      function(value) is.character(value) && length(value) == 1L &&
        !is.na(value) && nzchar(value),
      logical(1)))) {
    stopf("input_scale text fields must be non-empty strings.")
  }
  if (!identical(scale$status, "analysis_ready")) {
    stopf("input_scale.status must be 'analysis_ready'.")
  }
  if (!identical(scale$algorithm_internal_cosine_normalization, TRUE) ||
      !identical(scale$algorithm_applies_other_expression_transform, FALSE)) {
    stopf(
      paste0(
        "The canonical method performs official cosine normalization but no ",
        "other expression transform in the algorithm layer."
      )
    )
  }

  expected_input_hash <- validate_sha256_string(scale$input_sha256, "input_scale.input_sha256")
  expected_marker_hash <- validate_sha256_string(
    scale$marker_file_sha256, "input_scale.marker_file_sha256"
  )
  expected_record_hash <- validate_sha256_string(
    scale$preprocessing_record_sha256,
    "input_scale.preprocessing_record_sha256"
  )
  if (!identical(sha256_file(input_path), expected_input_hash)) {
    stopf("Input SHA-256 does not match the benchmark configuration.")
  }
  if (!identical(sha256_file(marker_path), expected_marker_hash)) {
    stopf("Marker-file SHA-256 does not match the benchmark configuration.")
  }
  preprocessing_record_path <- resolve_existing_path(
    scale$preprocessing_record_file,
    dirname(config_path)
  )
  if (!identical(sha256_file(preprocessing_record_path), expected_record_hash)) {
    stopf("Preprocessing-record SHA-256 does not match the benchmark configuration.")
  }
  validate_preprocessing_record(
    preprocessing_record_path,
    expected_input_hash,
    expected_marker_hash,
    config$batch_levels
  )

  config$algorithm_parameters <- validate_canonical_parameters(config$algorithm_parameters)
  config$input_scale$preprocessing_record_file <- preprocessing_record_path
  list(
    config = config,
    preprocessing_record_path = preprocessing_record_path,
    expected_artifact_hashes = c(
      input = expected_input_hash,
      marker_file = expected_marker_hash,
      preprocessing_record = expected_record_hash
    )
  )
}

write_json <- function(object, path) {
  jsonlite::write_json(
    object,
    path,
    pretty = TRUE,
    auto_unbox = TRUE,
    null = "null",
    digits = 17
  )
}

arguments <- parse_cli(commandArgs(trailingOnly = TRUE))
for (package in c("jsonlite", "digest", "readr")) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stopf("Missing runner package: %s.", package)
  }
}

input_path <- resolve_existing_path(arguments$input)
marker_path <- resolve_existing_path(arguments$markers)
config_path <- resolve_existing_path(arguments$config)
output_parent <- normalizePath(dirname(arguments$output), mustWork = TRUE)
output_path <- file.path(output_parent, basename(arguments$output))
if (file.exists(output_path) || dir.exists(output_path)) {
  stopf("Output already exists: %s.", output_path)
}

initial_config_sha256 <- sha256_file(config_path)
config_raw <- jsonlite::read_json(config_path, simplifyVector = TRUE)
if (!identical(sha256_file(config_path), initial_config_sha256)) {
  stopf("Config file changed while it was being read.")
}
validated <- validate_config(config_raw, config_path, input_path, marker_path)
config <- validated$config
artifact_paths <- c(
  input = input_path,
  marker_file = marker_path,
  preprocessing_record = validated$preprocessing_record_path,
  config = config_path
)
expected_artifact_hashes <- c(
  validated$expected_artifact_hashes,
  config = initial_config_sha256
)
assert_artifacts_unchanged <- function(context) {
  actual <- vapply(artifact_paths, sha256_file, character(1))
  changed <- names(actual)[actual != expected_artifact_hashes]
  if (length(changed)) {
    stopf(
      "Authenticated artifact(s) changed during %s: %s.",
      context, paste(changed, collapse = ", ")
    )
  }
  invisible(TRUE)
}
markers <- read_marker_file(marker_path)
input <- read_analysis_table(
  input_path,
  character_columns = c(
    config$cell_id_column, config$batch_column, config$sample_column
  )
)
assert_artifacts_unchanged("input parsing")

fit <- fit_fastmnn_official(
  input = input,
  markers = markers,
  cell_id_column = config$cell_id_column,
  batch_column = config$batch_column,
  sample_column = config$sample_column,
  batch_levels = config$batch_levels,
  parameters = config$algorithm_parameters,
  required_library = local_library
)
assert_artifacts_unchanged("model fitting")

publish_outputs <- function() {
  stage_path <- file.path(
    output_parent,
    sprintf(
      ".%s.staging-%d-%s",
      basename(output_path), Sys.getpid(), format(Sys.time(), "%Y%m%d%H%M%S")
    )
  )
  if (!dir.create(stage_path, recursive = FALSE, showWarnings = FALSE)) {
    stopf("Could not create staging directory: %s.", stage_path)
  }
  published <- FALSE
  on.exit({
    if (!published && dir.exists(stage_path)) {
      unlink(stage_path, recursive = TRUE, force = TRUE)
    }
  }, add = TRUE)

  embedding_table <- data.frame(
    cell_id = rownames(fit$corrected_embedding),
    fit$corrected_embedding,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  readr::write_csv(
    embedding_table,
    file.path(stage_path, "fastmnn_corrected_embedding.csv.gz")
  )
  if (identical(
      Sys.getenv("FINAL_XU_FASTMNN_TEST_FAULT", unset = ""),
      "after_first_staged_output")) {
    stopf("Injected internal test fault after the first staged output.")
  }
  saveRDS(
    fit$corrected_embedding,
    file.path(stage_path, "fastmnn_corrected_embedding.rds"),
    version = 3
  )
  readr::write_csv(
    fit$cell_metadata,
    file.path(stage_path, "cell_metadata_label_free.csv.gz")
  )
  readr::write_csv(
    fit$merge$steps,
    file.path(stage_path, "fastmnn_merge_diagnostics.csv")
  )
  readr::write_csv(
    fit$merge$lost_variance,
    file.path(stage_path, "fastmnn_lost_variance.csv")
  )
  saveRDS(
    fit$merge$raw,
    file.path(stage_path, "fastmnn_merge_info.rds"),
    version = 3
  )
  readr::write_csv(
    fit$pca$variance,
    file.path(stage_path, "fastmnn_pca_variance.csv")
  )
  rotation_table <- data.frame(
    marker = rownames(fit$pca$rotation),
    fit$pca$rotation,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  readr::write_csv(
    rotation_table,
    file.path(stage_path, "fastmnn_pca_rotation.csv.gz")
  )
  readr::write_csv(
    fit$pca$centers,
    file.path(stage_path, "fastmnn_pca_centers.csv")
  )
  readr::write_csv(
    fit$marker_batch_statistics,
    file.path(stage_path, "input_marker_batch_statistics.csv")
  )
  readr::write_csv(
    fit$cell_l2_statistics,
    file.path(stage_path, "input_cell_l2_statistics.csv")
  )
  preprocessing_snapshot_path <- file.path(
    stage_path, "authenticated_preprocessing_manifest.json"
  )
  copied_preprocessing_record <- file.copy(
    validated$preprocessing_record_path,
    preprocessing_snapshot_path,
    overwrite = FALSE
  )
  if (!isTRUE(copied_preprocessing_record)) {
    stopf("Could not snapshot the authenticated preprocessing manifest.")
  }
  if (!identical(
      sha256_file(preprocessing_snapshot_path),
      expected_artifact_hashes[["preprocessing_record"]])) {
    stopf("Preprocessing-manifest snapshot differs from its authenticated source.")
  }

  column_audit <- list(
    fitting_input_columns = fit$fitting_input_columns,
    provenance_only_input_columns = fit$provenance_only_input_columns,
    ignored_input_columns = fit$ignored_input_columns
  )
  write_json(column_audit, file.path(stage_path, "input_column_audit.json"))
  write_json(config, file.path(stage_path, "resolved_config.json"))
  write_json(
    list(
      method_label = FASTMNN_METHOD_LABEL,
      primary_output = "reducedDim(result, 'corrected')",
      output_type = "cell_by_latent_dimension_corrected_embedding",
      marker_level_interpretation = "not_applicable",
      reconstructed_assay_exported = FALSE,
      reconstructed_assay_reason = paste0(
        "The official manual describes it as a low-rank reconstruction after ",
        "cosine normalization and advises against quantitative analysis."
      ),
      standard_fastmnn_pca_used = TRUE,
      marker_space_d_NA_used = FALSE,
      requested_d = fit$pca$requested_d,
      resolved_d = fit$pca$resolved_d,
      actual_d = fit$pca$actual_d,
      cosine_normalization = TRUE,
      upstream_assay_transform_in_algorithm_layer = "none",
      merge_policy = "official progressive merge in configured batch_levels order",
      cell_type_labels_used_for_fit = FALSE
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
    file.path(stage_path, "fastmnn_runtime_messages.txt")
  )
  writeLines(
    capture.output(sessionInfo()),
    file.path(stage_path, "sessionInfo.txt")
  )

  output_files <- sort(list.files(stage_path, full.names = FALSE))
  output_hashes <- lapply(output_files, function(file) {
    list(file = file, sha256 = sha256_file(file.path(stage_path, file)))
  })
  manifest <- list(
    schema_version = FASTMNN_REPRO_SCHEMA,
    status = "complete",
    created_at = format(Sys.time(), tz = "UTC", usetz = TRUE),
    method_label = FASTMNN_METHOD_LABEL,
    input = list(
      path = input_path,
      sha256 = expected_artifact_hashes[["input"]],
      n_cells = nrow(input),
      marker_file = marker_path,
      marker_file_sha256 = expected_artifact_hashes[["marker_file"]],
      n_markers = length(markers),
      markers = markers,
      batch_levels_in_progressive_merge_order = fit$batch_levels,
      preprocessing_record = validated$preprocessing_record_path,
      preprocessing_record_sha256 = expected_artifact_hashes[["preprocessing_record"]],
      input_scale_name = config$input_scale$name
    ),
    algorithm = list(
      package = fit$installation$package,
      version = fit$installation$package_version,
      source_sha256 = fit$installation$source_sha256,
      upstream_commit = fit$installation$upstream_commit,
      resolved_parameters = fit$parameters,
      requested_d = fit$pca$requested_d,
      resolved_d = fit$pca$resolved_d,
      actual_d = fit$pca$actual_d,
      reconstructed_assay_exported = FALSE
    ),
    implementation = list(
      wrapper_file = list(
        path = file.path(algorithm_dir, "fastmnn_repro.R"),
        sha256 = sha256_file(file.path(algorithm_dir, "fastmnn_repro.R"))
      ),
      runner_file = list(path = script_path, sha256 = sha256_file(script_path)),
      upstream_lock = list(
        path = file.path(algorithm_dir, "upstream_lock.json"),
        sha256 = sha256_file(file.path(algorithm_dir, "upstream_lock.json"))
      ),
      runtime_lock = list(
        path = file.path(algorithm_dir, "runtime_lock.csv"),
        sha256 = sha256_file(file.path(algorithm_dir, "runtime_lock.csv"))
      )
    ),
    column_audit = column_audit,
    diagnostics = list(
      merge_steps = nrow(fit$merge$steps),
      any_lost_variance_over_10_percent = any(
        fit$merge$lost_variance$review_flag_over_10_percent
      ),
      total_variance = fit$pca$total_variance
    ),
    output_files = output_hashes,
    config_file = list(path = config_path, sha256 = expected_artifact_hashes[["config"]])
  )
  write_json(manifest, file.path(stage_path, "manifest.json"))
  write_json(
    list(
      status = "complete",
      schema_version = FASTMNN_REPRO_SCHEMA,
      manifest_sha256 = sha256_file(file.path(stage_path, "manifest.json"))
    ),
    file.path(stage_path, "COMPLETE.json")
  )

  assert_artifacts_unchanged("atomic publication")
  if (identical(
      Sys.getenv("FINAL_XU_FASTMNN_TEST_FAULT", unset = ""),
      "before_atomic_publish")) {
    stopf("Injected internal test fault before atomic publication.")
  }

  if (!file.rename(stage_path, output_path)) {
    stopf("Atomic publish failed from %s to %s.", stage_path, output_path)
  }
  published <- TRUE
  invisible(output_path)
}

publish_outputs()
cat(sprintf(
  paste0(
    "PASS: %s; %d cells x %d markers; %d batches; requested d=%d, ",
    "resolved/actual d=%d/%d; output=%s\n"
  ),
  FASTMNN_METHOD_LABEL,
  nrow(input),
  length(markers),
  length(config$batch_levels),
  fit$pca$requested_d,
  fit$pca$resolved_d,
  fit$pca$actual_d,
  output_path
))
