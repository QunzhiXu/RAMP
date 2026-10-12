#!/usr/bin/env Rscript

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_argument)) stop("Cannot determine script location.", call. = FALSE)
script_path <- sub("^--file=", "", script_argument[[1L]])
script_path <- gsub("~+~", " ", script_path, fixed = TRUE)
script_path <- normalizePath(script_path, mustWork = TRUE)
algorithm_dir <- dirname(script_path)
local_library <- file.path(algorithm_dir, ".Rlib")
.libPaths(c(local_library, .libPaths()))

source(file.path(algorithm_dir, "cycombine_repro.R"), local = FALSE)

usage <- function() {
  cat(paste0(
    "Usage:\n",
    "  Rscript run_cycombine.R --input INPUT.csv.gz --markers markers.txt \\\n",
    "    --config config.json --output OUTPUT_DIRECTORY\n"
  ))
}

parse_cli <- function(arguments) {
  if (!length(arguments) || any(arguments %in% c("-h", "--help"))) {
    usage()
    quit(status = if (length(arguments)) 0L else 2L)
  }
  if (length(arguments) %% 2L != 0L) stop("Every CLI flag requires one value.", call. = FALSE)
  allowed <- c("--input", "--markers", "--config", "--output")
  keys <- arguments[seq(1L, length(arguments), by = 2L)]
  values <- arguments[seq(2L, length(arguments), by = 2L)]
  if (any(!keys %in% allowed) || anyDuplicated(keys)) {
    stop(sprintf("Unknown or repeated CLI flags: %s", paste(keys, collapse = ", ")), call. = FALSE)
  }
  missing <- setdiff(allowed, keys)
  if (length(missing)) stop(sprintf("Missing CLI flags: %s", paste(missing, collapse = ", ")), call. = FALSE)
  result <- as.list(values)
  names(result) <- sub("^--", "", keys)
  result
}

resolve_relative_path <- function(path, base_directory) {
  if (!is.character(path) || length(path) != 1L || !nzchar(path)) {
    stop("Configured path must be one non-empty string.", call. = FALSE)
  }
  candidate <- if (grepl("^/", path)) path else file.path(base_directory, path)
  normalizePath(candidate, mustWork = TRUE)
}

hash_file <- function(path) {
  digest::digest(file = path, algo = "sha256", serialize = FALSE)
}

validate_preprocessing_manifest <- function(
    path,
    batch_levels,
    combined_input_path,
    combined_input,
    marker_path,
    markers,
    cell_id_column,
    batch_column,
    sample_column) {
  manifest <- jsonlite::read_json(path, simplifyVector = FALSE)
  expected_fields <- c(
    "schema_version", "combined_input_sha256", "shared_marker_policy",
    "correction_marker_file", "correction_marker_sha256",
    "marker_harmonization_manifest_file", "marker_harmonization_manifest_sha256",
    "cell_selection_manifest_file", "cell_selection_manifest_sha256",
    "cell_selection_id_column", "datasets"
  )
  if (!is.list(manifest) || !setequal(names(manifest), expected_fields)) {
    stop("Preprocessing manifest fields do not exactly match the v2 schema.", call. = FALSE)
  }
  header_text_fields <- c(
    manifest$shared_marker_policy,
    manifest$correction_marker_file,
    manifest$marker_harmonization_manifest_file,
    manifest$cell_selection_manifest_file,
    manifest$cell_selection_id_column
  )
  header_hash_fields <- c(
    manifest$combined_input_sha256,
    manifest$correction_marker_sha256,
    manifest$marker_harmonization_manifest_sha256,
    manifest$cell_selection_manifest_sha256
  )
  if (!identical(manifest$schema_version, "final_xu.cycombine_preprocessing.v2") ||
      !is.character(manifest$combined_input_sha256) || length(manifest$combined_input_sha256) != 1L ||
      length(header_text_fields) != 5L || length(header_hash_fields) != 4L ||
      any(!vapply(header_text_fields, function(x) is.character(x) && length(x) == 1L && nzchar(x), logical(1))) ||
      any(!vapply(header_hash_fields, function(x) is.character(x) && length(x) == 1L && grepl("^[0-9a-fA-F]{64}$", x), logical(1))) ||
      !is.list(manifest$datasets) || length(manifest$datasets) != length(batch_levels)) {
    stop("Preprocessing manifest header is invalid or has the wrong number of datasets.", call. = FALSE)
  }
  actual_combined_sha256 <- hash_file(combined_input_path)
  if (!identical(tolower(actual_combined_sha256), tolower(manifest$combined_input_sha256))) {
    stop(sprintf(
      "Combined --input hash mismatch: preprocessing manifest expects %s, found %s.",
      manifest$combined_input_sha256, actual_combined_sha256
    ), call. = FALSE)
  }
  required_columns <- c(cell_id_column, batch_column, sample_column, markers)
  missing_combined_columns <- setdiff(required_columns, colnames(combined_input))
  if (length(missing_combined_columns)) {
    stop(sprintf(
      "Combined input is missing required columns: %s",
      paste(missing_combined_columns, collapse = ", ")
    ), call. = FALSE)
  }

  correction_marker_path <- resolve_relative_path(
    manifest$correction_marker_file,
    dirname(path)
  )
  actual_correction_marker_sha256 <- hash_file(correction_marker_path)
  if (!identical(tolower(actual_correction_marker_sha256), tolower(manifest$correction_marker_sha256)) ||
      !identical(tolower(actual_correction_marker_sha256), tolower(hash_file(marker_path))) ||
      !identical(read_marker_file(correction_marker_path), markers)) {
    stop("Correction marker file does not exactly match the CLI marker file and manifest hash.", call. = FALSE)
  }

  marker_harmonization_path <- resolve_relative_path(
    manifest$marker_harmonization_manifest_file,
    dirname(path)
  )
  actual_marker_harmonization_sha256 <- hash_file(marker_harmonization_path)
  if (!identical(
      tolower(actual_marker_harmonization_sha256),
      tolower(manifest$marker_harmonization_manifest_sha256))) {
    stop("Marker-harmonization manifest hash mismatch.", call. = FALSE)
  }

  cell_selection_path <- resolve_relative_path(
    manifest$cell_selection_manifest_file,
    dirname(path)
  )
  actual_cell_selection_sha256 <- hash_file(cell_selection_path)
  if (!identical(tolower(actual_cell_selection_sha256), tolower(manifest$cell_selection_manifest_sha256))) {
    stop("Cell-selection manifest hash mismatch.", call. = FALSE)
  }
  cell_selection <- read_analysis_table(
    cell_selection_path,
    character_columns = manifest$cell_selection_id_column
  )
  if (!manifest$cell_selection_id_column %in% colnames(cell_selection)) {
    stop("Cell-selection manifest is missing its configured cell ID column.", call. = FALSE)
  }
  selected_ids <- as.character(cell_selection[[manifest$cell_selection_id_column]])
  combined_ids <- as.character(combined_input[[cell_id_column]])
  if (anyNA(selected_ids) || any(!nzchar(selected_ids)) || anyDuplicated(selected_ids)) {
    stop("Cell-selection manifest cell IDs must be complete and unique.", call. = FALSE)
  }
  if (!identical(selected_ids, combined_ids)) {
    stop(
      "Cell-selection manifest IDs and order do not exactly match the combined cyCombine input.",
      call. = FALSE
    )
  }

  dataset_fields <- c(
    "dataset", "technology", "input_matrix_file", "input_matrix_sha256",
    "source_state", "assay_or_layer", "processing_steps", "cycombine_input_scale",
    "transform_formula", "transform_scope", "processing_reference",
    "processing_code_file", "processing_code_sha256", "double_transform_checked"
  )
  allowed_source_states <- c(
    "raw_or_unprocessed",
    "technically_normalized_untransformed",
    "nonlinearly_transformed",
    "standardized",
    "normalized_and_nonlinearly_transformed",
    "provider_analysis_scale"
  )
  dataset_names <- character(length(manifest$datasets))
  input_files <- vector("list", length(manifest$datasets))
  dataset_tables <- vector("list", length(manifest$datasets))
  for (index in seq_along(manifest$datasets)) {
    entry <- manifest$datasets[[index]]
    if (!is.list(entry) || !setequal(names(entry), dataset_fields)) {
      stop(sprintf("Preprocessing manifest dataset entry %d has invalid fields.", index), call. = FALSE)
    }
    scalar_text <- c(
      entry$dataset, entry$technology, entry$input_matrix_file,
      entry$input_matrix_sha256, entry$source_state, entry$assay_or_layer,
      entry$cycombine_input_scale, entry$transform_formula, entry$transform_scope,
      entry$processing_reference, entry$processing_code_file,
      entry$processing_code_sha256
    )
    if (length(scalar_text) != 12L || any(!vapply(scalar_text, is.character, logical(1))) ||
        any(!nzchar(unlist(scalar_text, use.names = FALSE)))) {
      stop(sprintf("Preprocessing manifest dataset entry %d has missing text fields.", index), call. = FALSE)
    }
    if (!grepl("^[0-9a-fA-F]{64}$", entry$input_matrix_sha256)) {
      stop(sprintf("Preprocessing manifest dataset entry %d has an invalid input_matrix_sha256.", index), call. = FALSE)
    }
    resolved_input <- resolve_relative_path(entry$input_matrix_file, dirname(path))
    actual_input_sha256 <- hash_file(resolved_input)
    if (!identical(tolower(actual_input_sha256), tolower(entry$input_matrix_sha256))) {
      stop(sprintf(
        "Preprocessing manifest dataset entry %d input hash mismatch: expected %s, found %s.",
        index, entry$input_matrix_sha256, actual_input_sha256
      ), call. = FALSE)
    }
    if (!entry$source_state %in% allowed_source_states) {
      stop(sprintf(
        "Preprocessing manifest dataset entry %d has unsupported source_state '%s'.",
        index, entry$source_state
      ), call. = FALSE)
    }
    if (entry$source_state %in% c("raw_or_unprocessed", "technically_normalized_untransformed") &&
        tolower(trimws(entry$transform_formula)) %in% c("none", "passthrough", "not applied")) {
      stop(sprintf(
        paste0(
          "Dataset entry %d is declared untransformed but records no transformation. ",
          "Technical normalization alone is not evidence of an analysis-scale transform."
        ),
        index
      ), call. = FALSE)
    }
    if (!is.list(entry$processing_steps) || !length(entry$processing_steps) ||
        any(!vapply(entry$processing_steps, function(x) is.character(x) && length(x) == 1L && nzchar(x), logical(1)))) {
      stop(sprintf("Preprocessing manifest dataset entry %d needs at least one explicit processing step.", index), call. = FALSE)
    }
    if (!identical(entry$double_transform_checked, TRUE)) {
      stop(sprintf("Preprocessing manifest dataset entry %d must confirm double_transform_checked=true.", index), call. = FALSE)
    }

    resolved_processing_code <- resolve_relative_path(entry$processing_code_file, dirname(path))
    actual_processing_code_sha256 <- hash_file(resolved_processing_code)
    if (!grepl("^[0-9a-fA-F]{64}$", entry$processing_code_sha256) ||
        !identical(tolower(actual_processing_code_sha256), tolower(entry$processing_code_sha256))) {
      stop(sprintf("Preprocessing code hash mismatch for dataset entry %d.", index), call. = FALSE)
    }

    dataset_table <- read_analysis_table(
      resolved_input,
      character_columns = c(cell_id_column, batch_column, sample_column)
    )
    missing_columns <- setdiff(required_columns, colnames(dataset_table))
    if (length(missing_columns)) {
      stop(sprintf(
        "Dataset entry %d analysis-ready matrix is missing columns: %s",
        index, paste(missing_columns, collapse = ", ")
      ), call. = FALSE)
    }
    if (any(as.character(dataset_table[[batch_column]]) != entry$dataset)) {
      stop(sprintf("Dataset entry %d contains rows assigned to another batch.", index), call. = FALSE)
    }

    dataset_names[index] <- entry$dataset
    manifest$datasets[[index]]$input_matrix_file <- resolved_input
    manifest$datasets[[index]]$processing_code_file <- resolved_processing_code
    dataset_tables[[index]] <- dataset_table[, required_columns, drop = FALSE]
    input_files[[index]] <- list(
      dataset = entry$dataset,
      path = resolved_input,
      sha256 = actual_input_sha256,
      rows = nrow(dataset_table),
      processing_code = list(
        path = resolved_processing_code,
        sha256 = actual_processing_code_sha256
      )
    )
  }
  if (!identical(dataset_names, batch_levels)) {
    stop(
      sprintf(
        "Preprocessing manifest dataset order [%s] must exactly match batch_levels [%s].",
        paste(dataset_names, collapse = ", "), paste(batch_levels, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  reconstructed <- do.call(rbind, dataset_tables)
  rownames(reconstructed) <- NULL
  combined_required <- combined_input[, required_columns, drop = FALSE]
  rownames(combined_required) <- NULL
  if (nrow(reconstructed) != nrow(combined_required)) {
    stop("Per-dataset matrices do not have the same total rows as the combined input.", call. = FALSE)
  }
  for (column in required_columns) {
    left <- reconstructed[[column]]
    right <- combined_required[[column]]
    equal <- if (column %in% markers) {
      identical(as.numeric(left), as.numeric(right))
    } else {
      identical(as.character(left), as.character(right))
    }
    if (!equal) {
      stop(sprintf(
        "Ordered per-dataset row-bind does not exactly reproduce combined input column '%s'.",
        column
      ), call. = FALSE)
    }
  }

  manifest$correction_marker_file <- correction_marker_path
  manifest$marker_harmonization_manifest_file <- marker_harmonization_path
  manifest$cell_selection_manifest_file <- cell_selection_path
  list(
    manifest = manifest,
    combined_input = list(path = combined_input_path, sha256 = actual_combined_sha256),
    input_files = input_files,
    correction_marker_file = list(
      path = correction_marker_path,
      sha256 = actual_correction_marker_sha256,
      exact_cli_marker_match_verified = TRUE
    ),
    marker_harmonization_manifest = list(
      path = marker_harmonization_path,
      sha256 = actual_marker_harmonization_sha256
    ),
    cell_selection_manifest = list(
      path = cell_selection_path,
      sha256 = actual_cell_selection_sha256,
      rows = nrow(cell_selection),
      exact_id_order_verified = TRUE
    ),
    exact_dataset_binding_verified = TRUE
  )
}

write_csv_gz <- function(data, path) {
  readr::write_csv(data, path, na = "")
}

arguments <- parse_cli(commandArgs(trailingOnly = TRUE))
input_path <- normalizePath(arguments$input, mustWork = TRUE)
marker_path <- normalizePath(arguments$markers, mustWork = TRUE)
config_path <- normalizePath(arguments$config, mustWork = TRUE)
output_path <- normalizePath(arguments$output, mustWork = FALSE)

if (!requireNamespace("jsonlite", quietly = TRUE) ||
    !requireNamespace("digest", quietly = TRUE) ||
    !requireNamespace("readr", quietly = TRUE)) {
  stop("jsonlite, digest, and readr are required by the reproducibility runner.", call. = FALSE)
}

config <- jsonlite::read_json(config_path, simplifyVector = TRUE)
if (!identical(config$schema_version, CYCOMBINE_REPRO_SCHEMA)) {
  stop(sprintf("config schema_version must be %s.", CYCOMBINE_REPRO_SCHEMA), call. = FALSE)
}
required_config_fields <- c(
  "schema_version", "cell_id_column", "batch_column", "sample_column",
  "batch_levels", "input_scale", "algorithm_parameters"
)
if (!setequal(names(config), required_config_fields)) {
  stop("Config fields do not exactly match the v1 schema.", call. = FALSE)
}
if (!is.list(config$input_scale) ||
    !setequal(names(config$input_scale), c("status", "description", "provenance_file", "algorithm_applies_transformation"))) {
  stop("input_scale must contain status, description, provenance_file, and algorithm_applies_transformation.", call. = FALSE)
}
if (!identical(config$input_scale$status, "analysis_ready") ||
    !identical(config$input_scale$algorithm_applies_transformation, FALSE) ||
    !is.character(config$input_scale$description) ||
    length(config$input_scale$description) != 1L ||
    !nzchar(config$input_scale$description)) {
  stop("Input must be explicitly documented as analysis_ready and the algorithm must not apply a transform.", call. = FALSE)
}

config_directory <- dirname(config_path)
preprocessing_provenance <- resolve_relative_path(config$input_scale$provenance_file, config_directory)
parameters <- validate_canonical_parameters(config$algorithm_parameters)
markers <- read_marker_file(marker_path)
input <- read_analysis_table(
  input_path,
  character_columns = c(config$cell_id_column, config$batch_column, config$sample_column)
)
preprocessing_validation <- validate_preprocessing_manifest(
  path = preprocessing_provenance,
  batch_levels = as.character(config$batch_levels),
  combined_input_path = input_path,
  combined_input = input,
  marker_path = marker_path,
  markers = markers,
  cell_id_column = config$cell_id_column,
  batch_column = config$batch_column,
  sample_column = config$sample_column
)
preprocessing_manifest <- preprocessing_validation$manifest

if (file.exists(output_path) || dir.exists(output_path)) {
  stop(sprintf("Output path already exists; choose a new directory: %s", output_path), call. = FALSE)
}
output_parent <- dirname(output_path)
if (!dir.exists(output_parent) && !dir.create(output_parent, recursive = TRUE, showWarnings = FALSE)) {
  stop(sprintf("Could not create output parent directory: %s", output_parent), call. = FALSE)
}

fit_messages <- character()
fit_warnings <- character()
result <- withCallingHandlers(
  fit_cycombine_publication(
    input = input,
    markers = markers,
    cell_id_column = config$cell_id_column,
    batch_column = config$batch_column,
    sample_column = config$sample_column,
    batch_levels = as.character(config$batch_levels),
    parameters = parameters,
    required_library = local_library
  ),
  message = function(condition) {
    fit_messages <<- c(fit_messages, conditionMessage(condition))
    invokeRestart("muffleMessage")
  },
  warning = function(condition) {
    fit_warnings <<- c(fit_warnings, conditionMessage(condition))
    invokeRestart("muffleWarning")
  }
)

staging_path <- tempfile(
  pattern = paste0(".", basename(output_path), ".staging-"),
  tmpdir = output_parent
)
if (!dir.create(staging_path, recursive = FALSE, showWarnings = FALSE)) {
  stop(sprintf("Could not create staging directory: %s", staging_path), call. = FALSE)
}
on.exit({
  if (!is.null(staging_path) && dir.exists(staging_path)) {
    unlink(staging_path, recursive = TRUE, force = TRUE)
  }
}, add = TRUE)

corrected_path <- file.path(staging_path, "corrected_expression.csv.gz")
corrected_rds_path <- file.path(staging_path, "corrected_expression.rds")
metadata_path <- file.path(staging_path, "cell_metadata_label_free.csv.gz")
som_path <- file.path(staging_path, "correction_som_assignments.csv.gz")
node_counts_path <- file.path(staging_path, "correction_som_node_batch_counts.csv")
node_modes_path <- file.path(staging_path, "correction_som_node_modes.csv")
input_stats_path <- file.path(staging_path, "input_marker_batch_statistics.csv")
resolved_config_path <- file.path(staging_path, "resolved_config.json")
resolved_preprocessing_path <- file.path(staging_path, "resolved_preprocessing_manifest.json")
session_path <- file.path(staging_path, "sessionInfo.txt")
run_log_path <- file.path(staging_path, "cycombine_messages.log")

corrected_table <- data.frame(cell_id = result$cell_ids, result$corrected_expression, check.names = FALSE)
metadata_table <- data.frame(
  cell_id = result$cell_ids,
  batch = result$batches,
  sample = result$samples,
  stringsAsFactors = FALSE
)
som_table <- data.frame(cell_id = result$cell_ids, som_node = result$som_labels, stringsAsFactors = FALSE)

write_csv_gz(corrected_table, corrected_path)
saveRDS(corrected_table, corrected_rds_path, version = 3)
write_csv_gz(metadata_table, metadata_path)
write_csv_gz(som_table, som_path)
readr::write_csv(result$node_batch_counts, node_counts_path)
readr::write_csv(result$node_correction_modes, node_modes_path)
readr::write_csv(result$marker_batch_statistics, input_stats_path)

resolved_config <- config
resolved_config$input_scale$provenance_file <- preprocessing_provenance
resolved_config$resolved_input <- input_path
resolved_config$resolved_markers <- marker_path
resolved_config$resolved_package_library <- normalizePath(local_library, mustWork = TRUE)
jsonlite::write_json(
  resolved_config,
  resolved_config_path,
  pretty = TRUE,
  auto_unbox = TRUE,
  null = "null",
  digits = NA
)
jsonlite::write_json(
  preprocessing_manifest,
  resolved_preprocessing_path,
  pretty = TRUE,
  auto_unbox = TRUE,
  null = "null",
  digits = NA
)
writeLines(capture.output(sessionInfo()), session_path)
writeLines(c(
  "[messages]",
  if (length(fit_messages)) fit_messages else "(none)",
  "",
  "[warnings]",
  if (length(fit_warnings)) fit_warnings else "(none)"
), run_log_path)

output_files <- c(
  corrected_expression_csv = corrected_path,
  corrected_expression_rds = corrected_rds_path,
  label_free_metadata = metadata_path,
  correction_som_assignments = som_path,
  correction_som_node_batch_counts = node_counts_path,
  correction_som_node_modes = node_modes_path,
  input_marker_batch_statistics = input_stats_path,
  resolved_config = resolved_config_path,
  resolved_preprocessing_manifest = resolved_preprocessing_path,
  session_info = session_path,
  cycombine_messages = run_log_path
)
output_hashes <- lapply(output_files, function(path) list(path = basename(path), sha256 = hash_file(path)))

mode_levels <- c(
  "empty", "single_batch_passthrough", "multi_batch_mean_only_combat",
  "multi_batch_full_combat"
)
mode_counts <- table(factor(result$node_correction_modes$correction_mode, levels = mode_levels))
mode_counts <- as.list(stats::setNames(as.integer(mode_counts), mode_levels))
manifest <- list(
  schema_version = "final_xu.cycombine_run_manifest.v2",
  created_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  method = "cyCombine v0.2.5 official cross-platform vignette settings",
  upstream = c(result$installation_provenance, list(
    publication_commit = CYCOMBINE_PUBLICATION_COMMIT,
    cross_platform_vignette_url = "https://biosurf.org/cyCombine_CITEseq_Spectral_CyTOF.html",
    cross_platform_vignette_sha256 = CYCOMBINE_CROSS_PLATFORM_VIGNETTE_SHA256
  )),
  inputs = list(
    expression = list(path = input_path, sha256 = hash_file(input_path)),
    markers = list(path = marker_path, sha256 = hash_file(marker_path)),
    config = list(path = config_path, sha256 = hash_file(config_path)),
    preprocessing_provenance = list(path = preprocessing_provenance, sha256 = hash_file(preprocessing_provenance))
  ),
  preprocessing_manifest_schema = preprocessing_manifest$schema_version,
  preprocessing_combined_input = preprocessing_validation$combined_input,
  preprocessing_dataset_inputs = preprocessing_validation$input_files,
  correction_marker_file = preprocessing_validation$correction_marker_file,
  marker_harmonization_manifest = preprocessing_validation$marker_harmonization_manifest,
  cell_selection_manifest = preprocessing_validation$cell_selection_manifest,
  exact_dataset_binding_verified = preprocessing_validation$exact_dataset_binding_verified,
  dimensions = list(cells = nrow(input), markers = length(markers), batches = length(config$batch_levels)),
  batch_levels = as.character(config$batch_levels),
  markers = markers,
  fitting_input_columns = result$fitting_input_columns,
  provenance_only_input_columns = result$provenance_only_input_columns,
  ignored_input_columns = result$ignored_input_columns,
  algorithm_parameters = result$parameters,
  rng = result$rng,
  som_audit = list(
    grid_nodes = parameters$xdim * parameters$ydim,
    occupied_nodes = length(unique(result$som_labels)),
    nodes_with_two_batches = sum(result$node_correction_modes$present_batches == 2L),
    nodes_with_three_batches = sum(result$node_correction_modes$present_batches == 3L),
    correction_mode_counts = mode_counts
  ),
  runtime_diagnostics = list(
    message_count = length(fit_messages),
    warning_count = length(fit_warnings),
    warnings = as.list(unique(fit_warnings)),
    full_log = "cycombine_messages.log"
  ),
  interpretation_limits = list(
    correction_som_is_pre_correction_rank_som = TRUE,
    post_correction_evaluation_som_not_computed = TRUE,
    batch_is_dataset_platform = TRUE,
    covariate_is_null = TRUE,
    technical_and_cohort_effects_may_be_confounded = TRUE
  ),
  local_source_sha256 = list(
    runner = hash_file(script_path),
    wrapper = hash_file(file.path(algorithm_dir, "cycombine_repro.R")),
    upstream_lock = hash_file(file.path(algorithm_dir, "upstream_lock.json"))
  ),
  outputs = output_hashes
)
manifest_path <- file.path(staging_path, "manifest.json")
jsonlite::write_json(
  manifest,
  manifest_path,
  pretty = TRUE,
  auto_unbox = TRUE,
  null = "null",
  digits = NA
)
complete_path <- file.path(staging_path, "COMPLETE.json")
jsonlite::write_json(
  list(
    status = "complete",
    manifest = "manifest.json",
    manifest_sha256 = hash_file(manifest_path)
  ),
  complete_path,
  pretty = TRUE,
  auto_unbox = TRUE
)

if (!file.rename(staging_path, output_path)) {
  stop(sprintf("Could not atomically move completed output into %s", output_path), call. = FALSE)
}
staging_path <- NULL

cat(sprintf(
  "cyCombine completed: %d cells, %d shared markers, %d batches; output: %s\n",
  nrow(input), length(markers), length(config$batch_levels), output_path
))
