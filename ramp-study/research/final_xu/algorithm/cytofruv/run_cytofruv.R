#!/usr/bin/env Rscript

parse_cli <- function(args) {
  expected <- c("--input", "--config", "--output")
  if (length(args) != 6L || !all(args[c(TRUE, FALSE)] %in% expected)) {
    stop("Usage: Rscript --vanilla run_cytofruv.R --input INPUT.csv[.gz] --config CONFIG.json --output OUTPUT_DIR")
  }
  values <- setNames(args[c(FALSE, TRUE)], sub("^--", "", args[c(TRUE, FALSE)]))
  expected_names <- sub("^--", "", expected)
  if (anyDuplicated(names(values)) || !setequal(names(values), expected_names)) {
    stop("CLI flags must be unique and complete")
  }
  values
}

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1L) stop("Unable to resolve run_cytofruv.R path")
runner_arg <- sub("^--file=", "", script_arg)
# R's nested system() launcher can encode spaces in the --file token as ~+~.
runner_arg <- gsub("~+~", " ", runner_arg, fixed = TRUE)
runner_path <- normalizePath(runner_arg, mustWork = TRUE)
algorithm_dir <- dirname(runner_path)
Sys.setenv(CYTOFRUV_ALGORITHM_DIR = algorithm_dir)
source(file.path(algorithm_dir, "cytofruv_repro.R"), local = globalenv())
activate_cytofruv_runtime()

if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite is required")

args <- parse_cli(commandArgs(trailingOnly = TRUE))
input_path <- normalizePath(args[["input"]], mustWork = TRUE)
config_path <- normalizePath(args[["config"]], mustWork = TRUE)
for (path in c(input_path, config_path)) {
  if (isTRUE(file.info(path)$isdir)) stop("Input and config paths must be regular files")
}
output_arg <- path.expand(args[["output"]])
output_parent <- normalizePath(dirname(output_arg), mustWork = TRUE)
output_path <- file.path(output_parent, basename(output_arg))
if (file.exists(output_path)) stop("Output path already exists; refusing to overwrite: ", output_path)

# Cooperative lock closes the normal concurrent-run race. The final existence
# check remains in place immediately before the atomic same-filesystem rename.
lock_path <- paste0(output_path, ".cytofruv.lock")
if (!dir.create(lock_path, recursive = FALSE, showWarnings = FALSE)) {
  stop("Output is locked by another CytofRUV run: ", lock_path)
}
lock_held <- TRUE
on.exit({
  if (lock_held && dir.exists(lock_path)) unlink(lock_path, recursive = TRUE, force = TRUE)
}, add = TRUE)

anchor_paths <- c(
  input = input_path,
  config = config_path,
  wrapper = file.path(algorithm_dir, "cytofruv_repro.R"),
  runner = runner_path,
  upstream_lock = file.path(algorithm_dir, "upstream_lock.json"),
  runtime_lock = file.path(algorithm_dir, "runtime_lock.csv"),
  runtime_function_lock = file.path(algorithm_dir, "runtime_function_lock.csv")
)
anchor_paths <- vapply(anchor_paths, normalizePath, character(1), mustWork = TRUE)
anchor_hashes <- vapply(anchor_paths, cytofruv_sha256, character(1))

config <- jsonlite::fromJSON(config_path, simplifyVector = FALSE)
validate_cytofruv_config(config)

resolve_document <- function(path) {
  path <- cytofruv_scalar_character(path, "provenance path")
  if (!startsWith(path, .Platform$file.sep)) path <- file.path(dirname(config_path), path)
  resolved <- normalizePath(path, mustWork = TRUE)
  if (isTRUE(file.info(resolved)$isdir)) stop("Provenance path must be a regular file: ", resolved)
  resolved
}
config$input_scale$source_document <- resolve_document(config$input_scale$source_document)
config$clustering$provenance_document <- resolve_document(config$clustering$provenance_document)
config$replicate_design$source_document <- resolve_document(config$replicate_design$source_document)
config$replicate_design$replicate_map <- resolve_document(config$replicate_design$replicate_map)

provenance_paths <- c(
  input_scale_source = config$input_scale$source_document,
  clustering_provenance = config$clustering$provenance_document,
  replicate_design_source = config$replicate_design$source_document,
  replicate_map = config$replicate_design$replicate_map
)
provenance_hashes <- vapply(provenance_paths, cytofruv_sha256, character(1))

connection <- if (grepl("[.]gz$", input_path, ignore.case = TRUE)) gzfile(input_path, "rt") else input_path
input <- read.csv(
  connection,
  stringsAsFactors = FALSE,
  colClasses = "character",
  check.names = FALSE
)
if (inherits(connection, "connection")) close(connection)
if (anyDuplicated(names(input))) stop("Input column names must be globally unique")
markers <- cytofruv_json_string_array(config$marker_columns, "marker_columns", minimum = 2L)
if (!all(markers %in% names(input))) stop("One or more configured markers are absent from input")
for (marker in markers) {
  converted <- suppressWarnings(as.numeric(input[[marker]]))
  if (anyNA(converted) || any(!is.finite(converted))) {
    stop("Marker column is not strictly finite numeric: ", marker)
  }
  input[[marker]] <- converted
}

result <- run_cytofruv_official_core(input, config)

# Verify that every byte used as an execution/provenance anchor stayed fixed
# throughout the numerical run.
late_anchor_hashes <- vapply(anchor_paths, cytofruv_sha256, character(1))
if (!identical(anchor_hashes, late_anchor_hashes)) stop("An execution anchor changed during the run")
late_provenance_hashes <- vapply(provenance_paths, cytofruv_sha256, character(1))
if (!identical(provenance_hashes, late_provenance_hashes)) stop("A provenance document changed during the run")

staging <- tempfile(paste0(basename(output_path), ".staging-"), tmpdir = output_parent)
if (!dir.create(staging, recursive = FALSE)) stop("Unable to create output staging directory")
completed <- FALSE
on.exit({
  if (!completed && dir.exists(staging)) unlink(staging, recursive = TRUE, force = TRUE)
}, add = TRUE)

write_csv_gz <- function(data, path) {
  con <- gzfile(path, "wt", compression = 6)
  on.exit(close(con), add = TRUE)
  write.csv(data, con, row.names = FALSE, quote = TRUE, na = "")
}

corrected_path <- file.path(staging, "cytofruv_corrected_values.csv.gz")
metadata_path <- file.path(staging, "cell_metadata_label_free.csv.gz")
audit_path <- file.path(staging, "reference_design_audit.csv")
manifest_path <- file.path(staging, "manifest.json")
provenance_dir <- file.path(staging, "provenance")
if (!dir.create(provenance_dir, recursive = FALSE)) stop("Unable to create provenance directory")

corrected_table <- data.frame(
  cell_id = as.character(input$cell_id),
  result$corrected,
  check.names = FALSE,
  stringsAsFactors = FALSE
)
write_csv_gz(corrected_table, corrected_path)
write_csv_gz(input[c("cell_id", "sample_id", "batch", "cluster")], metadata_path)
write.csv(result$design_audit, audit_path, row.names = FALSE, quote = TRUE)

bound_sources <- c(config = config_path, provenance_paths)
bound_names <- c(
  config = "config.json",
  input_scale_source = "input_scale_source_document",
  clustering_provenance = "clustering_provenance.json",
  replicate_design_source = "replicate_design_source_document",
  replicate_map = "replicate_map.csv"
)
bound_records <- list()
for (name in names(bound_sources)) {
  destination <- file.path(provenance_dir, bound_names[[name]])
  if (!file.copy(bound_sources[[name]], destination, overwrite = FALSE, copy.mode = TRUE)) {
    stop("Unable to copy bound provenance file: ", name)
  }
  expected <- if (identical(name, "config")) anchor_hashes[["config"]] else provenance_hashes[[name]]
  observed <- cytofruv_sha256(destination)
  if (!identical(observed, expected)) stop("Copied provenance hash mismatch: ", name)
  bound_records[[name]] <- list(
    file = file.path("provenance", bound_names[[name]]),
    sha256 = observed
  )
}

runtime <- cytofruv_runtime_lock()
runtime$observed_path <- vapply(runtime$package, function(package) {
  if (identical(package, "R")) return(R.home())
  normalizePath(getNamespaceInfo(asNamespace(package), "path"), mustWork = TRUE)
}, character(1))

platform <- list(
  R_version = paste(R.version$major, R.version$minor, sep = "."),
  R_platform = R.version$platform,
  os = as.list(Sys.info()),
  BLAS = unname(extSoftVersion()[["BLAS"]]),
  LAPACK = tryCatch(La_library(), error = function(e) NA_character_),
  session_info = capture.output(sessionInfo())
)

manifest <- list(
  schema_version = "final_xu.cytofruv.output.v1",
  status = "complete",
  method = "CytofRUV",
  method_mode = config$method_mode,
  method_scope = "official replicated-reference design only",
  upstream = list(
    repository = "https://github.com/mtrussart/CytofRUV",
    package_version = "0.1",
    commit = "7088af8794ede0fc7e3c6ef545638e279e8101eb",
    source_archive_sha256 = "a614b250a7f14488679a8822e1b1b6bb72f5fe292a74289b4f63b3f8b4d9f6b6",
    core_source_sha256 = "cb0c07105985d38b86c198816172ea3144b749bf2cbe50e9681ade4e8ae692ab"
  ),
  implementation = list(
    official_run_RUVIII_called_directly = TRUE,
    upstream_core_modified = FALSE,
    upstream_double_rsvd_behavior_preserved = TRUE,
    hidden_preprocessing = FALSE,
    fit_schema_rejects_reserved_label_columns = TRUE,
    biological_reference_identity = "externally_attested_and_hash_bound",
    wrapper_sha256 = anchor_hashes[["wrapper"]],
    runner_sha256 = anchor_hashes[["runner"]],
    upstream_lock_sha256 = anchor_hashes[["upstream_lock"]],
    runtime_lock_sha256 = anchor_hashes[["runtime_lock"]],
    runtime_function_lock_sha256 = anchor_hashes[["runtime_function_lock"]]
  ),
  input = list(
    path = input_path,
    sha256 = anchor_hashes[["input"]],
    config_sha256 = anchor_hashes[["config"]],
    rows = nrow(input),
    markers = markers,
    marker_count = length(markers),
    batches = unique(as.character(input$batch)),
    samples = length(unique(as.character(input$sample_id))),
    scale = config$input_scale,
    clustering = config$clustering
  ),
  parameters = list(
    seed = as.integer(config$seed),
    k = as.integer(config$k),
    replicate_design = config$replicate_design,
    replicate_sets = config$replicate_sets,
    minimum_cells_per_reference_sample_cluster = as.integer(config$minimum_cells_per_reference_sample_cluster)
  ),
  provenance = bound_records,
  runtime = runtime,
  platform = platform,
  outputs = list(
    corrected_values = list(file = basename(corrected_path), sha256 = cytofruv_sha256(corrected_path)),
    metadata = list(file = basename(metadata_path), sha256 = cytofruv_sha256(metadata_path)),
    reference_design_audit = list(file = basename(audit_path), sha256 = cytofruv_sha256(audit_path))
  )
)
jsonlite::write_json(manifest, manifest_path, auto_unbox = TRUE, pretty = TRUE, null = "null", digits = NA)

complete <- list(
  schema_version = "final_xu.cytofruv.complete.v1",
  status = "complete",
  manifest_sha256 = cytofruv_sha256(manifest_path)
)
jsonlite::write_json(
  complete,
  file.path(staging, "COMPLETE.json"),
  auto_unbox = TRUE,
  pretty = TRUE,
  null = "null"
)

if (file.exists(output_path)) stop("Output path appeared during the run; refusing to replace it")
if (!file.rename(staging, output_path)) stop("Atomic output directory rename failed")
completed <- TRUE
unlink(lock_path, recursive = TRUE, force = TRUE)
lock_held <- FALSE
cat("PASS: CytofRUV official replicated-reference run completed at ", output_path, "\n", sep = "")
