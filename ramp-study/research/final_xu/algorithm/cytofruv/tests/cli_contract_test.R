#!/usr/bin/env Rscript

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
test_path <- normalizePath(sub("^--file=", "", script_arg), mustWork = TRUE)
algorithm_dir <- dirname(dirname(test_path))
Sys.setenv(CYTOFRUV_ALGORITHM_DIR = algorithm_dir)
source(file.path(algorithm_dir, "cytofruv_repro.R"), local = globalenv())
source(file.path(algorithm_dir, "tests", "fixture_helpers.R"), local = globalenv())
activate_cytofruv_runtime()
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite is required")

fixture_root <- tempfile("cytofruv-cli-")
dir.create(fixture_root, recursive = TRUE)
on.exit(unlink(fixture_root, recursive = TRUE, force = TRUE), add = TRUE)

batches <- c("b1", "b2", "b3")
references <- paste0("ref_", batches)
markers <- paste0("M", seq_len(6))
set.seed(43)
rows <- list()
index <- 1L
for (i in seq_along(batches)) {
  for (sample in c(references[[i]], paste0("patient_", batches[[i]]))) {
    for (cluster in c("1", "2")) {
      values <- matrix(rnorm(5L * length(markers), mean = i * 0.15 + as.integer(cluster)), ncol = length(markers))
      block <- data.frame(
        cell_id = paste(sample, cluster, seq_len(5), sep = "__"),
        sample_id = sample,
        batch = batches[[i]],
        cluster = cluster,
        values,
        check.names = FALSE,
        stringsAsFactors = FALSE
      )
      names(block)[5:ncol(block)] <- markers
      rows[[index]] <- block
      index <- index + 1L
    }
  }
}
input <- do.call(rbind, rows)
input_path <- file.path(fixture_root, "input.csv.gz")
con <- gzfile(input_path, "wt")
write.csv(input, con, row.names = FALSE, quote = TRUE)
close(con)

replicate_map <- data.frame(
  reference_id = rep("reference_A", length(references)),
  sample_id = references,
  batch = batches,
  stringsAsFactors = FALSE
)
provenance <- write_cytofruv_fixture_provenance(
  fixture_root,
  input,
  markers,
  replicate_map,
  clustering_seed = 1234L,
  number_of_clusters = 2L
)
config <- list(
  schema_version = "final_xu.cytofruv.config.v1",
  method_mode = "official_replicated_reference",
  seed = 1234L,
  k = 2L,
  marker_columns = markers,
  input_scale = list(
    name = "arcsinh_intensity_over_cofactor", cofactor = 5,
    already_transformed = TRUE,
    source_document = basename(provenance$scale_path),
    source_document_sha256 = provenance$scale_sha256
  ),
  clustering = list(
    method = "CytofRUV::cluster_data", type_markers = markers,
    number_of_clusters = 2L, seed = 1234L,
    provenance_document = basename(provenance$clustering_path),
    provenance_document_sha256 = provenance$clustering_sha256
  ),
  replicate_design = list(
    attestation = "same_biological_reference_sample_remeasured_across_all_target_batches",
    target_batches = batches,
    source_document = basename(provenance$design_path),
    source_document_sha256 = provenance$design_sha256,
    replicate_map = basename(provenance$map_path),
    replicate_map_sha256 = provenance$map_sha256
  ),
  replicate_sets = list(list(
    reference_id = "reference_A", sample_ids = references, clusters = c("1", "2")
  )),
  minimum_cells_per_reference_sample_cluster = 2L
)
config_path <- file.path(fixture_root, "config.json")
jsonlite::write_json(config, config_path, auto_unbox = TRUE, pretty = TRUE)

output_path <- file.path(fixture_root, "output")
command <- c(
  file.path(algorithm_dir, "run_cytofruv.R"),
  "--input", input_path,
  "--config", config_path,
  "--output", output_path
)
command_line <- paste(
  shQuote(file.path(R.home("bin"), "Rscript")),
  "--vanilla",
  paste(shQuote(command), collapse = " "),
  "2>&1"
)
run_output <- system(command_line, intern = TRUE)
status <- attr(run_output, "status")
if (!is.null(status) && status != 0L) stop(paste(run_output, collapse = "\n"))

required <- c(
  "cytofruv_corrected_values.csv.gz", "cell_metadata_label_free.csv.gz",
  "reference_design_audit.csv", "manifest.json", "COMPLETE.json"
)
stopifnot(all(file.exists(file.path(output_path, required))))
corrected <- read.csv(gzfile(file.path(output_path, required[[1]])), check.names = FALSE)
stopifnot(identical(as.character(corrected$cell_id), as.character(input$cell_id)))
stopifnot(identical(names(corrected), c("cell_id", markers)))

manifest_path <- file.path(output_path, "manifest.json")
manifest <- jsonlite::fromJSON(manifest_path, simplifyVector = FALSE)
complete <- jsonlite::fromJSON(file.path(output_path, "COMPLETE.json"), simplifyVector = FALSE)
stopifnot(identical(manifest$status, "complete"))
stopifnot(isTRUE(manifest$implementation$official_run_RUVIII_called_directly))
stopifnot(isTRUE(manifest$implementation$fit_schema_rejects_reserved_label_columns))
stopifnot(identical(manifest$input$config_sha256, cytofruv_sha256(config_path)))
stopifnot(length(manifest$provenance) == 5L)
stopifnot(all(file.exists(file.path(output_path, vapply(manifest$provenance, `[[`, character(1), "file")))))
stopifnot(identical(complete$manifest_sha256, cytofruv_sha256(manifest_path)))

second <- suppressWarnings(system(command_line, intern = TRUE))
second_status <- attr(second, "status")
stopifnot(!is.null(second_status), second_status != 0L)

cat("PASS: CytofRUV CLI, outputs, row lock, manifest, and no-overwrite contract\n")
