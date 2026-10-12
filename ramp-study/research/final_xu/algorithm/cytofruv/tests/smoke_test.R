#!/usr/bin/env Rscript

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
test_path <- normalizePath(sub("^--file=", "", script_arg), mustWork = TRUE)
algorithm_dir <- dirname(dirname(test_path))
Sys.setenv(CYTOFRUV_ALGORITHM_DIR = algorithm_dir)
source(file.path(algorithm_dir, "cytofruv_repro.R"), local = globalenv())
source(file.path(algorithm_dir, "tests", "fixture_helpers.R"), local = globalenv())
activate_cytofruv_runtime()

fixture_root <- tempfile("cytofruv-smoke-")
dir.create(fixture_root, recursive = TRUE)
on.exit(unlink(fixture_root, recursive = TRUE, force = TRUE), add = TRUE)

set.seed(9182)
markers <- paste0("M", seq_len(6))
batches <- paste0("batch_", seq_len(3))
reference_samples <- paste0("reference_A_", batches)
other_samples <- paste0("patient_", seq_len(3))

make_sample <- function(sample_id, batch, reference, offset) {
  clusters <- rep(c("1", "2"), each = 12L)
  base <- matrix(rnorm(length(clusters) * length(markers), sd = 0.25), ncol = length(markers))
  biological <- ifelse(clusters == "1", 0.8, -0.4)
  values <- sweep(base, 1L, biological, "+")
  values <- sweep(values, 2L, seq(-0.2, 0.3, length.out = length(markers)), "+")
  values <- values + offset
  data.frame(
    cell_id = paste(sample_id, seq_along(clusters), sep = "__"),
    sample_id = sample_id,
    batch = batch,
    cluster = clusters,
    values,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}

input <- do.call(rbind, c(
  lapply(seq_along(batches), function(i) make_sample(reference_samples[[i]], batches[[i]], TRUE, (i - 2) * 0.35)),
  lapply(seq_along(batches), function(i) make_sample(other_samples[[i]], batches[[i]], FALSE, (i - 2) * 0.35 + 0.1))
))
names(input)[5:ncol(input)] <- markers

replicate_map <- data.frame(
  reference_id = rep("reference_A", length(reference_samples)),
  sample_id = reference_samples,
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
  marker_columns = as.list(markers),
  input_scale = list(
    name = "arcsinh_intensity_over_cofactor",
    cofactor = 5,
    already_transformed = TRUE,
    source_document = provenance$scale_path,
    source_document_sha256 = provenance$scale_sha256
  ),
  clustering = list(
    method = "CytofRUV::cluster_data",
    type_markers = as.list(markers),
    number_of_clusters = 2L,
    seed = 1234L,
    provenance_document = provenance$clustering_path,
    provenance_document_sha256 = provenance$clustering_sha256
  ),
  replicate_design = list(
    attestation = "same_biological_reference_sample_remeasured_across_all_target_batches",
    target_batches = as.list(batches),
    source_document = provenance$design_path,
    source_document_sha256 = provenance$design_sha256,
    replicate_map = provenance$map_path,
    replicate_map_sha256 = provenance$map_sha256
  ),
  replicate_sets = list(list(
    reference_id = "reference_A",
    sample_ids = as.list(reference_samples),
    clusters = list("1", "2")
  )),
  minimum_cells_per_reference_sample_cluster = 2L
)

had_global_sd <- exists("sd", envir = .GlobalEnv, inherits = FALSE)
if (had_global_sd) previous_global_sd <- get("sd", envir = .GlobalEnv, inherits = FALSE)
assign("sd", function(...) stop("GLOBAL_SD_POISON"), envir = .GlobalEnv)
on.exit({
  if (had_global_sd) {
    assign("sd", previous_global_sd, envir = .GlobalEnv)
  } else if (exists("sd", envir = .GlobalEnv, inherits = FALSE)) {
    rm("sd", envir = .GlobalEnv)
  }
}, add = TRUE)

first <- run_cytofruv_official_core(input, config)
second <- run_cytofruv_official_core(input, config)
stopifnot(identical(first$corrected, second$corrected))
stopifnot(identical(dim(first$corrected), c(nrow(input), length(markers))))
stopifnot(all(is.finite(first$corrected)))
stopifnot(max(abs(first$corrected - as.matrix(input[markers]))) > 1e-6)
stopifnot(nrow(first$design_audit) == length(reference_samples) * 2L)

upstream <- load_official_cytofruv_core()
raw_data <- data.frame(
  sample = input$sample_id,
  cluster = input$cluster,
  input[markers],
  check.names = FALSE,
  stringsAsFactors = FALSE
)
set.seed(1234L, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
direct <- upstream$run_RUVIII(
  raw_data,
  norm_clusters = list(c("1", "2")),
  k = 2L,
  rep_samples = list(reference_samples)
)
colnames(direct) <- markers
stopifnot(identical(first$corrected, as.matrix(direct)))

cat("PASS: CytofRUV smoke, determinism, and exact official-core parity\n")
