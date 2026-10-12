#!/usr/bin/env Rscript

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
test_path <- normalizePath(sub("^--file=", "", script_arg), mustWork = TRUE)
algorithm_dir <- dirname(dirname(test_path))
Sys.setenv(CYTOFRUV_ALGORITHM_DIR = algorithm_dir)
source(file.path(algorithm_dir, "cytofruv_repro.R"), local = globalenv())
source(file.path(algorithm_dir, "tests", "fixture_helpers.R"), local = globalenv())
activate_cytofruv_runtime()

fixture_root <- tempfile("cytofruv-guard-")
dir.create(fixture_root, recursive = TRUE)
on.exit(unlink(fixture_root, recursive = TRUE, force = TRUE), add = TRUE)

expect_error <- function(expression, pattern) {
  message <- tryCatch({ force(expression); NA_character_ }, error = function(e) conditionMessage(e))
  if (is.na(message) || !grepl(pattern, message, fixed = TRUE)) {
    stop("Expected error containing: ", pattern, "; observed: ", message)
  }
}

markers <- c("M1", "M2", "M3")
input <- data.frame(
  cell_id = paste0("c", seq_len(12)),
  sample_id = rep(c("donor_A", "donor_B"), each = 6),
  batch = rep(c("platform_A", "platform_B"), each = 6),
  cluster = rep(c("1", "2"), 6),
  M1 = seq_len(12),
  M2 = seq_len(12)^2,
  M3 = log1p(seq_len(12)),
  check.names = FALSE,
  stringsAsFactors = FALSE
)

# The evidence map truthfully records two unrelated donors. The deliberately
# fabricated config below tries to relabel them as one reference and must fail.
replicate_map <- data.frame(
  reference_id = c("donor_A_reference", "donor_B_reference"),
  sample_id = c("donor_A", "donor_B"),
  batch = c("platform_A", "platform_B"),
  stringsAsFactors = FALSE
)
provenance <- write_cytofruv_fixture_provenance(
  fixture_root,
  input,
  markers,
  replicate_map,
  clustering_seed = 1L,
  number_of_clusters = 2L
)

base_config <- list(
  schema_version = "final_xu.cytofruv.config.v1",
  method_mode = "official_replicated_reference",
  seed = 1L,
  k = 1L,
  marker_columns = as.list(markers),
  input_scale = list(
    name = "arcsinh_intensity_over_cofactor", cofactor = 5,
    already_transformed = TRUE,
    source_document = provenance$scale_path,
    source_document_sha256 = provenance$scale_sha256
  ),
  clustering = list(
    method = "CytofRUV::cluster_data", type_markers = as.list(markers),
    number_of_clusters = 2L, seed = 1L,
    provenance_document = provenance$clustering_path,
    provenance_document_sha256 = provenance$clustering_sha256
  ),
  replicate_design = list(
    attestation = "same_biological_reference_sample_remeasured_across_all_target_batches",
    target_batches = list("platform_A", "platform_B"),
    source_document = provenance$design_path,
    source_document_sha256 = provenance$design_sha256,
    replicate_map = provenance$map_path,
    replicate_map_sha256 = provenance$map_sha256
  ),
  replicate_sets = list(list(
    reference_id = "fabricated_reference",
    sample_ids = list("donor_A", "donor_B"),
    clusters = list("1", "2")
  )),
  minimum_cells_per_reference_sample_cluster = 1L
)

bad_mode <- base_config
bad_mode$method_mode <- "adtnorm_all_samples_adaptation"
expect_error(validate_cytofruv_config(bad_mode), "Only official_replicated_reference mode")

no_references <- base_config
no_references$replicate_sets <- list()
expect_error(validate_cytofruv_config(no_references), "At least one real replicate set is required")

one_batch <- base_config
one_batch$replicate_sets[[1]]$sample_ids <- list("donor_A")
expect_error(validate_cytofruv_config(one_batch), "at least 2 non-empty character string")

missing_batch <- base_config
missing_batch$replicate_sets[[1]]$sample_ids <- list("donor_A", "donor_A")
expect_error(validate_cytofruv_config(missing_batch), "at least two unique acquisition-level sample_ids")

bad_attestation <- base_config
bad_attestation$replicate_design$attestation <- "similar_donors_across_platforms"
expect_error(validate_cytofruv_config(bad_attestation), "requires a real biological reference")

expect_error(
  validate_cytofruv_design(input, base_config),
  "disagree with the hash-bound replicate map"
)

numeric_markers <- base_config
numeric_markers$marker_columns <- list(1L, 2L)
expect_error(validate_cytofruv_config(numeric_markers), "must contain character strings only")

null_marker <- base_config
null_marker$marker_columns <- list("M1", NULL, "M2")
expect_error(validate_cytofruv_config(null_marker), "must contain character strings only")

reserved_marker <- base_config
reserved_marker$marker_columns <- list("cluster", "M1")
expect_error(validate_cytofruv_config(reserved_marker), "cannot reuse reserved metadata names")

logical_seed <- base_config
logical_seed$seed <- TRUE
expect_error(validate_cytofruv_config(logical_seed), "must be one explicit integer")

duplicate_key <- c(base_config, list(k = 2L))
expect_error(validate_cytofruv_config(duplicate_key), "duplicate keys")

duplicate_input <- input
names(duplicate_input)[ncol(duplicate_input)] <- "M2"
expect_error(validate_cytofruv_design(duplicate_input, base_config), "globally unique")

fake_cluster_input <- input
fake_cluster_input$cluster <- rev(fake_cluster_input$cluster)
expect_error(
  validate_cytofruv_design(fake_cluster_input, base_config),
  "hash-bound official clustering provenance"
)

cat("PASS: CytofRUV applicability guard rejects unsupported cross-platform adaptations\n")
