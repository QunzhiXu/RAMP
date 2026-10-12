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

validate_harmony_install(local_library)
parameters <- canonical_harmony_direct_parameters()
batch_levels <- c("platform_Z", "platform_A", "platform_M")
markers <- c("CD3", "CD4", "CD8", "CD19", "CD45", "HLA_DR")

policy_inputs <- c(16L, 45L, 60L, 3000L, 9000L)
policy_expected <- c(1L, 2L, 2L, 100L, 100L)
policy_actual <- vapply(policy_inputs, resolved_nclust, integer(1))
if (!identical(unname(policy_actual), policy_expected)) {
  stop("Resolved nclust policy, including the 100-cluster cap, differs from Harmony 1.2.3.", call. = FALSE)
}

set.seed(20250830)
n_per_batch <- 360L
parts <- lapply(seq_along(batch_levels), function(batch_index) {
  state <- sample.int(4L, n_per_batch, replace = TRUE)
  marker_matrix <- matrix(stats::rexp(n_per_batch * length(markers), rate = 1), nrow = n_per_batch)
  colnames(marker_matrix) <- markers
  if (batch_index == 2L) {
    marker_matrix[, "CD3"] <- marker_matrix[, "CD3"] + 1.5
    marker_matrix[, "CD8"] <- marker_matrix[, "CD8"] + 0.5
  }
  if (batch_index == 3L) {
    marker_matrix[, "CD4"] <- marker_matrix[, "CD4"] + 1.2
    marker_matrix[, "CD19"] <- marker_matrix[, "CD19"] + 0.7
  }
  data.frame(
    cell_id = sprintf("%s_cell_%04d", batch_levels[batch_index], seq_len(n_per_batch)),
    dataset = batch_levels[batch_index],
    unit_key = sprintf("%s_donor_%02d", batch_levels[batch_index], 1L + (seq_len(n_per_batch) - 1L) %% 8L),
    marker_matrix,
    provider_celltype = paste0("state_", state),
    survival_outcome = stats::rbinom(n_per_batch, 1, 0.35),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
})
fixture <- do.call(rbind, parts)
rownames(fixture) <- NULL

fit_once <- function(data = fixture) {
  fit_harmony_direct(
    input = data,
    markers = markers,
    cell_id_column = "cell_id",
    batch_column = "dataset",
    sample_column = "unit_key",
    batch_levels = batch_levels,
    parameters = parameters,
    required_library = local_library
  )
}

first <- fit_once()
second <- fit_once()
if (!identical(first$adjusted_marker_coordinates, second$adjusted_marker_coordinates)) {
  stop("Same-seed deterministic rerun failed.", call. = FALSE)
}
if (!identical(first$resolved_parameters$resolved_nclust, 36L)) {
  stop("Resolved nclust does not follow min(round(N/30),100).", call. = FALSE)
}
if (!identical(first$input_marker_coordinates, as.matrix(fixture[, markers, drop = FALSE]) |> `rownames<-`(fixture$cell_id))) {
  stop("The algorithm layer changed the analysis-ready input before Harmony.", call. = FALSE)
}

prepared <- build_harmony_direct_input(
  fixture, markers, "cell_id", "dataset", "unit_key", batch_levels
)
official_legacy <- with_locked_rng(parameters, suppressWarnings(suppressMessages(harmony::HarmonyMatrix(
  data_mat = prepared$marker_matrix,
  meta_data = as.character(prepared$batches),
  do_pca = FALSE
))))
official_modern <- with_locked_rng(parameters, suppressMessages(harmony::RunHarmony(
  data_mat = prepared$marker_matrix,
  meta_data = as.character(prepared$batches),
  verbose = FALSE
)))
legacy_error <- max(abs(first$adjusted_marker_coordinates - official_legacy))
modern_error <- max(abs(first$adjusted_marker_coordinates - official_modern))
if (!identical(first$adjusted_marker_coordinates, official_legacy) || legacy_error != 0) {
  stop(sprintf("ADTnorm-style HarmonyMatrix exact parity failed; max abs error %.17g.", legacy_error), call. = FALSE)
}
if (!identical(first$adjusted_marker_coordinates, official_modern) || modern_error != 0) {
  stop(sprintf("Modern RunHarmony default exact parity failed; max abs error %.17g.", modern_error), call. = FALSE)
}
if (!identical(dim(first$adjusted_marker_coordinates), c(nrow(fixture), length(markers))) ||
    !identical(rownames(first$adjusted_marker_coordinates), fixture$cell_id) ||
    !identical(colnames(first$adjusted_marker_coordinates), markers)) {
  stop("Cell/marker dimensions, names, or order were not preserved.", call. = FALSE)
}
if (any(c("provider_celltype", "survival_outcome") %in% first$fitting_input_columns) ||
    !all(c("provider_celltype", "survival_outcome") %in% first$ignored_input_columns)) {
  stop("A biological label leaked into fitting or the ignored-column audit is incomplete.", call. = FALSE)
}
mutated_provenance <- fixture
mutated_provenance$provider_celltype <- paste0("arbitrary_", rev(seq_len(nrow(mutated_provenance))))
mutated_provenance$survival_outcome <- 1L - mutated_provenance$survival_outcome
mutated_provenance$unit_key <- paste0("rewritten_", mutated_provenance$unit_key)
mutated_fit <- fit_once(mutated_provenance)
if (!identical(first$adjusted_marker_coordinates, mutated_fit$adjusted_marker_coordinates)) {
  stop("Changing ignored labels or sample provenance changed the Harmony fit.", call. = FALSE)
}
if (stats::median(first$marker_rank_audit$rank_changed_fraction) <= 0.5 ||
    max(first$marker_rank_audit$max_abs_midrank_displacement) <= 0.10 ||
    min(first$marker_rank_audit$spearman, na.rm = TRUE) >= 0.995) {
  stop("Synthetic regression fixture did not expose the expected non-rank-preserving direct-marker behavior.", call. = FALSE)
}
if (!nrow(first$marker_rank_audit_by_sample) ||
    !all(c("batch", "sample", "marker", "rank_changed_fraction") %in% colnames(first$marker_rank_audit_by_sample))) {
  stop("Within-sample rank audit is missing or malformed.", call. = FALSE)
}

small <- fixture[c(seq_len(30L), n_per_batch + seq_len(30L), 2L * n_per_batch + seq_len(30L)), ]
small[1L, markers] <- 0
small_fit <- fit_once(small)
if (!identical(small_fit$resolved_parameters$resolved_nclust, 3L) ||
    any(!is.finite(small_fit$adjusted_marker_coordinates))) {
  stop("Small-n nclust branch or all-zero-cell official-domain regression failed.", call. = FALSE)
}

cat(sprintf(
  paste0(
    "PASS: Harmony %s direct-marker; %d cells x %d markers; deterministic; ",
    "HarmonyMatrix max abs %.1f; RunHarmony max abs %.1f; nclust branches 36/3 and cap=100 verified; batch/sample rank audits active\n"
  ),
  as.character(utils::packageVersion("harmony")),
  nrow(fixture),
  length(markers),
  legacy_error,
  modern_error
))
