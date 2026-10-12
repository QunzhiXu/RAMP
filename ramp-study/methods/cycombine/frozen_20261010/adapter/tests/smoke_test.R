#!/usr/bin/env Rscript

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_argument)) stop("Cannot determine script location.", call. = FALSE)
test_path <- sub("^--file=", "", script_argument[[1L]])
test_path <- gsub("~+~", " ", test_path, fixed = TRUE)
test_path <- normalizePath(test_path, mustWork = TRUE)
algorithm_dir <- dirname(dirname(test_path))
local_library <- file.path(algorithm_dir, ".Rlib")
.libPaths(c(local_library, .libPaths()))
source(file.path(algorithm_dir, "cycombine_repro.R"), local = FALSE)

validate_cycombine_install(local_library)

set.seed(20260830)
batch_levels <- c("platform_A", "platform_B", "platform_C")
n_per_batch <- 900L
parts <- vector("list", length(batch_levels))

for (batch_index in seq_along(batch_levels)) {
  latent_class <- sample(1:4, n_per_batch, replace = TRUE, prob = c(0.3, 0.25, 0.25, 0.2))
  z1 <- stats::rnorm(n_per_batch, mean = latent_class * 0.65, sd = 0.45)
  z2 <- stats::rnorm(n_per_batch, mean = (5 - latent_class) * 0.45, sd = 0.5)
  batch_shift <- c(-0.30, 0.15, 0.45)[batch_index]
  parts[[batch_index]] <- data.frame(
    cell_id = sprintf("%s_cell_%04d", batch_levels[batch_index], seq_len(n_per_batch)),
    dataset = batch_levels[batch_index],
    unit_key = sprintf("%s_donor_%02d", batch_levels[batch_index], 1L + (seq_len(n_per_batch) - 1L) %% 9L),
    CD3 = 4.0 + z1 + batch_shift + stats::rnorm(n_per_batch, 0, 0.12),
    CD4 = 3.5 + z2 - 0.5 * batch_shift + stats::rnorm(n_per_batch, 0, 0.13),
    CD8 = 3.0 + 0.7 * z1 - 0.4 * z2 + 0.4 * batch_shift + stats::rnorm(n_per_batch, 0, 0.14),
    CD45 = 5.0 + 0.25 * z1 + 0.25 * z2 + 0.2 * batch_shift + stats::rnorm(n_per_batch, 0, 0.10),
    HLA_DR = 2.5 - 0.35 * z1 + 0.6 * z2 - 0.3 * batch_shift + stats::rnorm(n_per_batch, 0, 0.15),
    provider_celltype = paste0("class_", latent_class),
    survival_outcome = stats::rbinom(n_per_batch, 1, 0.4),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

fixture <- do.call(rbind, parts)
markers <- c("CD3", "CD4", "CD8", "CD45", "HLA_DR")
parameters <- canonical_cycombine_parameters()

fit_once <- function() {
  fit_cycombine_publication(
    input = fixture,
    markers = markers,
    cell_id_column = "cell_id",
    batch_column = "dataset",
    sample_column = "unit_key",
    batch_levels = batch_levels,
    parameters = parameters,
    required_library = local_library
  )
}

first <- suppressMessages(fit_once())
second <- suppressMessages(fit_once())
if (!isTRUE(all.equal(first$corrected_expression, second$corrected_expression, tolerance = 0))) {
  stop("Deterministic rerun check failed.", call. = FALSE)
}
if (!identical(first$som_labels, second$som_labels)) {
  stop("Deterministic SOM assignment check failed.", call. = FALSE)
}

direct_input <- build_cycombine_input(
  fixture, markers, "cell_id", "dataset", "unit_key", batch_levels
)
direct <- suppressMessages(cyCombine::batch_correct(
  df = direct_input,
  label = NULL,
  xdim = 8,
  ydim = 8,
  rlen = 10,
  parametric = TRUE,
  seed = 840,
  covar = NULL,
  anchor = NULL,
  markers = markers,
  norm_method = "rank",
  ties.method = "average"
))
direct_expression <- as.data.frame(direct[, markers, drop = FALSE], check.names = FALSE)
max_abs_error <- max(abs(as.matrix(first$corrected_expression) - as.matrix(direct_expression)))
if (!is.finite(max_abs_error) || max_abs_error != 0) {
  stop(sprintf("Direct official-call parity failed; max absolute error = %.17g", max_abs_error), call. = FALSE)
}

constant_fixture <- fixture
constant_fixture$CD3[constant_fixture$dataset == batch_levels[1L]] <- 2
constant_result <- NULL
invisible(capture.output(
  constant_result <- suppressMessages(fit_cycombine_publication(
    input = constant_fixture,
    markers = markers,
    cell_id_column = "cell_id",
    batch_column = "dataset",
    sample_column = "unit_key",
    batch_levels = batch_levels,
    parameters = parameters,
    required_library = local_library
  )),
  type = "output"
))
if (!all(vapply(constant_result$corrected_expression, function(x) all(is.finite(x)), logical(1)))) {
  stop("Officially accepted within-batch constant-marker regression failed.", call. = FALSE)
}

if (any(c("provider_celltype", "survival_outcome") %in% first$fitting_input_columns)) {
  stop("A label or outcome leaked into the fitting columns.", call. = FALSE)
}
if (!all(c("provider_celltype", "survival_outcome") %in% first$ignored_input_columns)) {
  stop("Ignored-column audit is incomplete.", call. = FALSE)
}
if (!identical(first$provenance_only_input_columns, c("cell_id", "unit_key"))) {
  stop("Provenance-only input-column audit is incorrect.", call. = FALSE)
}
if (nrow(first$node_correction_modes) != 64L ||
    any(!first$node_correction_modes$correction_mode %in% c(
      "empty", "single_batch_passthrough", "multi_batch_mean_only_combat",
      "multi_batch_full_combat"
    ))) {
  stop("SOM node correction-mode audit is incomplete.", call. = FALSE)
}
if (!identical(first$cell_ids, fixture$cell_id)) {
  stop("Cell ID order was not preserved.", call. = FALSE)
}
if (any(first$som_labels < 1L | first$som_labels > 64L)) {
  stop("SOM labels fall outside the locked 8x8 grid.", call. = FALSE)
}

cat(sprintf(
  "PASS: cyCombine %s; %d cells; %d occupied SOM nodes; deterministic; direct-call max abs error %.1f; constant-marker domain preserved\n",
  as.character(utils::packageVersion("cyCombine")),
  nrow(fixture),
  length(unique(first$som_labels)),
  max_abs_error
))
