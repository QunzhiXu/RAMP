#!/usr/bin/env Rscript

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_argument)) stop("Cannot determine test location.", call. = FALSE)
script_path <- normalizePath(sub("^--file=", "", script_argument[[1L]]), mustWork = TRUE)
algorithm_dir <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)
local_library <- file.path(algorithm_dir, ".Rlib")
.libPaths(c(local_library, .libPaths()))
source(file.path(algorithm_dir, "fastmnn_repro.R"), local = FALSE)

assert_true <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
}

assert_same_numeric <- function(left, right, tolerance = 0, message) {
  if (!identical(dim(left), dim(right)) ||
      !isTRUE(all.equal(
        unname(as.matrix(left)), unname(as.matrix(right)),
        tolerance = tolerance, check.attributes = FALSE
      ))) {
    stop(message, call. = FALSE)
  }
}

validate_fastmnn_install(local_library)

# Three batches share two biological populations, with small batch-specific shifts.
# Rows are deliberately interleaved so that preserving input cell order is tested.
set.seed(791L)
batch_levels <- c("dataset_C", "dataset_A", "dataset_B")
n_per_batch <- 60L
n_markers <- 7L
batch <- rep(batch_levels, each = n_per_batch)
state <- rep(rep(c("state_1", "state_2"), each = n_per_batch / 2L), 3L)
base <- matrix(rexp(length(batch) * n_markers, rate = 2), ncol = n_markers)
base[state == "state_2", 1:3] <- base[state == "state_2", 1:3] + 1.4
batch_shift <- c(dataset_C = 0.00, dataset_A = 0.18, dataset_B = -0.08)
values <- base + batch_shift[batch]
values <- values + 0.25
colnames(values) <- paste0("marker_", seq_len(n_markers))

input <- data.frame(
  cell_id = sprintf("cell_%04d", seq_len(nrow(values))),
  dataset = batch,
  unit_key = paste0(batch, "_donor_", rep(seq_len(6L), each = 10L, times = 3L)),
  values,
  provider_celltype = state,
  survival_outcome = rep(c("event", "censored"), length.out = nrow(values)),
  stringsAsFactors = FALSE,
  check.names = FALSE
)
shuffle <- sample(seq_len(nrow(input)))
input <- input[shuffle, , drop = FALSE]
rownames(input) <- NULL
markers <- colnames(values)

run_wrapper <- function(table) {
  fit_fastmnn_official(
    input = table,
    markers = markers,
    cell_id_column = "cell_id",
    batch_column = "dataset",
    sample_column = "unit_key",
    batch_levels = batch_levels,
    parameters = canonical_fastmnn_parameters(),
    required_library = local_library
  )
}

# batchelor 1.18.1 intentionally retains an all-zero cell during cosine
# normalization via pmax(1e-8, l2). The wrapper must not impose a stricter
# deletion/pseudocount rule than the official method.
zero_l2_input <- input
zero_l2_input[1L, markers] <- 0
zero_l2_prepared <- prepare_fastmnn_input(
  input = zero_l2_input,
  markers = markers,
  cell_id_column = "cell_id",
  batch_column = "dataset",
  sample_column = "unit_key",
  batch_levels = batch_levels
)
assert_true(
  identical(unname(zero_l2_prepared$l2_norm[1L]), 0),
  "The wrapper did not faithfully retain the official zero-L2 input case."
)
zero_l2_normalized <- batchelor::cosineNorm(zero_l2_prepared$expression)
assert_true(
  all(zero_l2_normalized[, 1L] == 0),
  "Official cosineNorm did not leave the all-zero cell unchanged."
)
zero_l2_fit <- run_wrapper(zero_l2_input)
assert_true(
  all(is.finite(zero_l2_fit$corrected_embedding)) &&
    sum(zero_l2_fit$cell_l2_statistics$zero_l2_n) == 1L,
  "The wrapper failed to retain and audit one official zero-L2 cell."
)

fit_1 <- run_wrapper(input)
fit_2 <- run_wrapper(input)

assert_true(
  identical(fit_1$corrected_embedding, fit_2$corrected_embedding),
  "Two canonical wrapper runs with the same seed were not exactly identical."
)
assert_true(
  identical(rownames(fit_1$corrected_embedding), input$cell_id),
  "The wrapper did not preserve the original interleaved cell order."
)
assert_true(
  identical(fit_1$cell_metadata$cell_id, input$cell_id),
  "Metadata cell order differs from input order."
)
assert_true(
  identical(fit_1$merge$steps$left_batches, c("dataset_C", "dataset_C;dataset_A")) &&
    identical(fit_1$merge$steps$right_batches, c("dataset_A", "dataset_B")),
  "Progressive merge order does not follow configured batch_levels."
)
assert_true(
  all(c("provider_celltype", "survival_outcome") %in% fit_1$ignored_input_columns),
  "Provider labels/outcomes were not recorded as ignored columns."
)
assert_true(
  identical(fit_1$pca$requested_d, 50L) &&
    identical(fit_1$pca$resolved_d, n_markers - 1L) &&
    identical(fit_1$pca$actual_d, n_markers - 1L),
  "Small-panel d=50 resolution was not recorded correctly."
)

# Direct-call parity uses an independently constructed feature-by-cell matrix,
# so this test would detect any future transformation added to prepare_fastmnn_input().
direct_marker_matrix <- as.matrix(input[, markers, drop = FALSE])
storage.mode(direct_marker_matrix) <- "double"
direct_expression <- t(direct_marker_matrix)
rownames(direct_expression) <- markers
colnames(direct_expression) <- input$cell_id
direct_batch_factor <- factor(input$dataset, levels = batch_levels)
parameters <- canonical_fastmnn_parameters()
RNGkind(parameters$rng_kind, parameters$normal_kind, parameters$sample_kind)
set.seed(parameters$seed)
direct <- suppressWarnings(suppressMessages(batchelor::fastMNN(
  direct_expression,
  batch = direct_batch_factor,
  k = parameters$k,
  prop.k = parameters$prop_k,
  restrict = NULL,
  cos.norm = parameters$cos_norm,
  ndist = parameters$ndist,
  d = parameters$requested_d,
  deferred = parameters$deferred,
  weights = parameters$weights,
  get.variance = parameters$get_variance,
  merge.order = parameters$merge_order,
  auto.merge = parameters$auto_merge,
  min.batch.skip = parameters$min_batch_skip,
  subset.row = parameters$subset_row,
  correct.all = parameters$correct_all,
  BSPARAM = BiocSingular::IrlbaParam(),
  BNPARAM = BiocNeighbors::KmknnParam(),
  BPPARAM = BiocParallel::SerialParam()
)))
direct_embedding <- as.matrix(SingleCellExperiment::reducedDim(direct, "corrected"))
direct_embedding <- direct_embedding[
  match(input$cell_id, colnames(direct)), , drop = FALSE
]
assert_same_numeric(
  fit_1$corrected_embedding, direct_embedding, tolerance = 0,
  message = "Wrapper coordinates differ from a direct official fastMNN call."
)

# Diagnostic-only get.variance must not alter the corrected representation.
RNGkind(parameters$rng_kind, parameters$normal_kind, parameters$sample_kind)
set.seed(parameters$seed)
without_variance <- suppressWarnings(suppressMessages(batchelor::fastMNN(
  direct_expression,
  batch = direct_batch_factor,
  k = parameters$k,
  prop.k = parameters$prop_k,
  restrict = NULL,
  cos.norm = parameters$cos_norm,
  ndist = parameters$ndist,
  d = parameters$requested_d,
  deferred = parameters$deferred,
  weights = parameters$weights,
  get.variance = FALSE,
  merge.order = parameters$merge_order,
  auto.merge = parameters$auto_merge,
  min.batch.skip = parameters$min_batch_skip,
  subset.row = parameters$subset_row,
  correct.all = parameters$correct_all,
  BSPARAM = BiocSingular::IrlbaParam(),
  BNPARAM = BiocNeighbors::KmknnParam(),
  BPPARAM = BiocParallel::SerialParam()
)))
without_variance_embedding <- as.matrix(
  SingleCellExperiment::reducedDim(without_variance, "corrected")
)
without_variance_embedding <- without_variance_embedding[
  match(input$cell_id, colnames(without_variance)), , drop = FALSE
]
assert_same_numeric(
  fit_1$corrected_embedding, without_variance_embedding, tolerance = 0,
  message = "get.variance changed corrected coordinates."
)

# Labels and donor provenance are not fitting inputs.
label_changed <- input
label_changed$provider_celltype <- paste0("fake_label_", seq_len(nrow(label_changed)))
label_changed$survival_outcome <- rev(label_changed$survival_outcome)
label_changed$unit_key <- paste0("replacement_sample_", seq_len(nrow(label_changed)))
fit_label_changed <- run_wrapper(label_changed)
assert_true(
  identical(fit_1$corrected_embedding, fit_label_changed$corrected_embedding),
  "Changing labels/outcomes/sample provenance changed the fitted embedding."
)

noncanonical <- canonical_fastmnn_parameters()
noncanonical$requested_d <- NA_integer_
assert_true(
  inherits(try(validate_canonical_parameters(noncanonical), silent = TRUE), "try-error"),
  "The canonical validator accepted marker-space d=NA."
)

# The wrapper must not add a stricter batch-size rule than official fastMNN.
# batchelor 1.18.1 handles a batch smaller than k=20 internally.
small_input <- do.call(rbind, lapply(batch_levels, function(level) {
  utils::head(input[input$dataset == level, , drop = FALSE], 10L)
}))
rownames(small_input) <- NULL
small_fit <- run_wrapper(small_input)
assert_true(
  identical(dim(small_fit$corrected_embedding), c(30L, n_markers - 1L)),
  "The wrapper no longer follows official small-batch k handling."
)

cat(sprintf(
  "PASS: smoke test; %d cells, %d batches, %d markers, %d corrected dimensions.\n",
  nrow(input), length(batch_levels), length(markers), ncol(fit_1$corrected_embedding)
))
