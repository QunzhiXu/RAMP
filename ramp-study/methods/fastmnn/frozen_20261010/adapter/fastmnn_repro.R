FASTMNN_REPRO_SCHEMA <- "final_xu.fastmnn_repro.v1"
FASTMNN_METHOD_LABEL <- "fastMNN (batchelor 1.18.1; corrected embedding)"
FASTMNN_EXPECTED_VERSION <- "1.18.1"
FASTMNN_SOURCE_SHA256 <- "a58590880cf00dd2ff7c9fc75ee611e4d4cc5bae98246a1878f8d7d1da6c8dfc"
FASTMNN_UPSTREAM_COMMIT <- "c427893"
FASTMNN_RUNTIME_LOCK_SHA256 <- "87ffc357386d6f5190aac2583538ce1b3d73040bdae745ccf9d9f620ed2ab8f5"
FASTMNN_FUNCTION_BODY_SHA256 <- c(
  fastMNN = "02354b68a9c5c163d8ad920f3e974200a2cd5b1db11bda3dba8ff82dcf877a82",
  multiBatchPCA = "60fbaf8cbd2effd941abc5eb7d92b6c5c64ed1b4aeb4a1301b5392af38158915",
  .fast_mnn_core = "78b12a8144bb128ec7d36d3c742b9be95f0d71aad6c5dfdbe44d7ba6574f6336",
  .average_correction = "3530e1a314f7a4bb6a1f62cbf378d654df342e2201b5fe3f9a4f17978b80f076",
  .tricube_weighted_correction = "0dcf11f02667a823c89f6e6067b6a72885978998f7c197558d3b903912043419"
)

stopf <- function(fmt, ...) {
  stop(sprintf(fmt, ...), call. = FALSE)
}

`%||%` <- function(left, right) if (is.null(left)) right else left

sha256_file <- function(path) {
  if (!requireNamespace("digest", quietly = TRUE)) {
    stopf("Package digest is required for SHA-256 verification.")
  }
  digest::digest(file = path, algo = "sha256", serialize = FALSE)
}

canonical_fastmnn_parameters <- function() {
  list(
    seed = 42L,
    rng_kind = "Mersenne-Twister",
    normal_kind = "Inversion",
    sample_kind = "Rejection",
    k = 20L,
    prop_k = NULL,
    cos_norm = TRUE,
    ndist = 3,
    requested_d = 50L,
    d_policy = "request 50; batchelor realizes min(50, n_markers - 1, n_cells - 1)",
    deferred = TRUE,
    weights = NULL,
    get_variance = TRUE,
    merge_order = NULL,
    auto_merge = FALSE,
    min_batch_skip = 0,
    subset_row = NULL,
    correct_all = FALSE,
    pca_solver = "IrlbaParam",
    neighbor_solver = "KmknnParam",
    parallel_solver = "SerialParam"
  )
}

validate_canonical_parameters <- function(parameters) {
  expected <- canonical_fastmnn_parameters()
  if (!is.list(parameters) || !setequal(names(parameters), names(expected))) {
    stopf(
      "algorithm_parameters differ from the canonical schema. Missing: [%s]. Extra: [%s].",
      paste(setdiff(names(expected), names(parameters)), collapse = ", "),
      paste(setdiff(names(parameters), names(expected)), collapse = ", ")
    )
  }

  integer_fields <- c("seed", "k", "requested_d")
  numeric_fields <- c("ndist", "min_batch_skip")
  logical_fields <- c(
    "cos_norm", "deferred", "get_variance", "auto_merge", "correct_all"
  )
  character_fields <- c(
    "rng_kind", "normal_kind", "sample_kind", "d_policy", "pca_solver",
    "neighbor_solver", "parallel_solver"
  )

  for (field in integer_fields) {
    value <- parameters[[field]]
    if (length(value) != 1L || is.na(value) || !is.numeric(value) || value != as.integer(value)) {
      stopf("algorithm_parameters.%s must be one integer.", field)
    }
    parameters[[field]] <- as.integer(value)
  }
  for (field in numeric_fields) {
    value <- parameters[[field]]
    if (length(value) != 1L || is.na(value) || !is.numeric(value) || !is.finite(value)) {
      stopf("algorithm_parameters.%s must be one finite number.", field)
    }
    parameters[[field]] <- as.numeric(value)
  }
  for (field in logical_fields) {
    value <- parameters[[field]]
    if (length(value) != 1L || is.na(value) || !is.logical(value)) {
      stopf("algorithm_parameters.%s must be one logical value.", field)
    }
  }
  for (field in character_fields) {
    value <- parameters[[field]]
    if (length(value) != 1L || is.na(value) || !is.character(value) || !nzchar(value)) {
      stopf("algorithm_parameters.%s must be one non-empty string.", field)
    }
  }
  for (field in c("prop_k", "weights", "merge_order", "subset_row")) {
    if (!is.null(parameters[[field]])) {
      stopf(
        paste0(
          "algorithm_parameters.%s must be null in the canonical profile. ",
          "Put non-default sensitivities in a separate benchmark-owned configuration."
        ),
        field
      )
    }
  }

  for (field in names(expected)) {
    if (!identical(parameters[[field]], expected[[field]])) {
      stopf(
        paste0(
          "algorithm_parameters.%s differs from the locked official-default profile. ",
          "Do not tune the canonical comparator from this runner."
        ),
        field
      )
    }
  }
  expected
}

source_archive_path <- function(algorithm_dir) {
  file.path(algorithm_dir, "vendor", "batchelor_1.18.1.tar.gz")
}

current_build_environment_record <- function() {
  info <- sessionInfo()
  data.frame(
    key = c(
      "R_version", "R_platform", "R_arch", "OS_type", "sysname", "machine",
      "dynlib_ext", "BLAS", "LAPACK"
    ),
    value = c(
      as.character(getRversion()), R.version$platform, R.version$arch,
      .Platform$OS.type, unname(Sys.info()[["sysname"]]),
      unname(Sys.info()[["machine"]]), .Platform$dynlib.ext,
      info$BLAS %||% "", info$LAPACK %||% ""
    ),
    stringsAsFactors = FALSE
  )
}

expected_installed_fingerprint_files <- function() {
  sort(c(
    "batchelor/DESCRIPTION",
    "batchelor/NAMESPACE",
    "batchelor/Meta/nsInfo.rds",
    "batchelor/Meta/package.rds",
    "batchelor/R/batchelor",
    "batchelor/R/batchelor.rdb",
    "batchelor/R/batchelor.rdx",
    paste0("batchelor/libs/batchelor", .Platform$dynlib.ext)
  ))
}

installed_version_without_loading <- function(package, required_library = NULL) {
  if (identical(package, "R")) return(as.character(getRversion()))
  lib_locations <- if (identical(package, "batchelor")) required_library else .libPaths()
  package_path <- find.package(package, lib.loc = lib_locations, quiet = TRUE)
  if (!length(package_path) || !nzchar(package_path)) return(NA_character_)
  description <- file.path(package_path, "DESCRIPTION")
  if (!file.exists(description)) return(NA_character_)
  tryCatch(
    as.character(package_version(read.dcf(description, fields = "Version")[[1L]])),
    error = function(...) NA_character_
  )
}

validate_runtime_lock <- function(algorithm_dir, required_library) {
  lock_path <- file.path(algorithm_dir, "runtime_lock.csv")
  if (!file.exists(lock_path)) stopf("Missing runtime_lock.csv.")
  if (!identical(sha256_file(lock_path), FASTMNN_RUNTIME_LOCK_SHA256)) {
    stopf("runtime_lock.csv differs from the audited lock.")
  }
  lock <- utils::read.csv(lock_path, stringsAsFactors = FALSE, check.names = FALSE)
  if (!identical(colnames(lock), c("package", "version", "role")) ||
      anyDuplicated(lock$package) || any(!nzchar(lock$package)) ||
      any(!nzchar(lock$version)) || any(!nzchar(lock$role))) {
    stopf("runtime_lock.csv has an invalid schema.")
  }
  actual <- vapply(
    lock$package,
    installed_version_without_loading,
    character(1),
    required_library = required_library
  )
  bad <- is.na(actual) | actual != lock$version
  if (any(bad)) {
    stopf(
      "Runtime lock mismatch: %s.",
      paste(sprintf(
        "%s expected %s found %s",
        lock$package[bad], lock$version[bad], actual[bad]
      ), collapse = "; ")
    )
  }
  invisible(lock)
}

validate_fastmnn_install <- function(required_library) {
  required_library <- normalizePath(required_library, mustWork = TRUE)
  algorithm_dir <- dirname(required_library)

  archive <- source_archive_path(algorithm_dir)
  if (!file.exists(archive)) stopf("Missing vendored batchelor source archive.")
  if (!identical(sha256_file(archive), FASTMNN_SOURCE_SHA256)) {
    stopf("Vendored batchelor source archive fails SHA-256 verification.")
  }
  validate_runtime_lock(algorithm_dir, required_library)

  source_stamp <- file.path(required_library, ".batchelor_source_sha256")
  if (!file.exists(source_stamp) ||
      !identical(trimws(readLines(source_stamp, warn = FALSE, n = 1L)), FASTMNN_SOURCE_SHA256)) {
    stopf("Dedicated batchelor source stamp is missing or incorrect.")
  }

  build_record_path <- file.path(required_library, "build_environment.csv")
  if (!file.exists(build_record_path)) stopf("Missing build_environment.csv.")
  recorded_build <- utils::read.csv(
    build_record_path, stringsAsFactors = FALSE, check.names = FALSE
  )
  if (!identical(recorded_build, current_build_environment_record())) {
    stopf("The compiled fastMNN runtime does not match the current R/platform ABI.")
  }

  fingerprint_path <- file.path(required_library, "installed_package_fingerprints.csv")
  if (!file.exists(fingerprint_path)) stopf("Missing installed package fingerprints.")
  fingerprints <- utils::read.csv(
    fingerprint_path, stringsAsFactors = FALSE, check.names = FALSE
  )
  expected_files <- expected_installed_fingerprint_files()
  if (!identical(colnames(fingerprints), c("file", "sha256")) ||
      !identical(fingerprints$file, expected_files) || anyDuplicated(fingerprints$file) ||
      any(!grepl("^[0-9a-f]{64}$", fingerprints$sha256))) {
    stopf("Installed package fingerprint manifest is incomplete or invalid.")
  }
  installed_files <- file.path(required_library, expected_files)
  if (any(!file.exists(installed_files))) stopf("A fingerprinted batchelor file is missing.")
  current_hashes <- vapply(installed_files, sha256_file, character(1))
  if (!identical(unname(current_hashes), fingerprints$sha256)) {
    stopf("Installed batchelor files differ from their post-build fingerprints.")
  }

  if (!requireNamespace("batchelor", quietly = TRUE)) {
    stopf("batchelor is unavailable. Run bootstrap_library.R first.")
  }
  expected_path <- normalizePath(file.path(required_library, "batchelor"), mustWork = TRUE)
  actual_path <- normalizePath(find.package("batchelor"), mustWork = TRUE)
  if (!identical(actual_path, expected_path)) {
    stopf("batchelor loaded from %s instead of %s.", actual_path, expected_path)
  }
  actual_version <- as.character(utils::packageVersion("batchelor"))
  if (!identical(actual_version, FASTMNN_EXPECTED_VERSION)) {
    stopf("Expected batchelor %s but found %s.", FASTMNN_EXPECTED_VERSION, actual_version)
  }

  expected_formals <- c(
    "...", "batch", "k", "prop.k", "restrict", "cos.norm", "ndist", "d",
    "deferred", "weights", "get.variance", "merge.order", "auto.merge",
    "min.batch.skip", "subset.row", "correct.all", "assay.type", "BSPARAM",
    "BNPARAM", "BPPARAM"
  )
  if (!identical(names(formals(batchelor::fastMNN)), expected_formals)) {
    stopf("batchelor::fastMNN API does not match version 1.18.1.")
  }

  namespace <- asNamespace("batchelor")
  actual_function_hashes <- vapply(
    names(FASTMNN_FUNCTION_BODY_SHA256),
    function(function_name) {
      function_object <- get(function_name, envir = namespace, inherits = FALSE)
      digest::digest(
        paste(deparse(body(function_object), width.cutoff = 500L), collapse = "\n"),
        algo = "sha256", serialize = FALSE
      )
    },
    character(1)
  )
  if (!identical(actual_function_hashes, FASTMNN_FUNCTION_BODY_SHA256)) {
    stopf(
      "Installed batchelor function bodies differ from the verified source: %s.",
      paste(names(actual_function_hashes)[
        actual_function_hashes != FASTMNN_FUNCTION_BODY_SHA256
      ], collapse = ", ")
    )
  }

  invisible(list(
    package = "batchelor",
    package_version = actual_version,
    installed_path = actual_path,
    source_sha256 = FASTMNN_SOURCE_SHA256,
    upstream_commit = FASTMNN_UPSTREAM_COMMIT,
    function_body_sha256 = as.list(actual_function_hashes)
  ))
}

read_marker_file <- function(path) {
  markers <- trimws(readLines(path, warn = FALSE))
  markers <- markers[nzchar(markers)]
  if (length(markers) < 3L) {
    stopf("The canonical fastMNN wrapper requires at least three shared markers.")
  }
  if (anyDuplicated(markers)) {
    stopf(
      "The marker file contains duplicate names: %s.",
      paste(unique(markers[duplicated(markers)]), collapse = ", ")
    )
  }
  markers
}

read_analysis_table <- function(path, character_columns = character()) {
  lower <- tolower(path)
  if (grepl("\\.rds$", lower)) {
    object <- readRDS(path)
  } else if (grepl("\\.csv(\\.gz)?$", lower)) {
    if (!requireNamespace("readr", quietly = TRUE)) {
      stopf("Package readr is required for CSV input.")
    }
    character_spec <- stats::setNames(
      lapply(character_columns, function(...) readr::col_character()),
      character_columns
    )
    column_spec <- do.call(
      readr::cols,
      c(list(.default = readr::col_guess()), character_spec)
    )
    object <- suppressMessages(readr::read_csv(
      path,
      col_types = column_spec,
      show_col_types = FALSE,
      progress = FALSE,
      name_repair = "minimal"
    ))
  } else {
    stopf("Unsupported input format. Use .csv, .csv.gz, or .rds.")
  }
  if (!is.data.frame(object)) stopf("Input must resolve to a data.frame/tibble.")
  if (anyDuplicated(colnames(object))) {
    stopf(
      "Input contains duplicate columns: %s.",
      paste(unique(colnames(object)[duplicated(colnames(object))]), collapse = ", ")
    )
  }
  as.data.frame(object, stringsAsFactors = FALSE, check.names = FALSE)
}

prepare_fastmnn_input <- function(
    input,
    markers,
    cell_id_column,
    batch_column,
    sample_column,
    batch_levels) {
  required <- c(cell_id_column, batch_column, sample_column, markers)
  missing <- setdiff(required, colnames(input))
  if (length(missing)) stopf("Input is missing columns: %s.", paste(missing, collapse = ", "))
  if (anyDuplicated(required)) {
    stopf("Cell ID, batch, sample, and marker columns must all be distinct.")
  }

  cell_ids <- as.character(input[[cell_id_column]])
  batches <- as.character(input[[batch_column]])
  samples <- as.character(input[[sample_column]])
  if (anyNA(cell_ids) || any(!nzchar(cell_ids)) || anyDuplicated(cell_ids)) {
    stopf("Cell IDs must be complete, non-empty, and globally unique.")
  }
  if (anyNA(batches) || any(!nzchar(batches)) || anyNA(samples) || any(!nzchar(samples))) {
    stopf("Batch and sample provenance must be complete and non-empty.")
  }
  if (!is.character(batch_levels) || length(batch_levels) < 2L ||
      anyNA(batch_levels) || any(!nzchar(batch_levels)) || anyDuplicated(batch_levels)) {
    stopf("batch_levels must contain at least two unique non-empty names.")
  }
  if (!setequal(unique(batches), batch_levels)) {
    stopf(
      "Observed batches [%s] do not match configured batch_levels [%s].",
      paste(unique(batches), collapse = ", "), paste(batch_levels, collapse = ", ")
    )
  }

  marker_table <- input[, markers, drop = FALSE]
  non_numeric <- markers[!vapply(marker_table, is.numeric, logical(1))]
  if (length(non_numeric)) {
    stopf("Marker columns must be numeric: %s.", paste(non_numeric, collapse = ", "))
  }
  marker_matrix <- as.matrix(marker_table)
  storage.mode(marker_matrix) <- "double"
  if (any(!is.finite(marker_matrix))) stopf("Marker matrix contains NA, NaN, or Inf.")
  global_marker_variance <- apply(marker_matrix, 2L, stats::var)
  if (any(!is.finite(global_marker_variance) | global_marker_variance <= 0)) {
    bad <- markers[!is.finite(global_marker_variance) | global_marker_variance <= 0]
    stopf(
      "Every shared marker must vary across the combined input; invalid: %s.",
      paste(bad, collapse = ", ")
    )
  }

  l2_norm <- sqrt(rowSums(marker_matrix^2))
  if (any(!is.finite(l2_norm))) {
    bad <- cell_ids[!is.finite(l2_norm)]
    stopf(
      "Cell L2 norms are non-finite for %d cell(s), e.g. %s.",
      length(bad), paste(utils::head(bad, 5L), collapse = ", ")
    )
  }

  # Do not reject all-zero cells here. batchelor 1.18.1 deliberately handles
  # them inside cosineNorm() by replacing the divisor with pmax(1e-8, l2),
  # which leaves an all-zero column unchanged. Retaining that upstream behavior
  # is more faithful than deleting cells or inventing an epsilon preprocessing
  # step in this wrapper.

  batch_factor <- factor(batches, levels = batch_levels)

  expression <- t(marker_matrix)
  rownames(expression) <- markers
  colnames(expression) <- cell_ids
  list(
    expression = expression,
    cell_ids = cell_ids,
    batch_factor = batch_factor,
    batches = batches,
    samples = samples,
    l2_norm = l2_norm,
    ignored_columns = setdiff(colnames(input), required),
    fitting_input_columns = c(batch_column, markers),
    provenance_only_input_columns = c(cell_id_column, sample_column)
  )
}

input_marker_batch_statistics <- function(prepared, markers, batch_levels) {
  values <- t(prepared$expression)
  rows <- vector("list", length(batch_levels) * length(markers))
  cursor <- 1L
  for (batch in batch_levels) {
    index <- prepared$batches == batch
    for (marker in markers) {
      marker_values <- values[index, marker]
      rows[[cursor]] <- data.frame(
        batch = batch,
        marker = marker,
        n = length(marker_values),
        minimum = min(marker_values),
        maximum = max(marker_values),
        mean = mean(marker_values),
        sd = stats::sd(marker_values),
        zero_fraction = mean(marker_values == 0),
        zero_variance = stats::var(marker_values) == 0,
        stringsAsFactors = FALSE
      )
      cursor <- cursor + 1L
    }
  }
  do.call(rbind, rows)
}

format_batch_set <- function(value, batch_levels) {
  value <- unlist(value, use.names = FALSE)
  if (!length(value)) return("")
  if (is.numeric(value) || is.integer(value)) {
    indices <- as.integer(value)
    if (any(indices < 1L | indices > length(batch_levels))) {
      stopf("fastMNN merge metadata contains an invalid batch index.")
    }
    value <- batch_levels[indices]
  }
  paste(as.character(value), collapse = ";")
}

extract_merge_diagnostics <- function(result, batch_levels) {
  merge_info <- S4Vectors::metadata(result)$merge.info
  if (is.null(merge_info) || nrow(merge_info) != length(batch_levels) - 1L) {
    stopf("fastMNN merge.info is absent or has an unexpected number of steps.")
  }
  required_fields <- c("left", "right", "pairs", "batch.size", "skipped", "lost.var")
  if (!all(required_fields %in% colnames(merge_info))) {
    stopf("fastMNN merge.info is missing required fields.")
  }
  steps <- data.frame(
    merge_step = seq_len(nrow(merge_info)),
    left_batches = vapply(merge_info$left, format_batch_set, character(1), batch_levels = batch_levels),
    right_batches = vapply(merge_info$right, format_batch_set, character(1), batch_levels = batch_levels),
    mnn_pairs = vapply(merge_info$pairs, nrow, integer(1)),
    relative_batch_effect_magnitude = as.numeric(merge_info$batch.size),
    skipped = as.logical(merge_info$skipped),
    stringsAsFactors = FALSE
  )
  lost <- as.matrix(merge_info$lost.var)
  if (!identical(dim(lost), c(length(batch_levels) - 1L, length(batch_levels)))) {
    stopf("fastMNN lost.var has unexpected dimensions.")
  }
  if (!identical(colnames(lost), batch_levels)) {
    stopf("fastMNN lost.var batch order differs from the configured batch order.")
  }
  lost_long <- do.call(rbind, lapply(seq_len(nrow(lost)), function(step) {
    data.frame(
      merge_step = step,
      batch = batch_levels,
      lost_variance_fraction = as.numeric(lost[step, ]),
      lost_variance_percent = 100 * as.numeric(lost[step, ]),
      review_flag_over_10_percent = as.numeric(lost[step, ]) > 0.10,
      stringsAsFactors = FALSE
    )
  }))
  list(raw = merge_info, steps = steps, lost_variance = lost_long)
}

extract_pca_diagnostics <- function(result, markers, requested_d, resolved_d) {
  pca_info <- S4Vectors::metadata(result)$pca.info
  corrected <- SingleCellExperiment::reducedDim(result, "corrected")
  rotation <- SummarizedExperiment::rowData(result)$rotation
  if (is.null(pca_info) || is.null(rotation)) stopf("fastMNN PCA diagnostics are incomplete.")
  rotation <- as.matrix(rotation)
  actual_d <- ncol(corrected)
  if (nrow(rotation) != length(markers) || ncol(rotation) != actual_d) {
    stopf("fastMNN PCA rotation dimensions do not match markers and corrected embedding.")
  }
  rownames(rotation) <- markers
  colnames(rotation) <- sprintf("PC%02d", seq_len(actual_d))

  centers <- as.numeric(pca_info$centers)
  if (length(centers) != length(markers)) stopf("fastMNN PCA centers are incomplete.")
  names(centers) <- markers
  variance <- as.numeric(pca_info$var.explained)
  total <- as.numeric(pca_info$var.total)
  if (length(variance) != actual_d || length(total) != 1L || !is.finite(total) || total <= 0) {
    stopf("fastMNN PCA variance diagnostics are incomplete.")
  }
  variance_table <- data.frame(
    component = sprintf("PC%02d", seq_len(actual_d)),
    variance_explained = variance,
    fraction_of_total_variance = variance / total,
    cumulative_fraction_of_total_variance = cumsum(variance / total),
    requested_d = requested_d,
    resolved_d = resolved_d,
    actual_d = actual_d,
    stringsAsFactors = FALSE
  )
  list(
    requested_d = requested_d,
    resolved_d = resolved_d,
    actual_d = actual_d,
    centers = data.frame(marker = markers, center = centers, stringsAsFactors = FALSE),
    rotation = rotation,
    variance = variance_table,
    total_variance = total
  )
}

fit_fastmnn_official <- function(
    input,
    markers,
    cell_id_column,
    batch_column,
    sample_column,
    batch_levels,
    parameters = canonical_fastmnn_parameters(),
    required_library) {
  installation <- validate_fastmnn_install(required_library)
  parameters <- validate_canonical_parameters(parameters)
  prepared <- prepare_fastmnn_input(
    input = input,
    markers = markers,
    cell_id_column = cell_id_column,
    batch_column = batch_column,
    sample_column = sample_column,
    batch_levels = batch_levels
  )
  resolved_d <- min(
    parameters$requested_d,
    length(markers) - 1L,
    nrow(input) - 1L
  )
  if (resolved_d < 2L) stopf("Resolved PCA dimensionality must be at least two.")

  previous_rng_kind <- RNGkind()
  on.exit(do.call(RNGkind, as.list(previous_rng_kind)), add = TRUE)
  RNGkind(
    kind = parameters$rng_kind,
    normal.kind = parameters$normal_kind,
    sample.kind = parameters$sample_kind
  )
  set.seed(parameters$seed)

  captured_messages <- character()
  captured_warnings <- character()
  result <- withCallingHandlers(
    batchelor::fastMNN(
      prepared$expression,
      batch = prepared$batch_factor,
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
    ),
    message = function(condition) {
      captured_messages <<- c(captured_messages, conditionMessage(condition))
      invokeRestart("muffleMessage")
    },
    warning = function(condition) {
      captured_warnings <<- c(captured_warnings, conditionMessage(condition))
      invokeRestart("muffleWarning")
    }
  )

  result_ids <- colnames(result)
  if (is.null(result_ids) || anyDuplicated(result_ids)) {
    stopf("fastMNN output lacks unique cell IDs.")
  }
  order_index <- match(prepared$cell_ids, result_ids)
  if (anyNA(order_index)) stopf("fastMNN output cell IDs do not match the input.")
  corrected <- SingleCellExperiment::reducedDim(result, "corrected")
  corrected <- as.matrix(corrected[order_index, , drop = FALSE])
  if (nrow(corrected) != nrow(input) || any(!is.finite(corrected))) {
    stopf("fastMNN returned an invalid corrected embedding.")
  }
  rownames(corrected) <- prepared$cell_ids
  colnames(corrected) <- sprintf("fastMNN_%02d", seq_len(ncol(corrected)))

  merge <- extract_merge_diagnostics(result, batch_levels)
  pca <- extract_pca_diagnostics(
    result, markers, parameters$requested_d, resolved_d
  )
  if (pca$actual_d != resolved_d) {
    stopf(
      "batchelor realized %d PCs, but the locked Irlba policy expected %d.",
      pca$actual_d, resolved_d
    )
  }
  if (pca$actual_d != ncol(corrected)) stopf("PCA and corrected dimensions disagree.")

  list(
    corrected_embedding = corrected,
    cell_metadata = data.frame(
      cell_id = prepared$cell_ids,
      batch = prepared$batches,
      sample = prepared$samples,
      stringsAsFactors = FALSE
    ),
    merge = merge,
    pca = pca,
    marker_batch_statistics = input_marker_batch_statistics(prepared, markers, batch_levels),
    cell_l2_statistics = do.call(rbind, lapply(batch_levels, function(batch) {
      values <- prepared$l2_norm[prepared$batches == batch]
      data.frame(
        batch = batch,
        n = length(values),
        zero_l2_n = sum(values == 0),
        zero_l2_fraction = mean(values == 0),
        minimum = min(values),
        median = stats::median(values),
        maximum = max(values),
        stringsAsFactors = FALSE
      )
    })),
    fitting_input_columns = prepared$fitting_input_columns,
    provenance_only_input_columns = prepared$provenance_only_input_columns,
    ignored_input_columns = prepared$ignored_columns,
    parameters = parameters,
    resolved_d = resolved_d,
    batch_levels = batch_levels,
    installation = installation,
    messages = unique(captured_messages),
    warnings = unique(captured_warnings)
  )
}
