HARMONY_DIRECT_SCHEMA <- "final_xu.harmony_direct.v1"
HARMONY_DIRECT_METHOD_LABEL <- "Harmony-direct (ADTnorm protocol; no PCA)"
HARMONY_EXPECTED_VERSION <- "1.2.3"
HARMONY_SOURCE_SHA256 <- "9ab815ab0260a6726b2bb48de3fa4c3cad36acbe9ac6158d515fb16ea66bbddd"
RCPPPROGRESS_SOURCE_SHA256 <- "b1624b21b7aeb1dafb30f092b2a4bef4c3504efd2d6b00b2cdf55dc9df194b48"
RHPCBLASCTL_SOURCE_SHA256 <- "5c889d5b69e264060b9f1f0383c447f594855b8afc15b7d76d39e4d62b946615"
HARMONY_RUNTIME_LOCK_SHA256 <- "00fb9fe3f19891045bfe5e893ded8dfcc94d74d72ba2b85138caf6f9041b1691"
HARMONY_REQUIRED_RUNTIME_PACKAGES <- c(
  "R", "bit", "bit64", "cli", "clipr", "cowplot", "cpp11", "crayon",
  "digest", "dplyr", "farver", "generics", "ggplot2", "glue", "graphics",
  "grDevices", "grid", "gtable", "harmony", "hms", "isoband", "jsonlite",
  "labeling", "lattice", "lifecycle", "magrittr", "MASS", "Matrix", "methods",
  "mgcv", "nlme", "pillar", "pkgconfig", "prettyunits", "progress", "R6",
  "RColorBrewer", "Rcpp", "RcppArmadillo", "RcppProgress", "readr",
  "RhpcBLASctl", "rlang", "scales", "splines", "stats", "tibble",
  "tidyselect", "tzdb", "utf8", "utils", "vctrs", "viridisLite", "vroom",
  "withr"
)

stopf <- function(fmt, ...) {
  stop(sprintf(fmt, ...), call. = FALSE)
}

`%||%` <- function(left, right) if (is.null(left)) right else left

canonical_harmony_direct_parameters <- function() {
  list(
    seed = 42L,
    rng_kind = "Mersenne-Twister",
    normal_kind = "Inversion",
    sample_kind = "Rejection",
    theta = 2,
    sigma = 0.1,
    lambda = 1,
    nclust_policy = "min(round(n_cells / 30), 100)",
    max_iter = 10L,
    early_stop = TRUE,
    ncores = 1L,
    plot_convergence = FALSE,
    return_object = FALSE,
    alpha = 0.2,
    tau = 0,
    block_size = 0.05,
    max_iter_cluster = 20L,
    epsilon_cluster = 0.001,
    epsilon_harmony = 0.01
  )
}

validate_canonical_parameters <- function(parameters) {
  expected <- canonical_harmony_direct_parameters()
  if (!is.list(parameters) || !setequal(names(parameters), names(expected))) {
    stopf(
      "algorithm_parameters do not match the locked schema. Missing: [%s]. Extra: [%s].",
      paste(setdiff(names(expected), names(parameters)), collapse = ", "),
      paste(setdiff(names(parameters), names(expected)), collapse = ", ")
    )
  }

  integer_fields <- c("seed", "max_iter", "ncores", "max_iter_cluster")
  numeric_fields <- c(
    "theta", "sigma", "lambda", "alpha", "tau", "block_size",
    "epsilon_cluster", "epsilon_harmony"
  )
  logical_fields <- c("early_stop", "plot_convergence", "return_object")
  character_fields <- c("rng_kind", "normal_kind", "sample_kind", "nclust_policy")

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

  for (field in names(expected)) {
    if (!identical(parameters[[field]], expected[[field]])) {
      stopf(
        paste0(
          "algorithm_parameters.%s differs from the locked ADTnorm-style Harmony profile. ",
          "Any predeclared sensitivity run belongs in a benchmark-specific configuration, ",
          "not in this canonical comparator."
        ),
        field
      )
    }
  }
  expected
}

sha256_file <- function(path) {
  if (!requireNamespace("digest", quietly = TRUE)) {
    stopf("Package digest is required for SHA-256 verification.")
  }
  digest::digest(file = path, algo = "sha256", serialize = FALSE)
}

source_archive_paths <- function(algorithm_dir) {
  c(
    harmony = file.path(algorithm_dir, "vendor", "harmony_1.2.3.tar.gz"),
    RcppProgress = file.path(algorithm_dir, "vendor", "RcppProgress_0.4.2.tar.gz"),
    RhpcBLASctl = file.path(algorithm_dir, "vendor", "RhpcBLASctl_0.23-42.tar.gz")
  )
}

source_archive_hashes <- function() {
  c(
    harmony = HARMONY_SOURCE_SHA256,
    RcppProgress = RCPPPROGRESS_SOURCE_SHA256,
    RhpcBLASctl = RHPCBLASCTL_SOURCE_SHA256
  )
}

expected_installed_fingerprint_files <- function(dynlib_ext = .Platform$dynlib.ext) {
  sort(c(
    "harmony/DESCRIPTION",
    "harmony/NAMESPACE",
    "harmony/Meta/nsInfo.rds",
    "harmony/Meta/package.rds",
    "harmony/R/harmony",
    "harmony/R/harmony.rdb",
    "harmony/R/harmony.rdx",
    paste0("harmony/libs/harmony", dynlib_ext),
    "RcppProgress/DESCRIPTION",
    "RcppProgress/NAMESPACE",
    "RcppProgress/Meta/nsInfo.rds",
    "RcppProgress/Meta/package.rds",
    "RcppProgress/include/interruptable_progress_monitor.hpp",
    "RcppProgress/include/interrupts.hpp",
    "RcppProgress/include/progress.hpp",
    "RcppProgress/include/progress_bar.hpp",
    "RcppProgress/include/simple_progress_bar.hpp",
    "RhpcBLASctl/DESCRIPTION",
    "RhpcBLASctl/NAMESPACE",
    "RhpcBLASctl/Meta/nsInfo.rds",
    "RhpcBLASctl/Meta/package.rds",
    "RhpcBLASctl/R/RhpcBLASctl",
    "RhpcBLASctl/R/RhpcBLASctl.rdb",
    "RhpcBLASctl/R/RhpcBLASctl.rdx",
    paste0("RhpcBLASctl/libs/RhpcBLASctl", dynlib_ext)
  ))
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

validate_source_archives <- function(algorithm_dir) {
  archives <- source_archive_paths(algorithm_dir)
  expected_hashes <- source_archive_hashes()
  missing_archives <- names(archives)[!file.exists(archives)]
  if (length(missing_archives)) {
    stopf("Missing vendored source archive(s): %s.", paste(missing_archives, collapse = ", "))
  }
  actual_hashes <- vapply(archives, sha256_file, character(1))
  if (!identical(actual_hashes, expected_hashes)) {
    stopf("One or more vendored source archives fail SHA-256 verification.")
  }
  invisible(actual_hashes)
}

validate_source_stamps <- function(required_library) {
  expected_hashes <- source_archive_hashes()
  stamp_paths <- file.path(required_library, sprintf(".%s_source_sha256", names(expected_hashes)))
  stamp_ok <- vapply(seq_along(stamp_paths), function(index) {
    path <- stamp_paths[[index]]
    file.exists(path) && identical(
      trimws(readLines(path, warn = FALSE, n = 1L)),
      unname(expected_hashes[[index]])
    )
  }, logical(1))
  if (any(!stamp_ok)) {
    stopf("Installed package source stamp mismatch: %s.", paste(names(expected_hashes)[!stamp_ok], collapse = ", "))
  }
  invisible(TRUE)
}

validate_installed_fingerprint_manifest <- function(required_library) {
  fingerprint_path <- file.path(required_library, "installed_package_fingerprints.csv")
  if (!file.exists(fingerprint_path)) stopf("Missing installed-package fingerprint manifest.")
  fingerprints <- tryCatch(
    utils::read.csv(fingerprint_path, stringsAsFactors = FALSE, check.names = FALSE),
    error = function(error) stopf("Could not read installed-package fingerprint manifest: %s.", conditionMessage(error))
  )
  expected_files <- expected_installed_fingerprint_files()
  valid_schema <- identical(colnames(fingerprints), c("file", "sha256")) &&
    !anyDuplicated(fingerprints$file) &&
    identical(fingerprints$file, expected_files) &&
    all(grepl("^[0-9a-f]{64}$", fingerprints$sha256))
  if (!valid_schema) {
    stopf("Installed-package fingerprint manifest does not contain the exact required file set.")
  }
  fingerprint_files <- file.path(required_library, expected_files)
  if (any(!file.exists(fingerprint_files))) {
    stopf("A required fingerprinted installed-package file is missing.")
  }
  current_fingerprints <- vapply(fingerprint_files, sha256_file, character(1))
  if (!identical(unname(current_fingerprints), fingerprints$sha256)) {
    stopf("Installed Harmony/dependency files differ from their post-build fingerprints.")
  }
  invisible(fingerprints)
}

installed_version_without_loading <- function(package, required_library = NULL) {
  if (identical(package, "R")) return(as.character(getRversion()))
  vendored_packages <- c("harmony", "RcppProgress", "RhpcBLASctl")
  library_locations <- if (!is.null(required_library) && package %in% vendored_packages) {
    required_library
  } else {
    .libPaths()
  }
  package_path <- find.package(package, lib.loc = library_locations, quiet = TRUE)
  if (!length(package_path) || !nzchar(package_path)) return(NA_character_)
  description_path <- file.path(package_path, "DESCRIPTION")
  if (!file.exists(description_path)) return(NA_character_)
  version <- tryCatch(
    read.dcf(description_path, fields = "Version")[[1L]],
    error = function(...) NA_character_
  )
  if (is.na(version)) return(NA_character_)
  as.character(package_version(version))
}

validate_runtime_lock_file <- function(algorithm_dir, check_versions = TRUE, required_library = NULL) {
  runtime_lock_path <- file.path(algorithm_dir, "runtime_lock.csv")
  if (!file.exists(runtime_lock_path)) stopf("Missing runtime_lock.csv.")
  if (!identical(sha256_file(runtime_lock_path), HARMONY_RUNTIME_LOCK_SHA256)) {
    stopf("runtime_lock.csv differs from the audited fail-closed lock.")
  }
  runtime_lock <- tryCatch(
    utils::read.csv(runtime_lock_path, stringsAsFactors = FALSE, check.names = FALSE),
    error = function(error) stopf("Could not read runtime_lock.csv: %s.", conditionMessage(error))
  )
  valid_schema <- identical(colnames(runtime_lock), c("package", "version", "role")) &&
    !anyDuplicated(runtime_lock$package) &&
    identical(runtime_lock$package, HARMONY_REQUIRED_RUNTIME_PACKAGES) &&
    all(nzchar(runtime_lock$version)) && all(nzchar(runtime_lock$role))
  if (!valid_schema) stopf("runtime_lock.csv has an invalid or incomplete schema.")
  if (isTRUE(check_versions)) {
    actual_versions <- vapply(
      runtime_lock$package,
      installed_version_without_loading,
      character(1),
      required_library = required_library
    )
    bad_runtime <- is.na(actual_versions) | actual_versions != runtime_lock$version
    if (any(bad_runtime)) {
      stopf(
        "Runtime lock mismatch: %s.",
        paste(sprintf(
          "%s expected %s found %s",
          runtime_lock$package[bad_runtime],
          runtime_lock$version[bad_runtime],
          actual_versions[bad_runtime]
        ), collapse = "; ")
      )
    }
  }
  invisible(runtime_lock)
}

validate_build_environment_record <- function(required_library) {
  build_environment_path <- file.path(required_library, "build_environment.csv")
  if (!file.exists(build_environment_path)) stopf("Missing compiled-library build environment record.")
  recorded <- tryCatch(
    utils::read.csv(build_environment_path, stringsAsFactors = FALSE, check.names = FALSE),
    error = function(error) stopf("Could not read compiled-library build environment record: %s.", conditionMessage(error))
  )
  if (!identical(recorded, current_build_environment_record())) {
    stopf("Compiled .Rlib build platform/R ABI/BLAS differs from the current runtime; rerun bootstrap_library.R.")
  }
  invisible(TRUE)
}

validate_harmony_install <- function(required_library = NULL) {
  algorithm_dir <- NULL
  if (!is.null(required_library)) {
    required_library <- normalizePath(required_library, mustWork = TRUE)
    algorithm_dir <- dirname(required_library)

    # Validate source, installed code, and the full runtime without loading the
    # Harmony namespace or its native library. This ordering is intentional.
    validate_source_archives(algorithm_dir)
    validate_source_stamps(required_library)
    validate_installed_fingerprint_manifest(required_library)
    validate_runtime_lock_file(algorithm_dir, check_versions = TRUE, required_library = required_library)
    validate_build_environment_record(required_library)

    expected_path <- normalizePath(file.path(required_library, "harmony"), mustWork = TRUE)
    discovered_path <- find.package("harmony", lib.loc = required_library, quiet = TRUE)
    if (!length(discovered_path) || !identical(normalizePath(discovered_path, mustWork = TRUE), expected_path)) {
      stopf("Dedicated Harmony installation is unavailable at %s.", expected_path)
    }
  }

  if (!requireNamespace("harmony", quietly = TRUE)) {
    stopf("Harmony is unavailable. Run bootstrap_library.R first.")
  }
  installed_version <- as.character(utils::packageVersion("harmony"))
  if (!identical(installed_version, HARMONY_EXPECTED_VERSION)) {
    stopf("Expected Harmony %s, but loaded %s from %s.", HARMONY_EXPECTED_VERSION, installed_version, find.package("harmony"))
  }
  if (!is.null(required_library)) {
    expected_path <- normalizePath(file.path(required_library, "harmony"), mustWork = TRUE)
    actual_path <- normalizePath(find.package("harmony"), mustWork = TRUE)
    if (!identical(actual_path, expected_path)) {
      stopf("Harmony loaded from %s instead of dedicated library %s.", actual_path, expected_path)
    }
  }

  default_method <- getS3method("RunHarmony", "default", envir = asNamespace("harmony"))
  expected_formals <- c(
    "data_mat", "meta_data", "vars_use", "theta", "sigma", "lambda",
    "nclust", "max_iter", "early_stop", "ncores", "plot_convergence",
    "return_object", "verbose", ".options", "..."
  )
  if (!identical(names(formals(default_method)), expected_formals)) {
    stopf("Harmony RunHarmony.default API differs from the verified 1.2.3 source.")
  }
  options <- harmony::harmony_options()
  expected_options <- c(
    alpha = 0.2, tau = 0, block.size = 0.05, max.iter.cluster = 20,
    epsilon.cluster = 0.001, epsilon.harmony = 0.01
  )
  actual_options <- unlist(options[names(expected_options)], use.names = TRUE)
  if (!identical(actual_options, expected_options)) {
    stopf("Harmony 1.2.3 advanced defaults differ from the locked source audit.")
  }

  invisible(list(
    package_version = installed_version,
    installed_path = normalizePath(find.package("harmony"), mustWork = TRUE),
    source_sha256 = HARMONY_SOURCE_SHA256,
    runharmony_formals = expected_formals,
    advanced_defaults = as.list(expected_options)
  ))
}

read_marker_file <- function(path) {
  markers <- trimws(readLines(path, warn = FALSE))
  markers <- markers[nzchar(markers)]
  if (length(markers) < 2L) stopf("Marker file must contain at least two markers.")
  if (anyDuplicated(markers)) {
    stopf("Marker file contains duplicate names: %s.", paste(unique(markers[duplicated(markers)]), collapse = ", "))
  }
  markers
}

read_analysis_table <- function(path, character_columns = character()) {
  lower <- tolower(path)
  if (grepl("\\.rds$", lower)) {
    object <- readRDS(path)
  } else if (grepl("\\.csv(\\.gz)?$", lower)) {
    if (!requireNamespace("readr", quietly = TRUE)) stopf("Package readr is required for CSV input.")
    character_spec <- stats::setNames(
      lapply(character_columns, function(...) readr::col_character()),
      character_columns
    )
    column_spec <- do.call(readr::cols, c(list(.default = readr::col_guess()), character_spec))
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
  if (is.null(colnames(object)) || anyDuplicated(colnames(object))) {
    stopf("Input column names must be present and unique.")
  }
  as.data.frame(object, check.names = FALSE, stringsAsFactors = FALSE)
}

build_harmony_direct_input <- function(
    input,
    markers,
    cell_id_column,
    batch_column,
    sample_column,
    batch_levels) {
  required <- c(cell_id_column, batch_column, sample_column, markers)
  missing <- setdiff(required, colnames(input))
  if (length(missing)) stopf("Input is missing required columns: %s.", paste(missing, collapse = ", "))
  if (length(unique(c(cell_id_column, batch_column, sample_column, markers))) != length(required)) {
    stopf("ID, batch, sample, and marker column roles must not overlap.")
  }

  cell_ids <- as.character(input[[cell_id_column]])
  batches <- as.character(input[[batch_column]])
  samples <- as.character(input[[sample_column]])
  if (anyNA(cell_ids) || any(!nzchar(cell_ids)) || anyDuplicated(cell_ids)) {
    stopf("Cell IDs must be complete, non-empty, and globally unique.")
  }
  if (anyNA(batches) || any(!nzchar(batches))) stopf("Batch values must be complete and non-empty.")
  if (anyNA(samples) || any(!nzchar(samples))) stopf("Sample values must be complete and non-empty.")
  if (!is.character(batch_levels) || length(batch_levels) < 2L || anyNA(batch_levels) ||
      any(!nzchar(batch_levels)) || anyDuplicated(batch_levels)) {
    stopf("batch_levels must list at least two unique, non-empty batches.")
  }
  observed_levels <- unique(batches)
  if (!setequal(observed_levels, batch_levels)) {
    stopf(
      "Observed batches do not exactly match configured batch_levels. Observed: [%s]; configured: [%s].",
      paste(observed_levels, collapse = ", "), paste(batch_levels, collapse = ", ")
    )
  }

  non_numeric <- markers[!vapply(input[, markers, drop = FALSE], is.numeric, logical(1))]
  if (length(non_numeric)) stopf("Marker columns must be numeric: %s.", paste(non_numeric, collapse = ", "))
  marker_matrix <- as.matrix(input[, markers, drop = FALSE])
  storage.mode(marker_matrix) <- "double"
  if (any(!is.finite(marker_matrix))) stopf("Marker matrix contains NA, NaN, or infinite values.")
  if (nrow(marker_matrix) < 16L) {
    stopf("At least 16 cells are required because Harmony 1.2.3's default nclust policy otherwise resolves below one.")
  }
  rownames(marker_matrix) <- cell_ids
  colnames(marker_matrix) <- markers

  list(
    marker_matrix = marker_matrix,
    cell_ids = cell_ids,
    batches = factor(batches, levels = batch_levels),
    samples = samples,
    metadata = data.frame(
      cell_id = cell_ids,
      batch = batches,
      sample = samples,
      stringsAsFactors = FALSE,
      check.names = FALSE
    ),
    fitting_input_columns = c(batch_column, markers),
    provenance_only_input_columns = c(cell_id_column, sample_column),
    ignored_input_columns = setdiff(colnames(input), required)
  )
}

with_locked_rng <- function(parameters, expression) {
  old_kind <- RNGkind()
  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_seed) old_seed <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  on.exit({
    do.call(RNGkind, as.list(old_kind))
    if (had_seed) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)
  RNGkind(parameters$rng_kind, parameters$normal_kind, parameters$sample_kind)
  set.seed(parameters$seed)
  force(expression)
}

resolved_nclust <- function(n_cells) {
  as.integer(min(round(n_cells / 30), 100))
}

run_harmony_direct_core <- function(marker_matrix, batches, parameters) {
  parameters <- validate_canonical_parameters(parameters)
  nclust <- resolved_nclust(nrow(marker_matrix))
  if (nclust < 1L) stopf("Resolved nclust is below one.")
  options <- harmony::harmony_options(
    alpha = parameters$alpha,
    tau = parameters$tau,
    block.size = parameters$block_size,
    max.iter.cluster = parameters$max_iter_cluster,
    epsilon.cluster = parameters$epsilon_cluster,
    epsilon.harmony = parameters$epsilon_harmony
  )
  adjusted <- with_locked_rng(parameters, harmony::RunHarmony(
    data_mat = marker_matrix,
    meta_data = as.character(batches),
    theta = parameters$theta,
    sigma = parameters$sigma,
    lambda = parameters$lambda,
    nclust = nclust,
    max_iter = parameters$max_iter,
    early_stop = parameters$early_stop,
    ncores = parameters$ncores,
    plot_convergence = parameters$plot_convergence,
    return_object = parameters$return_object,
    verbose = FALSE,
    .options = options
  ))
  adjusted <- as.matrix(adjusted)
  storage.mode(adjusted) <- "double"
  if (!identical(dim(adjusted), dim(marker_matrix)) ||
      !identical(rownames(adjusted), rownames(marker_matrix)) ||
      !identical(colnames(adjusted), colnames(marker_matrix))) {
    stopf("Harmony output did not preserve cells, marker axes, names, and order.")
  }
  if (any(!is.finite(adjusted))) stopf("Harmony output contains non-finite values.")
  list(adjusted = adjusted, resolved_nclust = nclust)
}

midrank_percentile <- function(x) {
  (rank(x, ties.method = "average") - 0.5) / length(x)
}

safe_cor <- function(x, y, method) {
  suppressWarnings(as.numeric(stats::cor(x, y, method = method, use = "complete.obs")))
}

compute_marker_audits <- function(input_matrix, adjusted_matrix, batches, samples, batch_levels) {
  rank_rows <- vector("list", length(batch_levels) * ncol(input_matrix))
  shift_rows <- vector("list", length(rank_rows))
  cursor <- 1L
  for (batch_name in batch_levels) {
    rows <- as.character(batches) == batch_name
    for (marker in colnames(input_matrix)) {
      before <- input_matrix[rows, marker]
      after <- adjusted_matrix[rows, marker]
      before_rank <- midrank_percentile(before)
      after_rank <- midrank_percentile(after)
      delta_rank <- abs(after_rank - before_rank)
      delta_value <- after - before
      input_min <- min(before)
      input_max <- max(before)
      rank_rows[[cursor]] <- data.frame(
        batch = batch_name,
        marker = marker,
        n_cells = length(before),
        spearman = safe_cor(before, after, "spearman"),
        kendall_tau_b = safe_cor(before, after, "kendall"),
        rank_changed_fraction = mean(delta_rank > 0),
        median_abs_midrank_displacement = stats::median(delta_rank),
        max_abs_midrank_displacement = max(delta_rank),
        fraction_abs_midrank_displacement_gt_0_05 = mean(delta_rank > 0.05),
        fraction_abs_midrank_displacement_gt_0_10 = mean(delta_rank > 0.10),
        fraction_abs_midrank_displacement_gt_0_25 = mean(delta_rank > 0.25),
        stringsAsFactors = FALSE,
        check.names = FALSE
      )
      shift_rows[[cursor]] <- data.frame(
        batch = batch_name,
        marker = marker,
        n_cells = length(before),
        input_mean = mean(before),
        adjusted_mean = mean(after),
        input_median = stats::median(before),
        adjusted_median = stats::median(after),
        mean_shift = mean(delta_value),
        median_shift = stats::median(delta_value),
        median_abs_shift = stats::median(abs(delta_value)),
        max_abs_shift = max(abs(delta_value)),
        adjusted_negative_fraction = mean(after < 0),
        adjusted_outside_within_batch_input_range_fraction = mean(after < input_min | after > input_max),
        stringsAsFactors = FALSE,
        check.names = FALSE
      )
      cursor <- cursor + 1L
    }
  }
  sample_groups <- unique(data.frame(
    batch = as.character(batches),
    sample = as.character(samples),
    stringsAsFactors = FALSE
  ))
  sample_groups$batch_order <- match(sample_groups$batch, batch_levels)
  sample_groups <- sample_groups[order(sample_groups$batch_order, sample_groups$sample), , drop = FALSE]
  sample_rank_rows <- vector("list", nrow(sample_groups) * ncol(input_matrix))
  cursor <- 1L
  for (group_index in seq_len(nrow(sample_groups))) {
    batch_name <- sample_groups$batch[group_index]
    sample_name <- sample_groups$sample[group_index]
    rows <- as.character(batches) == batch_name & as.character(samples) == sample_name
    for (marker in colnames(input_matrix)) {
      before <- input_matrix[rows, marker]
      after <- adjusted_matrix[rows, marker]
      before_rank <- midrank_percentile(before)
      after_rank <- midrank_percentile(after)
      delta_rank <- abs(after_rank - before_rank)
      sample_rank_rows[[cursor]] <- data.frame(
        batch = batch_name,
        sample = sample_name,
        marker = marker,
        n_cells = length(before),
        spearman = safe_cor(before, after, "spearman"),
        kendall_tau_b = safe_cor(before, after, "kendall"),
        rank_changed_fraction = mean(delta_rank > 0),
        median_abs_midrank_displacement = stats::median(delta_rank),
        max_abs_midrank_displacement = max(delta_rank),
        fraction_abs_midrank_displacement_gt_0_05 = mean(delta_rank > 0.05),
        fraction_abs_midrank_displacement_gt_0_10 = mean(delta_rank > 0.10),
        fraction_abs_midrank_displacement_gt_0_25 = mean(delta_rank > 0.25),
        stringsAsFactors = FALSE,
        check.names = FALSE
      )
      cursor <- cursor + 1L
    }
  }

  list(
    rank = do.call(rbind, rank_rows),
    rank_by_sample = do.call(rbind, sample_rank_rows),
    shift = do.call(rbind, shift_rows)
  )
}

fit_harmony_direct <- function(
    input,
    markers,
    cell_id_column,
    batch_column,
    sample_column,
    batch_levels,
    parameters = canonical_harmony_direct_parameters(),
    required_library = NULL) {
  installation <- validate_harmony_install(required_library)
  parameters <- validate_canonical_parameters(parameters)
  prepared <- build_harmony_direct_input(
    input = input,
    markers = markers,
    cell_id_column = cell_id_column,
    batch_column = batch_column,
    sample_column = sample_column,
    batch_levels = batch_levels
  )

  captured_messages <- character()
  captured_warnings <- character()
  core <- withCallingHandlers(
    run_harmony_direct_core(prepared$marker_matrix, prepared$batches, parameters),
    message = function(condition) {
      captured_messages <<- c(captured_messages, conditionMessage(condition))
      invokeRestart("muffleMessage")
    },
    warning = function(condition) {
      captured_warnings <<- c(captured_warnings, conditionMessage(condition))
      invokeRestart("muffleWarning")
    }
  )
  audits <- compute_marker_audits(
    prepared$marker_matrix,
    core$adjusted,
    prepared$batches,
    prepared$samples,
    batch_levels
  )
  resolved_parameters <- c(parameters, list(resolved_nclust = core$resolved_nclust))

  list(
    method_label = HARMONY_DIRECT_METHOD_LABEL,
    input_marker_coordinates = prepared$marker_matrix,
    adjusted_marker_coordinates = core$adjusted,
    cell_metadata = prepared$metadata,
    marker_rank_audit = audits$rank,
    marker_rank_audit_by_sample = audits$rank_by_sample,
    marker_value_shift_audit = audits$shift,
    fitting_input_columns = prepared$fitting_input_columns,
    provenance_only_input_columns = prepared$provenance_only_input_columns,
    ignored_input_columns = prepared$ignored_input_columns,
    resolved_parameters = resolved_parameters,
    installation = installation,
    messages = captured_messages,
    warnings = captured_warnings
  )
}
