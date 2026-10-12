CYCOMBINE_REPRO_SCHEMA <- "final_xu.cycombine_repro.v1"
CYCOMBINE_EXPECTED_VERSION <- "0.2.5"
CYCOMBINE_PUBLICATION_COMMIT <- "5b5903c1accc887b4a11c2b25c550a0b0e36876d"
CYCOMBINE_SOURCE_SHA256 <- "8ac3a3827b3f68ca4c1e036b22a2adff594b00fb25d09483648c5191da87eaee"
CYCOMBINE_CROSS_PLATFORM_VIGNETTE_SHA256 <- "f90ef5082f92827a19aa745f0121d6f7f4f2f36d8a6a7fdfc35e5673b624390a"
CYCOMBINE_FUNCTION_BODY_SHA256 <- c(
  normalize = "0cd7696d17cea6dac7c792d3a7792942042539b95b51db71727b68a69587ae09",
  create_som = "32bc7f1661577a8e08bd121a6234db50ca265bd044a1d41aed68657ef7d8a22e",
  correct_data = "407d2aaacefa88587c2463a2e56c889fd12931c9d8a65e6086c2cf561b53820c",
  batch_correct = "fd6ea1ea423c4aca51346f3c2e5b23c8e35c540a5cecbe3c55218cd7acd131f0"
)

stopf <- function(fmt, ...) {
  stop(sprintf(fmt, ...), call. = FALSE)
}

canonical_cycombine_parameters <- function() {
  list(
    xdim = 8L,
    ydim = 8L,
    rlen = 10L,
    seed = 840L,
    norm_method = "rank",
    ties_method = "average",
    parametric = TRUE,
    covariate = NULL,
    anchor = NULL
  )
}

validate_canonical_parameters <- function(parameters) {
  expected <- canonical_cycombine_parameters()
  if (!is.list(parameters)) {
    stopf("algorithm_parameters must be a JSON object / R list.")
  }
  if (!setequal(names(parameters), names(expected))) {
    missing <- setdiff(names(expected), names(parameters))
    extra <- setdiff(names(parameters), names(expected))
    stopf(
      "algorithm_parameters fields differ from the locked schema. Missing: [%s]. Extra: [%s].",
      paste(missing, collapse = ", "), paste(extra, collapse = ", ")
    )
  }

  integer_fields <- c("xdim", "ydim", "rlen", "seed")
  for (field in integer_fields) {
    value <- parameters[[field]]
    if (length(value) != 1L || is.na(value) || value != as.integer(value)) {
      stopf("algorithm_parameters.%s must be one integer.", field)
    }
    parameters[[field]] <- as.integer(value)
  }
  if (!is.character(parameters$norm_method) || length(parameters$norm_method) != 1L ||
      !is.character(parameters$ties_method) || length(parameters$ties_method) != 1L ||
      !is.logical(parameters$parametric) || length(parameters$parametric) != 1L ||
      is.na(parameters$parametric)) {
    stopf("norm_method, ties_method, and parametric have invalid types.")
  }
  if (!is.null(parameters$covariate) || !is.null(parameters$anchor)) {
    stopf("The official cross-platform vignette profile requires covariate=null and anchor=null.")
  }

  for (field in names(expected)) {
    if (!identical(parameters[[field]], expected[[field]])) {
      stopf(
        "algorithm_parameters.%s=%s differs from the locked official-vignette profile (%s). Put any predeclared sensitivity analysis outside this canonical runner.",
        field,
        paste(parameters[[field]], collapse = ","),
        paste(expected[[field]], collapse = ",")
      )
    }
  }
  expected
}

validate_cycombine_install <- function(required_library = NULL) {
  if (!requireNamespace("cyCombine", quietly = TRUE)) {
    stopf("cyCombine is unavailable. Run bootstrap_library.R first.")
  }
  installed_version <- as.character(utils::packageVersion("cyCombine"))
  if (!identical(installed_version, CYCOMBINE_EXPECTED_VERSION)) {
    stopf(
      "Expected cyCombine %s (publication release), but loaded %s from %s.",
      CYCOMBINE_EXPECTED_VERSION,
      installed_version,
      find.package("cyCombine")
    )
  }

  expected_formals <- c(
    "df", "label", "xdim", "ydim", "rlen", "parametric", "seed",
    "covar", "anchor", "markers", "norm_method", "ties.method"
  )
  actual_formals <- names(formals(cyCombine::batch_correct))
  if (!identical(actual_formals, expected_formals)) {
    stopf(
      "cyCombine::batch_correct API does not match v0.2.5. Found: %s",
      paste(actual_formals, collapse = ", ")
    )
  }

  if (!is.null(required_library)) {
    expected_path <- normalizePath(file.path(required_library, "cyCombine"), mustWork = TRUE)
    actual_path <- normalizePath(find.package("cyCombine"), mustWork = TRUE)
    if (!identical(actual_path, expected_path)) {
      stopf("cyCombine loaded from %s instead of the dedicated library %s.", actual_path, expected_path)
    }
    stamp <- file.path(required_library, ".cycombine_source_sha256")
    if (!file.exists(stamp)) {
      stopf("Missing source verification stamp: %s", stamp)
    }
    recorded <- trimws(readLines(stamp, warn = FALSE)[1L])
    if (!identical(recorded, CYCOMBINE_SOURCE_SHA256)) {
      stopf("Dedicated library source stamp does not match the locked archive SHA-256.")
    }

    source_archive <- file.path(dirname(required_library), "vendor", "cyCombine-v0.2.5.tar.gz")
    if (!file.exists(source_archive)) {
      stopf("Missing vendored cyCombine source archive: %s", source_archive)
    }
    actual_archive_sha256 <- digest::digest(
      file = source_archive,
      algo = "sha256",
      serialize = FALSE
    )
    if (!identical(actual_archive_sha256, CYCOMBINE_SOURCE_SHA256)) {
      stopf("Vendored cyCombine source archive does not match the locked SHA-256.")
    }
  }

  actual_function_hashes <- vapply(
    names(CYCOMBINE_FUNCTION_BODY_SHA256),
    function(function_name) {
      function_object <- getExportedValue("cyCombine", function_name)
      digest::digest(
        paste(deparse(body(function_object), width.cutoff = 500L), collapse = "\n"),
        algo = "sha256",
        serialize = FALSE
      )
    },
    character(1)
  )
  if (!identical(actual_function_hashes, CYCOMBINE_FUNCTION_BODY_SHA256)) {
    bad <- names(actual_function_hashes)[actual_function_hashes != CYCOMBINE_FUNCTION_BODY_SHA256]
    stopf(
      "Installed cyCombine function body differs from the verified v0.2.5 source: %s",
      paste(bad, collapse = ", ")
    )
  }

  invisible(list(
    package_version = installed_version,
    installed_path = normalizePath(find.package("cyCombine"), mustWork = TRUE),
    source_archive_sha256 = CYCOMBINE_SOURCE_SHA256,
    function_body_sha256 = as.list(actual_function_hashes)
  ))
}

read_marker_file <- function(path) {
  markers <- trimws(readLines(path, warn = FALSE))
  markers <- markers[nzchar(markers)]
  if (length(markers) < 2L) {
    stopf("The marker file must contain at least two shared markers.")
  }
  if (anyDuplicated(markers)) {
    stopf("The marker file contains duplicate marker names: %s", paste(unique(markers[duplicated(markers)]), collapse = ", "))
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
  if (!is.data.frame(object)) {
    stopf("Input must resolve to a data.frame/tibble.")
  }
  if (anyDuplicated(colnames(object))) {
    stopf("Input contains duplicate column names: %s", paste(unique(colnames(object)[duplicated(colnames(object))]), collapse = ", "))
  }
  as.data.frame(object, check.names = FALSE, stringsAsFactors = FALSE)
}

build_cycombine_input <- function(
    input,
    markers,
    cell_id_column,
    batch_column,
    sample_column,
    batch_levels) {
  required <- c(cell_id_column, batch_column, sample_column, markers)
  missing <- setdiff(required, colnames(input))
  if (length(missing)) {
    stopf("Input is missing required columns: %s", paste(missing, collapse = ", "))
  }
  if (anyDuplicated(required)) {
    stopf("cell_id, batch, sample, and marker column names must be distinct.")
  }
  if (anyNA(input[[cell_id_column]]) || any(!nzchar(as.character(input[[cell_id_column]])))) {
    stopf("cell IDs contain missing or empty values.")
  }
  if (anyDuplicated(as.character(input[[cell_id_column]]))) {
    stopf("cell IDs must be globally unique.")
  }
  if (anyNA(input[[batch_column]]) || anyNA(input[[sample_column]])) {
    stopf("batch or sample contains missing values.")
  }
  if (!is.character(batch_levels) || length(batch_levels) != 3L || anyDuplicated(batch_levels)) {
    stopf("The locked cross-platform profile requires exactly three ordered, unique dataset batches.")
  }
  observed_batches <- unique(as.character(input[[batch_column]]))
  if (!setequal(observed_batches, batch_levels)) {
    stopf(
      "Observed batches [%s] do not exactly match configured batch_levels [%s].",
      paste(observed_batches, collapse = ", "),
      paste(batch_levels, collapse = ", ")
    )
  }

  expression <- input[, markers, drop = FALSE]
  non_numeric <- markers[!vapply(expression, is.numeric, logical(1))]
  if (length(non_numeric)) {
    stopf("Marker columns must be numeric: %s", paste(non_numeric, collapse = ", "))
  }
  finite <- vapply(expression, function(x) all(is.finite(x)), logical(1))
  if (any(!finite)) {
    stopf("Marker columns contain NA/NaN/Inf: %s", paste(markers[!finite], collapse = ", "))
  }

  cy_input <- data.frame(id = seq_len(nrow(input)), check.names = FALSE)
  cy_input <- cbind(cy_input, expression)
  cy_input$batch <- factor(as.character(input[[batch_column]]), levels = batch_levels)
  cy_input$sample <- as.character(input[[sample_column]])
  cy_input
}

marker_batch_statistics <- function(cy_input, markers) {
  batches <- levels(cy_input$batch)
  rows <- vector("list", length(batches) * length(markers))
  cursor <- 1L
  for (batch in batches) {
    index <- cy_input$batch == batch
    for (marker in markers) {
      values <- cy_input[[marker]][index]
      rows[[cursor]] <- data.frame(
        batch = batch,
        marker = marker,
        n = length(values),
        minimum = min(values),
        maximum = max(values),
        mean = mean(values),
        sd = stats::sd(values),
        sum = sum(values),
        all_zero = all(values == 0),
        zero_variance = length(values) < 2L || stats::var(values) == 0,
        stringsAsFactors = FALSE
      )
      cursor <- cursor + 1L
    }
  }
  do.call(rbind, rows)
}

fit_cycombine_publication <- function(
    input,
    markers,
    cell_id_column,
    batch_column,
    sample_column,
    batch_levels,
    parameters = canonical_cycombine_parameters(),
    required_library = NULL) {
  installation_provenance <- validate_cycombine_install(required_library)
  parameters <- validate_canonical_parameters(parameters)
  cy_input <- build_cycombine_input(
    input = input,
    markers = markers,
    cell_id_column = cell_id_column,
    batch_column = batch_column,
    sample_column = sample_column,
    batch_levels = batch_levels
  )

  stats <- marker_batch_statistics(cy_input, markers)
  exact_zero_sum <- stats$sum == 0
  if (any(exact_zero_sum)) {
    details <- paste0(stats$batch[exact_zero_sum], ":", stats$marker[exact_zero_sum])
    stopf(
      paste0(
        "At least one batch-marker has an exact zero sum: %s. Official cyCombine v0.2.5 ",
        "rejects this condition inside normalize(); the wrapper reports it early without ",
        "silently deleting a marker. Note that exact zero sum is an upstream implementation ",
        "constraint, not a general test of biological or statistical degeneracy."
      ),
      paste(details, collapse = ", ")
    )
  }

  previous_rng_kind <- RNGkind()
  on.exit(do.call(RNGkind, as.list(previous_rng_kind)), add = TRUE)
  RNGkind(kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")

  normalized <- cyCombine::normalize(
    df = cy_input,
    markers = markers,
    norm_method = parameters$norm_method,
    ties.method = parameters$ties_method
  )
  som_labels <- cyCombine::create_som(
    df = normalized,
    markers = markers,
    seed = parameters$seed,
    rlen = parameters$rlen,
    xdim = parameters$xdim,
    ydim = parameters$ydim
  )

  corrected <- cyCombine::batch_correct(
    df = cy_input,
    label = som_labels,
    xdim = parameters$xdim,
    ydim = parameters$ydim,
    rlen = parameters$rlen,
    parametric = parameters$parametric,
    seed = parameters$seed,
    covar = parameters$covariate,
    anchor = parameters$anchor,
    markers = markers,
    norm_method = parameters$norm_method,
    ties.method = parameters$ties_method
  )

  expected_id <- seq_len(nrow(input))
  if (!identical(as.integer(corrected$id), expected_id)) {
    stopf("cyCombine output row IDs no longer match the input order.")
  }
  if (!all(markers %in% colnames(corrected))) {
    stopf("cyCombine output lost one or more marker columns.")
  }
  corrected_expression <- as.data.frame(corrected[, markers, drop = FALSE], check.names = FALSE)
  if (!all(vapply(corrected_expression, function(x) all(is.finite(x)), logical(1)))) {
    stopf("cyCombine returned non-finite corrected expression values.")
  }

  node_count <- parameters$xdim * parameters$ydim
  node_batch_counts <- as.data.frame(
    table(
      som_node = factor(som_labels, levels = seq_len(node_count)),
      batch = factor(cy_input$batch, levels = batch_levels)
    ),
    stringsAsFactors = FALSE
  )
  node_batch_counts$som_node <- as.integer(as.character(node_batch_counts$som_node))
  node_batch_counts$n <- as.integer(node_batch_counts$Freq)
  node_batch_counts$Freq <- NULL

  node_correction_modes <- do.call(
    rbind,
    lapply(seq_len(node_count), function(node) {
      counts <- node_batch_counts$n[node_batch_counts$som_node == node]
      positive_counts <- counts[counts > 0L]
      present_batches <- length(positive_counts)
      correction_mode <- if (present_batches == 0L) {
        "empty"
      } else if (present_batches == 1L) {
        "single_batch_passthrough"
      } else if (any(positive_counts == 1L)) {
        "multi_batch_mean_only_combat"
      } else {
        "multi_batch_full_combat"
      }
      data.frame(
        som_node = node,
        present_batches = present_batches,
        minimum_cells_in_present_batch = if (present_batches) min(positive_counts) else 0L,
        correction_mode = correction_mode,
        stringsAsFactors = FALSE
      )
    })
  )

  list(
    corrected_expression = corrected_expression,
    cell_ids = as.character(input[[cell_id_column]]),
    batches = as.character(input[[batch_column]]),
    samples = as.character(input[[sample_column]]),
    som_labels = as.integer(som_labels),
    node_batch_counts = node_batch_counts,
    node_correction_modes = node_correction_modes,
    marker_batch_statistics = stats,
    fitting_input_columns = c(batch_column, markers),
    provenance_only_input_columns = c(cell_id_column, sample_column),
    ignored_input_columns = setdiff(colnames(input), c(cell_id_column, batch_column, sample_column, markers)),
    parameters = parameters,
    rng = list(kind = "Mersenne-Twister", normal_kind = "Inversion", sample_kind = "Rejection"),
    installation_provenance = installation_provenance
  )
}
