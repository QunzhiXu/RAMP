.cytofruv_source_file <- tryCatch(
  normalizePath(sys.frame(1)$ofile, mustWork = TRUE),
  error = function(e) NA_character_
)

cytofruv_algorithm_dir <- function() {
  override <- Sys.getenv("CYTOFRUV_ALGORITHM_DIR", unset = "")
  if (nzchar(override)) return(normalizePath(override, mustWork = TRUE))
  if (!is.na(.cytofruv_source_file)) return(dirname(.cytofruv_source_file))
  stop("Set CYTOFRUV_ALGORITHM_DIR before sourcing cytofruv_repro.R")
}

cytofruv_sha256 <- function(path) {
  if (!requireNamespace("digest", quietly = TRUE)) stop("digest is required")
  digest::digest(file = path, algo = "sha256", serialize = FALSE)
}

cytofruv_exact_object <- function(value, expected, field) {
  observed <- names(value)
  if (!is.list(value) || is.null(observed) ||
      length(observed) != length(expected) || anyDuplicated(observed) ||
      !setequal(observed, expected)) {
    stop(field, " fields differ from the locked contract or contain duplicate keys")
  }
  invisible(value)
}

cytofruv_scalar_character <- function(value, field) {
  if (!is.character(value) || length(value) != 1L || is.na(value) || !nzchar(value)) {
    stop(field, " must be one non-empty character string")
  }
  value
}

cytofruv_character_vector <- function(value, field, minimum = 1L) {
  if (is.character(value)) {
    answer <- value
  } else if (is.list(value) && all(vapply(value, function(item) {
    is.character(item) && length(item) == 1L && !is.na(item)
  }, logical(1)))) {
    answer <- vapply(value, identity, character(1))
  } else {
    stop(field, " must contain character strings only")
  }
  if (length(answer) < minimum || anyNA(answer) || any(!nzchar(answer))) {
    stop(field, " must contain at least ", minimum, " non-empty character string(s)")
  }
  answer
}

cytofruv_json_string_array <- function(value, field, minimum = 1L) {
  if (!is.list(value) || !is.null(names(value))) {
    stop(field, " must be a JSON array of character strings")
  }
  cytofruv_character_vector(value, field, minimum = minimum)
}

cytofruv_integer_scalar <- function(value, field, minimum = NULL) {
  if (!typeof(value) %in% c("integer", "double") || length(value) != 1L ||
      is.na(value) || !is.finite(value) || value %% 1 != 0 ||
      value < -.Machine$integer.max || value > .Machine$integer.max) {
    stop(field, " must be one explicit integer")
  }
  answer <- as.integer(value)
  if (!is.null(minimum) && answer < minimum) stop(field, " must be >= ", minimum)
  answer
}

cytofruv_sha256_string <- function(value, field) {
  value <- cytofruv_scalar_character(value, field)
  if (!grepl("^[0-9a-f]{64}$", value)) stop(field, " must be a lowercase SHA-256 string")
  value
}

cytofruv_verify_file <- function(path, expected_sha256, field) {
  path <- cytofruv_scalar_character(path, paste0(field, " path"))
  expected_sha256 <- cytofruv_sha256_string(expected_sha256, paste0(field, " sha256"))
  resolved <- normalizePath(path, mustWork = TRUE)
  info <- file.info(resolved)
  if (is.na(info$isdir) || isTRUE(info$isdir)) stop(field, " must be a regular file")
  observed <- cytofruv_sha256(resolved)
  if (!identical(observed, expected_sha256)) stop(field, " SHA-256 mismatch")
  resolved
}

cytofruv_cell_cluster_sha256 <- function(input) {
  payload <- paste0(
    "cell_id\tcluster\n",
    paste0(as.character(input$cell_id), "\t", as.character(input$cluster), collapse = "\n"),
    "\n"
  )
  digest::digest(payload, algo = "sha256", serialize = FALSE)
}

cytofruv_runtime_lock <- function() {
  path <- file.path(cytofruv_algorithm_dir(), "runtime_lock.csv")
  read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

cytofruv_validate_dependency_archives <- function() {
  algorithm_dir <- cytofruv_algorithm_dir()
  expected <- c(
    "ruv_0.9.7.1.tar.gz" = "a0c54e56ba3d8f6ae178ae4d0e417a79295abf5dcb68bbae26c4b874734d98d8",
    "rsvd_1.0.5.tar.gz" = "e40686b869acd4f71fdb1e8e7a6c64cd6792fc9d52a78f9e559a7176ab84e21e"
  )
  for (filename in names(expected)) {
    path <- file.path(algorithm_dir, "vendor", filename)
    if (!file.exists(path) || !identical(cytofruv_sha256(path), expected[[filename]])) {
      stop("Vendored dependency archive SHA-256 mismatch: ", filename)
    }
  }
  invisible(expected)
}

cytofruv_function_sha256 <- function(fun) {
  if (!is.function(fun)) stop("Runtime fingerprint target is not a function")
  canonical <- paste(deparse(fun, width.cutoff = 500L, control = "all"), collapse = "\n")
  digest::digest(canonical, algo = "sha256", serialize = FALSE)
}

cytofruv_validate_runtime_functions <- function() {
  lock_path <- file.path(cytofruv_algorithm_dir(), "runtime_function_lock.csv")
  lock <- read.csv(lock_path, stringsAsFactors = FALSE, check.names = FALSE)
  if (!identical(names(lock), c("package", "symbol", "sha256")) ||
      !nrow(lock) || anyNA(lock) || anyDuplicated(paste(lock$package, lock$symbol))) {
    stop("Invalid runtime_function_lock.csv")
  }
  for (i in seq_len(nrow(lock))) {
    namespace <- asNamespace(lock$package[[i]])
    symbol <- lock$symbol[[i]]
    if (!exists(symbol, envir = namespace, inherits = FALSE)) {
      stop("Missing locked runtime function: ", lock$package[[i]], ":::", symbol)
    }
    observed <- cytofruv_function_sha256(get(symbol, envir = namespace, inherits = FALSE))
    if (!identical(observed, lock$sha256[[i]])) {
      stop("Runtime function SHA-256 mismatch: ", lock$package[[i]], ":::", symbol)
    }
  }
  invisible(lock)
}

activate_cytofruv_runtime <- function() {
  algorithm_dir <- cytofruv_algorithm_dir()
  cytofruv_validate_dependency_archives()
  project_library <- normalizePath(file.path(algorithm_dir, ".Rlib"), mustWork = TRUE)
  .libPaths(c(project_library, .libPaths()))

  lock <- cytofruv_runtime_lock()
  for (i in seq_len(nrow(lock))) {
    package <- lock$package[[i]]
    expected <- lock$version[[i]]
    location <- lock$required_location[[i]]
    if (identical(package, "R")) {
      observed <- paste(R.version$major, R.version$minor, sep = ".")
    } else {
      if (!requireNamespace(package, quietly = TRUE)) stop("Missing locked R package: ", package)
      observed <- as.character(utils::packageVersion(package))
      if (identical(location, "project_library")) {
        package_path <- normalizePath(getNamespaceInfo(asNamespace(package), "path"), mustWork = TRUE)
        prefix <- paste0(project_library, .Platform$file.sep)
        if (!startsWith(paste0(package_path, .Platform$file.sep), prefix)) {
          stop(package, " must resolve from algorithm/cytofruv/.Rlib")
        }
      }
    }
    if (!identical(observed, expected)) {
      stop("Runtime version mismatch for ", package, ": expected ", expected, ", observed ", observed)
    }
  }
  cytofruv_validate_runtime_functions()
  invisible(lock)
}

validate_cytofruv_source <- function() {
  algorithm_dir <- cytofruv_algorithm_dir()
  archive <- file.path(
    algorithm_dir,
    "vendor",
    "CytofRUV-7088af8794ede0fc7e3c6ef545638e279e8101eb.tar.gz"
  )
  expected_archive <- "a614b250a7f14488679a8822e1b1b6bb72f5fe292a74289b4f63b3f8b4d9f6b6"
  if (!file.exists(archive)) stop("Missing locked CytofRUV source archive")
  if (!identical(cytofruv_sha256(archive), expected_archive)) {
    stop("CytofRUV source archive SHA-256 mismatch")
  }
  invisible(archive)
}

load_official_cytofruv_core <- function() {
  activate_cytofruv_runtime()
  archive <- validate_cytofruv_source()
  extract_dir <- tempfile("cytofruv-source-")
  dir.create(extract_dir, recursive = TRUE)
  on.exit(unlink(extract_dir, recursive = TRUE, force = TRUE), add = TRUE)
  utils::untar(archive, exdir = extract_dir)
  core_paths <- list.files(
    extract_dir,
    pattern = "normalise_data[.]R$",
    recursive = TRUE,
    full.names = TRUE
  )
  if (length(core_paths) != 1L) stop("Locked archive must contain exactly one R/normalise_data.R")
  expected_core <- "cb0c07105985d38b86c198816172ea3144b749bf2cbe50e9681ade4e8ae692ab"
  if (!identical(cytofruv_sha256(core_paths[[1]]), expected_core)) {
    stop("Locked CytofRUV core source SHA-256 mismatch")
  }
  namespace_paths <- list.files(
    extract_dir,
    pattern = "^NAMESPACE$",
    recursive = TRUE,
    full.names = TRUE
  )
  if (length(namespace_paths) != 1L) stop("Locked archive must contain exactly one NAMESPACE")
  expected_namespace <- "19e1283ba9ee3dd44e1eacbc4ef41fb14a8e034dbe5ed44e06c818c4d733fd4f"
  if (!identical(cytofruv_sha256(namespace_paths[[1]]), expected_namespace)) {
    stop("Locked CytofRUV NAMESPACE SHA-256 mismatch")
  }
  namespace_lines <- readLines(namespace_paths[[1]], warn = FALSE)
  required_imports <- c("importFrom(rsvd,rsvd)", "importFrom(stats,sd)")
  if (!all(required_imports %in% namespace_lines)) stop("Locked CytofRUV namespace imports changed")

  # Reconstruct the package namespace needed by the locked core.  Using
  # baseenv() rather than globalenv() prevents .Rprofile or user objects from
  # changing unqualified symbol resolution.  These are exactly the two
  # unqualified imports declared by the locked NAMESPACE.
  upstream <- new.env(parent = baseenv())
  upstream$rsvd <- rsvd::rsvd
  upstream$sd <- stats::sd
  sys.source(core_paths[[1]], envir = upstream, keep.source = TRUE)
  required <- c("run_RUVIII", "make_residual_mat_several_rep", "fastRUVIII")
  if (!all(vapply(required, exists, logical(1), envir = upstream, inherits = FALSE))) {
    stop("Locked CytofRUV core functions are incomplete")
  }
  upstream
}

cytofruv_required_config_fields <- function() {
  c(
    "schema_version", "method_mode", "seed", "k", "marker_columns",
    "input_scale", "clustering", "replicate_design", "replicate_sets",
    "minimum_cells_per_reference_sample_cluster"
  )
}

validate_cytofruv_config <- function(config) {
  expected <- cytofruv_required_config_fields()
  cytofruv_exact_object(config, expected, "CytofRUV config")
  if (!identical(cytofruv_scalar_character(config$schema_version, "schema_version"),
                 "final_xu.cytofruv.config.v1")) {
    stop("Unsupported CytofRUV config schema")
  }
  if (!identical(cytofruv_scalar_character(config$method_mode, "method_mode"),
                 "official_replicated_reference")) {
    stop("Only official_replicated_reference mode is reportable as CytofRUV")
  }
  cytofruv_integer_scalar(config$seed, "seed")
  cytofruv_integer_scalar(config$k, "k", minimum = 1L)
  markers <- cytofruv_json_string_array(config$marker_columns, "marker_columns", minimum = 2L)
  reserved <- c("cell_id", "sample_id", "batch", "cluster")
  if (any(markers %in% reserved)) {
    stop("marker_columns cannot reuse reserved metadata names")
  }
  if (anyDuplicated(markers)) {
    stop("marker_columns must contain at least two unique non-empty marker names")
  }

  scale <- config$input_scale
  cytofruv_exact_object(scale, c(
    "name", "cofactor", "already_transformed", "source_document", "source_document_sha256"
  ), "input_scale")
  if (!identical(cytofruv_scalar_character(scale$name, "input_scale name"),
                 "arcsinh_intensity_over_cofactor") ||
      !isTRUE(scale$already_transformed) ||
      !typeof(scale$cofactor) %in% c("integer", "double") || length(scale$cofactor) != 1L ||
      !isTRUE(all.equal(as.numeric(scale$cofactor), 5))) {
    stop("Official mode requires values already transformed as arcsinh(intensity/5)")
  }
  cytofruv_scalar_character(scale$source_document, "input_scale source_document")
  cytofruv_sha256_string(scale$source_document_sha256, "input_scale source_document_sha256")

  clustering <- config$clustering
  clustering_fields <- c(
    "method", "type_markers", "number_of_clusters", "seed",
    "provenance_document", "provenance_document_sha256"
  )
  cytofruv_exact_object(clustering, clustering_fields, "clustering")
  if (!identical(cytofruv_scalar_character(clustering$method, "clustering method"),
                 "CytofRUV::cluster_data")) {
    stop("Production cluster assignments must come from CytofRUV::cluster_data")
  }
  type_markers <- cytofruv_json_string_array(clustering$type_markers, "type_markers")
  if (anyDuplicated(type_markers) || any(!type_markers %in% markers)) {
    stop("type_markers must be a non-empty subset of marker_columns")
  }
  cytofruv_integer_scalar(clustering$number_of_clusters, "number_of_clusters", minimum = 2L)
  cytofruv_integer_scalar(clustering$seed, "clustering seed")
  cytofruv_scalar_character(clustering$provenance_document, "clustering provenance_document")
  cytofruv_sha256_string(
    clustering$provenance_document_sha256,
    "clustering provenance_document_sha256"
  )

  design <- config$replicate_design
  design_fields <- c(
    "attestation", "target_batches", "source_document", "source_document_sha256",
    "replicate_map", "replicate_map_sha256"
  )
  cytofruv_exact_object(design, design_fields, "replicate_design")
  expected_attestation <- "same_biological_reference_sample_remeasured_across_all_target_batches"
  if (!identical(cytofruv_scalar_character(design$attestation, "replicate_design attestation"),
                 expected_attestation)) {
    stop("CytofRUV requires a real biological reference remeasured across target batches")
  }
  target_batches <- cytofruv_json_string_array(design$target_batches, "target_batches", minimum = 2L)
  if (anyDuplicated(target_batches)) {
    stop("replicate_design must declare at least two unique target batches")
  }
  cytofruv_scalar_character(design$source_document, "replicate_design source_document")
  cytofruv_sha256_string(design$source_document_sha256, "replicate_design source_document_sha256")
  cytofruv_scalar_character(design$replicate_map, "replicate_design replicate_map")
  cytofruv_sha256_string(design$replicate_map_sha256, "replicate_design replicate_map_sha256")

  replicate_sets <- config$replicate_sets
  if (!is.list(replicate_sets) || !is.null(names(replicate_sets)) || !length(replicate_sets)) {
    stop("At least one real replicate set is required as a JSON array")
  }
  reference_ids <- character()
  for (set in replicate_sets) {
    cytofruv_exact_object(set, c("reference_id", "sample_ids", "clusters"), "replicate set")
    reference_id <- cytofruv_scalar_character(set$reference_id, "reference_id")
    reference_ids <- c(reference_ids, reference_id)
    samples <- cytofruv_json_string_array(set$sample_ids, "sample_ids", minimum = 2L)
    clusters <- cytofruv_json_string_array(set$clusters, "clusters")
    if (anyDuplicated(samples)) {
      stop("Each reference requires at least two unique acquisition-level sample_ids")
    }
    if (anyDuplicated(clusters)) {
      stop("Each reference requires at least one unique shared cluster")
    }
  }
  if (anyDuplicated(reference_ids)) stop("reference_id values must be unique")
  cytofruv_integer_scalar(
    config$minimum_cells_per_reference_sample_cluster,
    "minimum_cells_per_reference_sample_cluster",
    minimum = 1L
  )
  invisible(config)
}

validate_cytofruv_provenance <- function(input, config) {
  scale_path <- cytofruv_verify_file(
    config$input_scale$source_document,
    config$input_scale$source_document_sha256,
    "input-scale source document"
  )
  clustering_path <- cytofruv_verify_file(
    config$clustering$provenance_document,
    config$clustering$provenance_document_sha256,
    "clustering provenance document"
  )
  design_path <- cytofruv_verify_file(
    config$replicate_design$source_document,
    config$replicate_design$source_document_sha256,
    "replicate-design source document"
  )
  replicate_map_path <- cytofruv_verify_file(
    config$replicate_design$replicate_map,
    config$replicate_design$replicate_map_sha256,
    "replicate map"
  )

  if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite is required")
  cluster_provenance <- jsonlite::fromJSON(clustering_path, simplifyVector = FALSE)
  expected_cluster_fields <- c(
    "schema_version", "method", "upstream_commit", "cluster_source_sha256",
    "cell_cluster_sha256", "type_markers", "number_of_clusters", "seed"
  )
  cytofruv_exact_object(
    cluster_provenance,
    expected_cluster_fields,
    "clustering provenance"
  )
  if (!identical(cluster_provenance$schema_version, "final_xu.cytofruv.clustering_provenance.v1") ||
      !identical(cluster_provenance$method, "CytofRUV::cluster_data") ||
      !identical(cluster_provenance$upstream_commit,
                 "7088af8794ede0fc7e3c6ef545638e279e8101eb") ||
      !identical(cluster_provenance$cluster_source_sha256,
                 "2154ae39693cc3036e59c824b14a6ddc7d76f0c776d19e997b7743eadbb6f937")) {
    stop("clustering provenance does not identify the locked official cluster_data source")
  }
  if (!identical(
    cytofruv_json_string_array(cluster_provenance$type_markers, "provenance type_markers"),
    cytofruv_json_string_array(config$clustering$type_markers, "type_markers")
  ) ||
      cytofruv_integer_scalar(cluster_provenance$number_of_clusters, "provenance number_of_clusters") !=
        cytofruv_integer_scalar(config$clustering$number_of_clusters, "number_of_clusters") ||
      cytofruv_integer_scalar(cluster_provenance$seed, "provenance clustering seed") !=
        cytofruv_integer_scalar(config$clustering$seed, "clustering seed")) {
    stop("clustering provenance parameters differ from config")
  }
  if (!identical(
    cytofruv_sha256_string(cluster_provenance$cell_cluster_sha256, "cell_cluster_sha256"),
    cytofruv_cell_cluster_sha256(input)
  )) {
    stop("cluster assignments do not match the hash-bound official clustering provenance")
  }

  replicate_map <- read.csv(
    replicate_map_path,
    stringsAsFactors = FALSE,
    colClasses = "character",
    check.names = FALSE
  )
  expected_map_columns <- c("reference_id", "sample_id", "batch")
  if (anyDuplicated(names(replicate_map)) || !identical(names(replicate_map), expected_map_columns)) {
    stop("replicate map columns must be exactly reference_id,sample_id,batch")
  }
  if (!nrow(replicate_map) || anyNA(replicate_map) ||
      any(!nzchar(as.matrix(replicate_map))) || anyDuplicated(replicate_map$sample_id)) {
    stop("replicate map must contain unique, complete acquisition-level sample IDs")
  }
  list(
    scale_path = scale_path,
    clustering_path = clustering_path,
    design_path = design_path,
    replicate_map_path = replicate_map_path,
    replicate_map = replicate_map
  )
}

validate_cytofruv_design <- function(input, config) {
  validate_cytofruv_config(config)
  markers <- cytofruv_json_string_array(config$marker_columns, "marker_columns", minimum = 2L)
  metadata <- c("cell_id", "sample_id", "batch", "cluster")
  expected_columns <- c(metadata, markers)
  if (anyDuplicated(names(input))) stop("Input column names must be globally unique")
  if (!identical(names(input), expected_columns)) {
    stop("Input columns must be exactly cell_id,sample_id,batch,cluster followed by marker_columns")
  }
  if (!nrow(input)) stop("Input contains no cells")
  for (column in metadata) {
    values <- as.character(input[[column]])
    if (anyNA(values) || any(!nzchar(values))) stop("Missing values in metadata column ", column)
  }
  if (anyDuplicated(as.character(input$cell_id))) stop("cell_id must be unique")
  for (marker in markers) {
    values <- input[[marker]]
    if (!is.numeric(values) || any(!is.finite(values))) stop("Non-finite/non-numeric marker: ", marker)
    if (stats::sd(values) == 0) stop("CytofRUV cannot standardize constant marker: ", marker)
  }
  observed_clusters <- unique(as.character(input$cluster))
  if (length(observed_clusters) != as.integer(config$clustering$number_of_clusters)) {
    stop("Observed cluster count differs from clustering.number_of_clusters")
  }
  if (config$k > min(nrow(input), length(markers)) - 1L) {
    stop("k exceeds the maximum rank available to the locked rsvd call")
  }

  provenance <- validate_cytofruv_provenance(input, config)
  sample_batch <- unique(data.frame(
    sample_id = as.character(input$sample_id),
    batch = as.character(input$batch),
    stringsAsFactors = FALSE
  ))
  if (anyDuplicated(as.character(sample_batch$sample_id))) {
    stop("Each acquisition-level sample_id must belong to exactly one batch")
  }
  observed_batches <- unique(as.character(input$batch))
  target_batches <- cytofruv_json_string_array(config$replicate_design$target_batches, "target_batches", minimum = 2L)
  if (!setequal(observed_batches, target_batches)) {
    stop("Declared target_batches must exactly match input batches")
  }

  used_samples <- character()
  used_references <- character()
  minimum <- as.integer(config$minimum_cells_per_reference_sample_cluster)
  replicate_map <- provenance$replicate_map
  if (!all(replicate_map$sample_id %in% sample_batch$sample_id)) {
    stop("replicate map contains sample IDs absent from the input")
  }
  mapped_batches <- sample_batch$batch[match(replicate_map$sample_id, sample_batch$sample_id)]
  if (!identical(as.character(mapped_batches), as.character(replicate_map$batch))) {
    stop("replicate map batch assignments differ from the input")
  }
  audit_rows <- list()
  index <- 1L
  for (set in config$replicate_sets) {
    reference_id <- cytofruv_scalar_character(set$reference_id, "reference_id")
    samples <- cytofruv_json_string_array(set$sample_ids, "sample_ids", minimum = 2L)
    clusters <- cytofruv_json_string_array(set$clusters, "clusters")
    mapped_samples <- replicate_map$sample_id[replicate_map$reference_id == reference_id]
    if (!setequal(samples, mapped_samples)) {
      stop("replicate_sets sample IDs disagree with the hash-bound replicate map")
    }
    used_references <- c(used_references, reference_id)
    if (any(samples %in% used_samples)) stop("A sample_id cannot belong to multiple reference sets")
    used_samples <- c(used_samples, samples)
    if (!all(samples %in% sample_batch$sample_id)) stop("Unknown reference sample_id in replicate_sets")
    batches <- as.character(sample_batch$batch[match(samples, sample_batch$sample_id)])
    if (!setequal(batches, target_batches)) {
      stop("Every reference set must cover every declared target batch")
    }
    for (sample in samples) {
      for (cluster in clusters) {
        count <- sum(as.character(input$sample_id) == sample & as.character(input$cluster) == cluster)
        if (count < minimum) {
          stop("Reference sample/cluster has too few cells: ", sample, "/", cluster)
        }
        audit_rows[[index]] <- data.frame(
          reference_id = reference_id,
          sample_id = sample,
          batch = as.character(sample_batch$batch[match(sample, sample_batch$sample_id)]),
          cluster = cluster,
          cells = count,
          stringsAsFactors = FALSE
        )
        index <- index + 1L
      }
    }
  }
  if (!setequal(used_samples, replicate_map$sample_id) ||
      !setequal(used_references, unique(replicate_map$reference_id))) {
    stop("replicate_sets must exactly cover the hash-bound replicate map")
  }
  raw_marker_data <- as.matrix(input[, markers, drop = FALSE])
  standardized <- scale(raw_marker_data, center = TRUE, scale = TRUE)
  residual <- standardized
  residual[] <- 0
  for (set in config$replicate_sets) {
    samples <- cytofruv_json_string_array(set$sample_ids, "sample_ids", minimum = 2L)
    clusters <- cytofruv_json_string_array(set$clusters, "clusters")
    for (cluster in clusters) {
      selected <- as.character(input$cluster) == cluster &
        as.character(input$sample_id) %in% samples
      center <- colMeans(standardized[selected, , drop = FALSE])
      residual[selected, ] <- sweep(standardized[selected, , drop = FALSE], 2L, center, "-")
    }
  }
  residual_rank <- qr(residual, LAPACK = FALSE)$rank
  if (residual_rank < as.integer(config$k)) {
    stop("k exceeds the numerical rank of the reference residual matrix")
  }
  do.call(rbind, audit_rows)
}

run_cytofruv_official_core <- function(input, config) {
  design_audit <- validate_cytofruv_design(input, config)
  markers <- cytofruv_json_string_array(config$marker_columns, "marker_columns", minimum = 2L)
  raw_data <- data.frame(
    sample = as.character(input$sample_id),
    cluster = as.character(input$cluster),
    input[, markers, drop = FALSE],
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  rep_samples <- lapply(config$replicate_sets, function(set) {
    cytofruv_json_string_array(set$sample_ids, "sample_ids", minimum = 2L)
  })
  norm_clusters <- lapply(config$replicate_sets, function(set) {
    cytofruv_json_string_array(set$clusters, "clusters")
  })
  upstream <- load_official_cytofruv_core()
  set.seed(as.integer(config$seed), kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
  corrected <- upstream$run_RUVIII(
    data = raw_data,
    norm_clusters = norm_clusters,
    k = as.integer(config$k),
    rep_samples = rep_samples
  )
  corrected <- as.matrix(corrected)
  colnames(corrected) <- markers
  if (!identical(dim(corrected), c(nrow(input), length(markers)))) {
    stop("Official CytofRUV core returned an unexpected matrix shape")
  }
  if (any(!is.finite(corrected))) stop("Official CytofRUV core returned non-finite values")
  list(corrected = corrected, design_audit = design_audit)
}
