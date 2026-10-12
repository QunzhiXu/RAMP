# Wrapper around public, unmodified CytoNorm / FlowSOM functions.
read_fcs <- function(path) flowCore::read.FCS(path, transformation = FALSE,
                                             truncate_max_range = FALSE, min.limit = -Inf)
write_json <- function(x, path) jsonlite::write_json(x, path, auto_unbox = TRUE,
                                                    pretty = TRUE, null = "null", digits = 17)
sha <- function(path) digest::digest(file = path, algo = "sha256")
write_table <- function(x, path) data.table::fwrite(x, path, quote = TRUE, na = "NA")

validate_runtime <- function(here) {
  for (p in c("CytoNorm", "FlowSOM", "flowCore", "jsonlite", "digest", "data.table")) {
    if (!requireNamespace(p, quietly = TRUE)) stop("Missing package: ", p, "; run bootstrap_library.R")
  }
  lock <- jsonlite::read_json(file.path(here, "upstream_lock.json"))
  stopifnot(as.character(packageVersion("CytoNorm")) == lock$CytoNorm_version,
            as.character(packageVersion("FlowSOM")) == lock$FlowSOM_version)
  for (entry in lock$archives) stopifnot(sha(file.path(here, "vendor", entry$file)) == entry$sha256)
  runtime <- read.csv(file.path(here, ".Rlib", "runtime_lock.csv"))
  for (i in seq_len(nrow(runtime))) {
    if (as.character(packageVersion(runtime$package[i])) != runtime$version[i]) {
      stop("Runtime dependency changed: ", runtime$package[i])
    }
  }
}

make_frame <- function(x) {
  # Wide metadata bounds stop FCS readers from censoring valid negative or high values.
  ff <- flowCore::flowFrame(x)
  ff@parameters@data$minRange <- rep(-1e30, ncol(x))
  ff@parameters@data$maxRange <- rep(1e30, ncol(x))
  ff@parameters@data$range <- rep(1e30, ncol(x))
  ff
}

fit_cytonorm <- function(input, markers, config, output) {
  stopifnot(identical(names(input), c("cell_id", "dataset", "unit_key", markers)))
  if (anyNA(input) || anyDuplicated(input$cell_id) || any(!nzchar(input$cell_id)) ||
      any(!nzchar(input$dataset)) || any(!nzchar(input$unit_key))) stop("Invalid row identity or missing data")
  if (!all(vapply(input[markers], is.numeric, logical(1)))) stop("Markers must be numeric")
  x <- as.matrix(input[markers])
  if (!all(is.finite(x))) stop("Nonfinite input")
  batches <- unique(input$dataset)
  if (length(batches) != 3L) stop("Expected exactly three dataset/platform batches")
  if (any(vapply(split(input$dataset, input$unit_key), function(z) length(unique(z)), 0L) != 1L)) stop("Unit belongs to multiple batches")
  if (".row_id" %in% markers) stop("Reserved marker name")
  RNGkind("Mersenne-Twister", "Inversion", "Rejection")
  set.seed(config$seed)
  work <- file.path(output, "work")
  dir.create(work)
  # Fixed cap per donor/sample. These are proxy controls, not technical replicates.
  selected <- unlist(lapply(unique(input$unit_key), function(u) {
    rows <- which(input$unit_key == u)
    sort(rows[sample.int(length(rows), min(length(rows), config$aggregate_cells_per_unit))])
  }), use.names = FALSE)
  write_table(input[selected, c("cell_id", "dataset", "unit_key")], file.path(output, "aggregate_cells.csv.gz"))
  aggregate_files <- file.path(work, sprintf("aggregate_%02d.fcs", seq_along(batches)))
  for (i in seq_along(batches)) {
    rows <- selected[input$dataset[selected] == batches[i]]
    flowCore::write.FCS(make_frame(x[rows, , drop = FALSE]), aggregate_files[i], what = "double")
  }
  fs_params <- config$flowsom
  fs_params$colsToUse <- markers
  fs_params$compensate <- FALSE
  fs_params$transform <- FALSE
  # Do not estimate new cluster counts after inspecting biological scores.
  train_dir <- file.path(work, "train")
  dir.create(train_dir)
  norm_params <- c(config$quantile, list(transformation = FALSE, truncate_max_range = FALSE, min.limit = -Inf))
  model <- CytoNorm::CytoNorm.train(files = aggregate_files, labels = batches,
    channels = markers, transformList = NULL, outputDir = train_dir,
    FlowSOM.params = fs_params, normParams = norm_params, seed = config$seed,
    recompute = TRUE, clean = TRUE, plot = FALSE, verbose = TRUE,
    transformation = FALSE, truncate_max_range = FALSE, min.limit = -Inf)
  saveRDS(model, file.path(output, "model.rds"))
  # A numeric row index is carried as a non-fitting channel, preserving exact row identity.
  frames <- lapply(seq_along(batches), function(i) {
    rows <- which(input$dataset == batches[i])
    make_frame(cbind(x[rows, , drop = FALSE], .row_id = as.double(rows)))
  })
  fs <- flowCore::flowSet(frames)
  flowCore::sampleNames(fs) <- sprintf("batch_%02d", seq_along(batches))
  clusters <- integer(nrow(input))
  for (i in seq_along(batches)) {
    rows <- which(input$dataset == batches[i])
    clusters[rows] <- FlowSOM::GetMetaclusters(FlowSOM::NewData(model$fsom, frames[[i]]))
  }
  # Public extension point: change FCS read options only, not correction formulas.
  normalize_unclipped <- function(...) CytoNorm::QuantileNorm.normalize(...,
    transformation = FALSE, truncate_max_range = FALSE, min.limit = -Inf)
  normalized <- CytoNorm::CytoNorm.normalize(model, fs, labels = batches,
    transformList = NULL, transformList.reverse = NULL,
    outputDir = file.path(work, "normalize"), normMethod.normalize = normalize_unclipped,
    clean = TRUE, write = FALSE, verbose = TRUE,
    transformation = FALSE, truncate_max_range = FALSE, min.limit = -Inf)
  combined <- do.call(rbind, lapply(seq_along(batches), function(i) flowCore::exprs(normalized[[i]])))
  rid <- combined[, ".row_id"]
  if (anyDuplicated(rid) || !setequal(rid, seq_len(nrow(input)))) stop("FCS row identity lost")
  y <- combined[match(seq_len(nrow(input)), rid), markers, drop = FALSE]
  if (!identical(dim(x), dim(y)) || !all(is.finite(y))) stop("Invalid corrected matrix")
  # Independently replay public model splines, accounting for upstream float32 FCS I/O.
  x_fcs <- matrix(as.double(readBin(writeBin(as.double(x), raw(), size = 4L), "double",
                                    n = length(x), size = 4L)), nrow(x), ncol(x), dimnames = dimnames(x))
  expected <- x_fcs
  modes <- list(); nr <- 0L
  for (cl in sort(unique(clusters))) for (b in batches) {
    rows <- which(clusters == cl & input$dataset == b)
    cluster_model <- model$clusterRes[[as.character(cl)]]
    for (m in markers) {
      fn <- cluster_model$splines[[b]][[m]]
      if (is.null(fn)) stop("Missing spline for cluster/batch/marker")
      expected[rows, m] <- fn(x_fcs[rows, m])
      q <- cluster_model$quantiles[[b]][, m]
      identity <- anyNA(q) || length(unique(q)) <= 1L || anyNA(cluster_model$refQuantiles)
      nr <- nr + 1L
      modes[[nr]] <- data.frame(cluster = cl, dataset = b, marker = m,
        cells = length(rows), identity_fallback = identity,
        quantile_min = if (all(is.na(q))) NA_real_ else min(q, na.rm = TRUE),
        quantile_max = if (all(is.na(q))) NA_real_ else max(q, na.rm = TRUE),
        extrapolated_cells = if (all(is.na(q))) NA_integer_ else sum(x_fcs[rows, m] < min(q) | x_fcs[rows, m] > max(q)))
    }
  }
  expected <- matrix(readBin(writeBin(as.double(expected), raw(), size = 4L), "double",
                             n = length(expected), size = 4L), nrow(x), ncol(x))
  err <- max(abs(y - expected))
  if (!isTRUE(all.equal(unname(y), unname(expected), tolerance = 1e-7))) stop("Official output differs from independent spline replay: ", err)
  write_table(data.frame(cell_id = input$cell_id, y, check.names = FALSE), file.path(output, "corrected_expression.csv.gz"))
  write_table(input[c("cell_id", "dataset", "unit_key")], file.path(output, "row_metadata.csv.gz"))
  write_table(data.frame(cell_id = input$cell_id, dataset = input$dataset,
                         unit_key = input$unit_key, cluster = clusters), file.path(output, "cell_clusters.csv.gz"))
  write_table(do.call(rbind, modes), file.path(output, "spline_diagnostics.csv"))
  counts <- as.data.frame(table(dataset = factor(input$dataset, levels = batches), cluster = clusters))
  names(counts)[3L] <- "cells"
  counts$fraction <- counts$cells / as.numeric(table(factor(input$dataset, levels = batches)))[match(counts$dataset, batches)]
  write_table(counts, file.path(output, "cluster_batch_coverage.csv"))
  cv <- aggregate(fraction ~ cluster, counts, function(z) sd(z) / mean(z))
  names(cv)[2L] <- "batch_fraction_cv"
  cv$interpretation <- "Descriptive only: independent cohorts can have different true cell-type proportions."
  write_table(cv, file.path(output, "cluster_cv.csv"))
  # Remove only run-owned intermediary FCS files after outputs and replay checks pass.
  unlink(work, recursive = TRUE)
  list(cells = nrow(x), markers = ncol(x), units = length(unique(input$unit_key)),
       batches = batches, aggregate_cells = length(selected), seed = config$seed,
       spline_replay_max_absolute_error = err, row_identity_verified = TRUE,
       source_labels_used = FALSE, nonfinite_outputs = sum(!is.finite(y)),
       identity_fallback_cells_markers = sum(do.call(rbind, modes)$cells[do.call(rbind, modes)$identity_fallback]),
       fcs_precision = "Official intermediate FCS uses float32; independently replayed and verified.",
       applicability = "Exploratory cross-platform application; successful execution is not validation of clustering assumptions.")
}
