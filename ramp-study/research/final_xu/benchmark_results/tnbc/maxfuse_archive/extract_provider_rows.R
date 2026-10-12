#!/usr/bin/env Rscript

# Extract the exact frozen Tietscher and Meyer cells directly from the provider
# SingleCellExperiment objects.  This bridge is deliberately local to the TNBC
# MaxFuse v2 package: it reads only the canonical identity/order columns and the
# provider `counts` assays, writes a temporary HDF5 export requested by
# build_inputs.py, and never reads an alignment, UMAP, cell-type label, or
# clinical outcome.

suppressPackageStartupMessages(library(SingleCellExperiment))
suppressPackageStartupMessages(library(rhdf5))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4L) {
  stop(
    "Usage: extract_provider_rows.R CANONICAL_METADATA TIETSCHER_RDS ",
    "MEYER_RDS OUTPUT_DIRECTORY"
  )
}

metadata_path <- normalizePath(args[[1]], mustWork = TRUE)
tietscher_path <- normalizePath(args[[2]], mustWork = TRUE)
meyer_path <- normalizePath(args[[3]], mustWork = TRUE)
output_directory <- normalizePath(args[[4]], mustWork = FALSE)
dir.create(output_directory, recursive = TRUE, showWarnings = FALSE)

identity_columns <- c(
  "cell_order", "source_cell_id", "source_index_zero_based", "dataset"
)
header <- names(read.csv(
  metadata_path,
  nrows = 0L,
  check.names = FALSE,
  stringsAsFactors = FALSE
))
missing_columns <- setdiff(identity_columns, header)
if (length(missing_columns) != 0L) {
  stop("Canonical metadata lacks: ", paste(missing_columns, collapse = ", "))
}
column_classes <- rep("NULL", length(header))
names(column_classes) <- header
column_classes[match("cell_order", header)] <- "integer"
column_classes[match("source_cell_id", header)] <- "character"
column_classes[match("source_index_zero_based", header)] <- "integer"
column_classes[match("dataset", header)] <- "character"
metadata <- read.csv(
  metadata_path,
  colClasses = column_classes,
  check.names = FALSE,
  stringsAsFactors = FALSE
)
if (!identical(metadata$cell_order, seq.int(0L, nrow(metadata) - 1L))) {
  stop("Canonical cell_order is not the exact metadata row order")
}

write_selected_counts <- function(
  dataset,
  provider_path,
  stem,
  expected_cells
) {
  selected_metadata <- metadata[metadata$dataset == dataset, , drop = FALSE]
  if (nrow(selected_metadata) != expected_cells) {
    stop(dataset, ": expected ", expected_cells, " frozen cells, observed ",
         nrow(selected_metadata))
  }
  selected <- as.integer(selected_metadata$source_index_zero_based) + 1L
  if (anyNA(selected) || any(selected < 1L) || anyDuplicated(selected)) {
    stop(dataset, ": invalid frozen provider indices")
  }

  sce <- readRDS(provider_path)
  if (!inherits(sce, "SingleCellExperiment")) {
    stop(dataset, ": provider object is not a SingleCellExperiment")
  }
  if (!("counts" %in% SummarizedExperiment::assayNames(sce))) {
    stop(dataset, ": provider object lacks the counts assay")
  }
  if (any(selected > ncol(sce))) {
    stop(dataset, ": frozen provider index exceeds SCE cell count")
  }
  observed_ids <- as.character(colnames(sce)[selected])
  expected_ids <- as.character(selected_metadata$source_cell_id)
  if (!identical(observed_ids, expected_ids)) {
    mismatch <- which(observed_ids != expected_ids)[[1L]]
    stop(dataset, ": frozen provider cell ID mismatch at local row ", mismatch)
  }

  values <- as.matrix(
    SummarizedExperiment::assay(sce, "counts")[, selected, drop = FALSE]
  )
  storage.mode(values) <- "double"
  if (any(!is.finite(values)) || any(values < 0)) {
    stop(dataset, ": provider counts must be finite and nonnegative")
  }
  source_markers <- as.character(rownames(sce))
  if (length(source_markers) != nrow(values) || anyDuplicated(source_markers)) {
    stop(dataset, ": provider marker names are invalid")
  }

  output_path <- file.path(output_directory, paste0(stem, ".h5"))
  if (file.exists(output_path) && !file.remove(output_path)) {
    stop("Cannot replace temporary provider export: ", output_path)
  }
  h5createFile(output_path)
  h5createDataset(
    output_path,
    "X_features_by_cells",
    dims = dim(values),
    storage.mode = "double",
    chunk = c(nrow(values), min(1000L, ncol(values))),
    level = 6
  )
  h5write(values, output_path, "X_features_by_cells")
  h5write(source_markers, output_path, "source_markers")
  h5write(observed_ids, output_path, "source_cell_ids")
  h5closeAll()
  message(
    "EXTRACTED_PROVIDER_ROWS: ", dataset, ": ", ncol(values),
    " cells x ", nrow(values), " markers"
  )
  rm(sce, values)
  invisible(gc())
}

write_selected_counts(
  "Tietscher IMC", tietscher_path, "tietscher_provider_rows", 20000L
)
write_selected_counts(
  "Meyer IMC", meyer_path, "meyer_provider_rows", 22701L
)
message("TNBC_PROVIDER_EXTRACTION_OK")
