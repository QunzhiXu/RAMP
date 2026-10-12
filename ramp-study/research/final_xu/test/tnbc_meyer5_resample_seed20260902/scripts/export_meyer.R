#!/usr/bin/env Rscript

# Read-only bridge for the Meyer SingleCellExperiment.  Donor selection is
# performed in Python from a PID-only candidate table; provider labels are
# opened only after the selected donor list has been frozen.

suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(SummarizedExperiment)
  library(data.table)
  library(rhdf5)
})

args <- commandArgs(trailingOnly = TRUE)
if (!(length(args) %in% c(3L, 4L))) {
  stop(
    "Usage: export_meyer.R MODE MEYER_RDS OUTPUT_DIR [SELECTED_DONORS_CSV]",
    call. = FALSE
  )
}

mode <- args[[1L]]
rds_path <- normalizePath(args[[2L]], mustWork = TRUE)
output_dir <- normalizePath(args[[3L]], mustWork = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

sce <- readRDS(rds_path)
if (!inherits(sce, "SingleCellExperiment")) {
  stop("Meyer source is not a SingleCellExperiment", call. = FALSE)
}
if (!("counts" %in% SummarizedExperiment::assayNames(sce))) {
  stop("Meyer source lacks the counts assay", call. = FALSE)
}
cd <- as.data.frame(SummarizedExperiment::colData(sce))
if (!("PID" %in% names(cd))) stop("Meyer colData lacks PID", call. = FALSE)
pid <- as.character(cd$PID)
if (anyNA(pid) || any(!nzchar(pid))) stop("Meyer PID is incomplete", call. = FALSE)

if (identical(mode, "candidates")) {
  counts <- as.data.table(as.data.frame(table(pid), stringsAsFactors = FALSE))
  setnames(counts, c("donor", "cells"))
  counts[, donor := as.character(donor)]
  counts[, cells := as.integer(cells)]
  setorder(counts, donor)
  fwrite(counts, file.path(output_dir, "meyer_candidate_donor_counts.csv"))
  message("MEYER_CANDIDATES_OK: ", nrow(counts), " donors; ", sum(counts$cells), " cells")
  quit(save = "no", status = 0L)
}

if (identical(mode, "coverage")) {
  if (!("metacluster_combined" %in% names(cd))) {
    stop("Meyer colData lacks metacluster_combined", call. = FALSE)
  }
  coverage <- data.table(
    donor = pid,
    provider_celltype = as.character(cd$metacluster_combined)
  )[, .(cells = .N), by = .(donor, provider_celltype)]
  setorder(coverage, donor, provider_celltype)
  fwrite(coverage, file.path(output_dir, "meyer_donor_celltype_counts.csv"))
  message("MEYER_COVERAGE_OK: ", uniqueN(coverage$donor), " donors")
  quit(save = "no", status = 0L)
}

if (!identical(mode, "selected") || length(args) != 4L) {
  stop("MODE must be candidates or selected", call. = FALSE)
}

selected_path <- normalizePath(args[[4L]], mustWork = TRUE)
selected_table <- fread(selected_path, colClasses = c(donor = "character"))
if (!("donor" %in% names(selected_table))) {
  stop("Selected donor table lacks donor", call. = FALSE)
}
selected <- unique(as.character(selected_table$donor))
if (length(selected) != 5L || anyNA(selected) || any(!nzchar(selected))) {
  stop("Exactly five unique selected Meyer donors are required", call. = FALSE)
}
missing_donors <- setdiff(selected, unique(pid))
if (length(missing_donors)) {
  stop("Selected donors absent from Meyer source: ", paste(missing_donors, collapse = ", "))
}

# The source positions are frozen before any provider cell label is read.
positions <- which(pid %in% selected)
if (!length(positions)) stop("No selected Meyer cells", call. = FALSE)

source_markers <- c(
  CD11C = "CD11c",
  CD20 = "CD20",
  CD3 = "CD3",
  CD31 = "CD31_vWF",
  CD4 = "CD4",
  CD68 = "CD68",
  CD8 = "CD8a",
  HLA_DR = "HLADR",
  KI67 = "Ki67",
  PANCK = "panCK",
  SMA = "SMA"
)
if (!all(source_markers %in% rownames(sce))) {
  stop("Meyer source lacks one or more benchmark markers", call. = FALSE)
}

values <- as.matrix(
  SummarizedExperiment::assay(sce, "counts")[source_markers, positions, drop = FALSE]
)
storage.mode(values) <- "double"
if (any(!is.finite(values)) || any(values < 0)) {
  stop("Meyer selected counts must be finite and nonnegative", call. = FALSE)
}

h5_path <- file.path(output_dir, "meyer_selected_counts.h5")
if (file.exists(h5_path) && !file.remove(h5_path)) {
  stop("Cannot replace ", h5_path, call. = FALSE)
}
h5createFile(h5_path)
h5createDataset(
  h5_path,
  "X_features_by_cells",
  dims = dim(values),
  storage.mode = "double",
  chunk = c(nrow(values), min(1000L, ncol(values))),
  level = 6
)
h5write(values, h5_path, "X_features_by_cells")
h5write(as.character(names(source_markers)), h5_path, "canonical_markers")
h5write(as.character(source_markers), h5_path, "source_markers")
h5closeAll()

required_metadata <- c(
  "cell_id", "sample_id", "PID", "Pos_X", "Pos_Y",
  "metacluster_combined", "metacluster_tumor_combined", "cluster_broad"
)
missing_columns <- setdiff(required_metadata, names(cd))
if (length(missing_columns)) {
  stop("Meyer colData lacks: ", paste(missing_columns, collapse = ", "), call. = FALSE)
}

metadata <- data.table(
  source_index_zero_based = as.integer(positions - 1L),
  source_cell_id = as.character(colnames(sce)[positions]),
  donor = as.character(cd$PID[positions]),
  sample_id = as.character(cd$sample_id[positions]),
  x = as.numeric(cd$Pos_X[positions]),
  y = as.numeric(cd$Pos_Y[positions]),
  provider_celltype = as.character(cd$metacluster_combined[positions]),
  provider_cell_subtype = as.character(cd$metacluster_tumor_combined[positions]),
  provider_cell_class = as.character(cd$cluster_broad[positions])
)
if (!identical(metadata$source_cell_id, as.character(cd$cell_id[positions]))) {
  stop("Meyer colnames and colData cell_id differ", call. = FALSE)
}
if (anyDuplicated(metadata$source_cell_id) || anyDuplicated(metadata$source_index_zero_based)) {
  stop("Meyer selected cell identities are duplicated", call. = FALSE)
}
fwrite(metadata, file.path(output_dir, "meyer_selected_metadata.csv.gz"))

selected_counts <- metadata[, .(cells = .N), by = donor][order(donor)]
fwrite(selected_counts, file.path(output_dir, "meyer_selected_donor_counts.csv"))
message(
  "MEYER_SELECTED_OK: ", nrow(metadata), " cells from ",
  uniqueN(metadata$donor), " donors"
)
