#!/usr/bin/env Rscript

# Export the frozen Shovik CODEX cohort used by the bone-marrow benchmark:
# all 12 normal-bone-marrow donors, 2,500 cells per donor.
#
# Sampling reads donor identity only. Marker values, provider annotations,
# spatial coordinates, and downstream results are not used to select cells.

suppressPackageStartupMessages({
  library(SeuratObject)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("usage: export_codex_30k.R SOURCE_RDS OUTPUT_DIR")
}

source_rds <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

sample_seed <- 20260820L
cells_per_donor <- 2500L
RNGversion("4.0.0")

source_markers <- c(
  "CD117", "CD11B", "CD11C", "CD123", "CD14", "CD3e", "CD33",
  "CD34", "CD38", "CD4", "CD45", "CD45RA", "CD61", "CD8", "CD19"
)
output_markers <- c(
  "CD117", "CD11B", "CD11C", "CD123", "CD14", "CD3", "CD33",
  "CD34", "CD38", "CD4", "CD45", "CD45RA", "CD61", "CD8", "CD19"
)

object <- readRDS(source_rds)
if (!("CODEX" %in% names(object@assays))) stop("CODEX assay is missing")

counts <- GetAssayData(object, assay = "CODEX", slot = "counts")
missing_markers <- setdiff(source_markers, rownames(counts))
if (length(missing_markers) > 0L) {
  stop(sprintf("missing CODEX markers: %s", paste(missing_markers, collapse = ", ")))
}

metadata <- object@meta.data
metadata$source_cell_id <- rownames(metadata)
metadata$donor_id <- as.character(metadata$orig.ident)
donors <- sort(unique(metadata$donor_id))
if (length(donors) != 12L) stop("expected exactly 12 CODEX donors")

selected <- integer(0)
for (i in seq_along(donors)) {
  candidates <- which(metadata$donor_id == donors[[i]])
  if (length(candidates) < cells_per_donor) {
    stop(sprintf("%s has fewer than 2,500 cells", donors[[i]]))
  }
  set.seed(sample_seed + i * 1009L)
  selected <- c(selected, sample(candidates, cells_per_donor, replace = FALSE))
}
selected <- sort(selected)
metadata <- metadata[selected, , drop = FALSE]

matrix <- t(as.matrix(counts[source_markers, metadata$source_cell_id, drop = FALSE]))
colnames(matrix) <- output_markers
if (!identical(rownames(matrix), metadata$source_cell_id)) {
  stop("CODEX matrix and metadata order differ")
}

matrix_frame <- data.frame(
  cell_id = rownames(matrix),
  dataset = "BM CODEX",
  donor = metadata$donor_id,
  unit_key = paste0("BM CODEX|", metadata$donor_id),
  matrix,
  check.names = FALSE,
  stringsAsFactors = FALSE
)
write.csv(
  matrix_frame,
  file.path(output_dir, "source_input_15.csv"),
  row.names = FALSE,
  quote = TRUE
)

metadata$cell_id <- metadata$source_cell_id
metadata$dataset <- "BM CODEX"
metadata$donor <- metadata$donor_id
metadata <- metadata[, c(
  "cell_id", "dataset", "donor",
  setdiff(colnames(metadata), c("cell_id", "dataset", "donor"))
), drop = FALSE]
write.csv(
  metadata,
  file.path(output_dir, "locked_metadata.csv"),
  row.names = FALSE,
  quote = TRUE
)

write.csv(
  data.frame(
    dataset = "BM CODEX",
    donor = donors,
    source_cells = as.integer(table(object@meta.data$orig.ident)[donors]),
    sampled_cells = cells_per_donor,
    stringsAsFactors = FALSE
  ),
  file.path(output_dir, "donor_sampling.csv"),
  row.names = FALSE,
  quote = TRUE
)

cat(sprintf("Exported %d CODEX cells from %d donors\n", nrow(matrix), length(donors)))
