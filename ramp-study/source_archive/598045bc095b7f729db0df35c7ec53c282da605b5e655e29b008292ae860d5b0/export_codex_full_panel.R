
suppressPackageStartupMessages({
  library(SeuratObject)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop("usage: export.R SOURCE_RDS LOCKED_METADATA_CSV_GZ OUTPUT_CSV_GZ")
}
options(digits = 17)
object <- readRDS(normalizePath(args[[1]], mustWork = TRUE))
locked <- read.csv(gzfile(args[[2]]), stringsAsFactors = FALSE, check.names = FALSE)
if (!("cell_id" %in% colnames(locked))) stop("row lock lacks cell_id")
ids <- as.character(locked$cell_id)
if (anyDuplicated(ids)) stop("CODEX row-lock cell IDs are not unique")
if (!("CODEX" %in% names(object@assays))) stop("CODEX assay is absent")
counts <- GetAssayData(object, assay = "CODEX", slot = "counts")
if (nrow(counts) != 49L) stop(sprintf("expected 49 CODEX proteins; found %d", nrow(counts)))
missing <- setdiff(ids, colnames(counts))
if (length(missing) > 0L) stop(sprintf("CODEX source misses %d frozen IDs", length(missing)))
block <- t(as.matrix(counts[, ids, drop = FALSE]))
if (!identical(rownames(block), ids)) stop("CODEX provider extraction changed row order")
frame <- data.frame(cell_id = ids, block, check.names = FALSE, stringsAsFactors = FALSE)
connection <- gzfile(args[[3]], open = "wt")
write.table(
  frame, connection,
  sep = ",", row.names = FALSE, col.names = TRUE, quote = FALSE, na = "NA"
)
close(connection)
