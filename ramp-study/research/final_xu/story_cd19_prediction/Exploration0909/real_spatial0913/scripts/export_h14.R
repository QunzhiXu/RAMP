#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(SeuratObject); library(Matrix)})
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2L)
obj <- readRDS(args[[1]])
md <- obj@meta.data
md$cell_id <- rownames(md)
md$donor <- as.character(md$orig.ident)
md <- md[md$donor == "SB67_NBM28_H14_CODEX_Mesmer", , drop=FALSE]
markers <- c("CD117", "CD11B", "CD11C", "CD123", "CD14", "CD3e", "CD33", "CD34", "CD38", "CD4", "CD45", "CD45RA", "CD61", "CD8", "CD19")
x <- t(as.matrix(GetAssayData(obj, assay="CODEX", slot="counts")[markers, md$cell_id, drop=FALSE]))
colnames(x)[colnames(x)=="CD3e"] <- "CD3"
stopifnot(identical(rownames(x), md$cell_id))
write.csv(md, gzfile(file.path(args[[2]], "H14_full_metadata.csv.gz")), row.names=FALSE)
write.csv(data.frame(cell_id=md$cell_id, x, check.names=FALSE), gzfile(file.path(args[[2]], "H14_full_source15.csv.gz")), row.names=FALSE)
cat("Exported all", nrow(md), "H14 cells; no sampling or exclusions\n")
