# Visualization-only reconstruction of the existing saved fit; no fitting.
args <- commandArgs(trailingOnly=TRUE)
source_dir <- normalizePath(args[1],mustWork=TRUE)
out <- normalizePath(args[2],mustWork=TRUE)
fit <- readRDS(file.path(source_dir,'methods/fastmnn/fit.rds'))
markers <- readLines(file.path(source_dir,'input/markers.txt'))
meta <- read.csv(gzfile(file.path(source_dir,'input/fit_metadata.csv.gz')),stringsAsFactors=FALSE)
stored <- read.csv(gzfile(file.path(source_dir,'methods/fastmnn/corrected_expression.csv.gz')),check.names=FALSE)
stopifnot(identical(fit$cell_metadata$cell_id,meta$cell_id),
          identical(stored$cell_id,meta$cell_id),
          identical(rownames(fit$pca$rotation),markers),
          isTRUE(all.equal(unname(fit$corrected_embedding),
                           unname(as.matrix(stored[,-1])),tolerance=1e-12)))
# Same crossproduct as batchelor::convertPCsToSCE; no centering/norm restoration.
y <- fit$corrected_embedding %*% t(fit$pca$rotation)
stopifnot(nrow(y)==nrow(meta),ncol(y)==length(markers),all(is.finite(y)))
colnames(y) <- markers
write.csv(data.frame(cell_id=meta$cell_id,y,check.names=FALSE),
          gzfile(file.path(out,'fastmnn_reconstructed.csv.gz')),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(out,'fastmnn_reconstruction_sessionInfo.txt'))
cat('Reconstructed saved fastMNN:',nrow(y),'cells x',ncol(y),'markers\n')
