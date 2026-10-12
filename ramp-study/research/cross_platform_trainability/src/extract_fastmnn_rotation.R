# Extract fastMNN rotation and centres from fit.rds for the three application
# tasks. Writes CSVs into data/<task>/. Run once. Reads final_xu, never writes there.
args <- commandArgs(trailingOnly = TRUE)
src <- args[1]; task <- args[2]; out <- args[3]
fit <- readRDS(file.path(src, "fit.rds"))
dir.create(out, recursive = TRUE, showWarnings = FALSE)
write.csv(fit$pca$rotation, file.path(out, "fastmnn_rotation.csv"))
write.csv(fit$pca$centers,  file.path(out, "fastmnn_centers.csv"))
cat(task, ": rotation", paste(dim(fit$pca$rotation), collapse="x"),
    " centers", paste(dim(fit$pca$centers), collapse="x"), "\n")
