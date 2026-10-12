#!/usr/bin/env Rscript

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_argument)) stop("Cannot determine script location.", call. = FALSE)
script_path <- sub("^--file=", "", script_argument[[1L]])
script_path <- gsub("~+~", " ", script_path, fixed = TRUE)
script_path <- normalizePath(script_path, mustWork = TRUE)
algorithm_dir <- dirname(script_path)
library_dir <- file.path(algorithm_dir, ".Rlib")
archive <- file.path(algorithm_dir, "vendor", "cyCombine-v0.2.5.tar.gz")
expected_sha256 <- "8ac3a3827b3f68ca4c1e036b22a2adff594b00fb25d09483648c5191da87eaee"

dir.create(library_dir, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(library_dir, .libPaths()))

if (!requireNamespace("digest", quietly = TRUE)) {
  stop("The bootstrap requires digest to verify the vendored source archive.", call. = FALSE)
}
actual_sha256 <- digest::digest(file = archive, algo = "sha256", serialize = FALSE)
if (!identical(actual_sha256, expected_sha256)) {
  stop(sprintf("Source archive SHA-256 mismatch: expected %s, found %s", expected_sha256, actual_sha256), call. = FALSE)
}

direct_dependencies <- c(
  "magrittr", "flowCore", "RColorBrewer", "kohonen", "tibble", "sva",
  "dplyr", "readxl", "readr", "Biobase", "purrr", "stringr", "tidyr",
  "ggplot2", "knitr", "cytolib", "jsonlite", "digest"
)
missing <- direct_dependencies[!vapply(direct_dependencies, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  stop(paste0(
    "Missing dependencies: ", paste(missing, collapse = ", "), ". ",
    "Restore the official publication environment from vendor/publication-renv.lock ",
    "or install these dependencies into a reviewed R library before rerunning bootstrap."
  ), call. = FALSE)
}

utils::install.packages(archive, repos = NULL, type = "source", lib = library_dir, dependencies = FALSE)
if (!identical(as.character(utils::packageVersion("cyCombine", lib.loc = library_dir)), "0.2.5")) {
  stop("Installed cyCombine version is not 0.2.5.", call. = FALSE)
}
writeLines(expected_sha256, file.path(library_dir, ".cycombine_source_sha256"))
source(file.path(algorithm_dir, "cycombine_repro.R"), local = FALSE)
validate_cycombine_install(library_dir)

package_names <- unique(c("cyCombine", direct_dependencies))
package_table <- utils::installed.packages(lib.loc = .libPaths())
package_table <- package_table[intersect(package_names, rownames(package_table)), c("Package", "Version", "LibPath"), drop = FALSE]
utils::write.csv(package_table, file.path(library_dir, "validated_direct_runtime_packages.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(library_dir, "bootstrap_sessionInfo.txt"))

cat(sprintf("Installed cyCombine 0.2.5 from the verified publication archive into %s\n", library_dir))
