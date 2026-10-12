#!/usr/bin/env Rscript

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_argument)) stop("Cannot determine script location.", call. = FALSE)
script_path <- sub("^--file=", "", script_argument[[1L]])
script_path <- gsub("~+~", " ", script_path, fixed = TRUE)
script_path <- normalizePath(script_path, mustWork = TRUE)
algorithm_dir <- dirname(script_path)
library_dir <- file.path(algorithm_dir, ".Rlib")
archive <- file.path(algorithm_dir, "vendor", "batchelor_1.18.1.tar.gz")
expected_sha256 <- "a58590880cf00dd2ff7c9fc75ee611e4d4cc5bae98246a1878f8d7d1da6c8dfc"

dir.create(library_dir, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(library_dir, .libPaths()))

if (!requireNamespace("digest", quietly = TRUE)) {
  stop("The bootstrap requires digest to verify source files.", call. = FALSE)
}
if (!file.exists(archive)) stop("The vendored batchelor source archive is missing.", call. = FALSE)
actual_sha256 <- digest::digest(file = archive, algo = "sha256", serialize = FALSE)
if (!identical(actual_sha256, expected_sha256)) {
  stop(sprintf(
    "Source archive SHA-256 mismatch: expected %s, found %s.",
    expected_sha256, actual_sha256
  ), call. = FALSE)
}

runtime_lock <- utils::read.csv(
  file.path(algorithm_dir, "runtime_lock.csv"),
  stringsAsFactors = FALSE,
  check.names = FALSE
)
if (!identical(colnames(runtime_lock), c("package", "version", "role"))) {
  stop("runtime_lock.csv has an invalid schema.", call. = FALSE)
}

dependency_lock <- runtime_lock[!runtime_lock$package %in% c("R", "batchelor"), , drop = FALSE]
actual_versions <- vapply(dependency_lock$package, function(package) {
  path <- find.package(package, quiet = TRUE)
  if (!length(path) || !nzchar(path)) return(NA_character_)
  as.character(utils::packageVersion(package))
}, character(1))
bad <- is.na(actual_versions) | actual_versions != dependency_lock$version
if (any(bad)) {
  stop(paste0(
    "Missing or mismatched locked dependencies: ",
    paste(sprintf(
      "%s expected %s found %s",
      dependency_lock$package[bad], dependency_lock$version[bad], actual_versions[bad]
    ), collapse = "; "),
    ". Restore runtime_lock.csv before rerunning; this bootstrap will not download or upgrade packages."
  ), call. = FALSE)
}
if (!identical(as.character(getRversion()), runtime_lock$version[runtime_lock$package == "R"])) {
  stop("The current R version does not match runtime_lock.csv.", call. = FALSE)
}

utils::install.packages(
  archive,
  repos = NULL,
  type = "source",
  lib = library_dir,
  dependencies = FALSE
)
if (!identical(
    as.character(utils::packageVersion("batchelor", lib.loc = library_dir)),
    "1.18.1")) {
  stop("Installed batchelor version is not 1.18.1.", call. = FALSE)
}

writeLines(expected_sha256, file.path(library_dir, ".batchelor_source_sha256"))
source(file.path(algorithm_dir, "fastmnn_repro.R"), local = FALSE)
utils::write.csv(
  current_build_environment_record(),
  file.path(library_dir, "build_environment.csv"),
  row.names = FALSE
)

fingerprint_files <- expected_installed_fingerprint_files()
absolute_fingerprint_files <- file.path(library_dir, fingerprint_files)
if (any(!file.exists(absolute_fingerprint_files))) {
  stop("The installed batchelor package is missing required files.", call. = FALSE)
}
fingerprints <- data.frame(
  file = fingerprint_files,
  sha256 = vapply(absolute_fingerprint_files, sha256_file, character(1)),
  stringsAsFactors = FALSE
)
utils::write.csv(
  fingerprints,
  file.path(library_dir, "installed_package_fingerprints.csv"),
  row.names = FALSE
)

validate_fastmnn_install(library_dir)

package_names <- runtime_lock$package[runtime_lock$package != "R"]
package_table <- data.frame(
  package = package_names,
  version = vapply(package_names, function(package) {
    as.character(utils::packageVersion(package))
  }, character(1)),
  path = vapply(package_names, function(package) {
    normalizePath(find.package(package), mustWork = TRUE)
  }, character(1)),
  stringsAsFactors = FALSE
)
utils::write.csv(
  package_table,
  file.path(library_dir, "validated_runtime_packages.csv"),
  row.names = FALSE
)
writeLines(
  capture.output(sessionInfo()),
  file.path(library_dir, "bootstrap_sessionInfo.txt")
)

cat(sprintf(
  "Installed and validated batchelor 1.18.1 from the verified source archive into %s\n",
  library_dir
))
