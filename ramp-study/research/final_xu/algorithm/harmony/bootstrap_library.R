#!/usr/bin/env Rscript

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_argument)) stop("Cannot determine script location.", call. = FALSE)
script_path <- sub("^--file=", "", script_argument[[1L]])
script_path <- gsub("~+~", " ", script_path, fixed = TRUE)
script_path <- normalizePath(script_path, mustWork = TRUE)
algorithm_dir <- dirname(script_path)
local_library <- file.path(algorithm_dir, ".Rlib")
dir.create(local_library, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(local_library, .libPaths()))

if (!requireNamespace("digest", quietly = TRUE)) {
  stop("Package digest must be available in the host R runtime for source verification.", call. = FALSE)
}
source(file.path(algorithm_dir, "harmony_direct_repro.R"), local = FALSE)

# The local R 4.3.2 framework bundles working Fortran runtime libraries, while
# its Makeconf can still point FLIBS at a removed /opt/gfortran installation.
# Use a temporary, project-local Makevars only when that broken condition is
# detected. Never modify ~/.R/Makevars or any global toolchain file.
configured_flibs <- system2(file.path(R.home("bin"), "R"), c("CMD", "config", "FLIBS"), stdout = TRUE)
bundled_fortran <- file.path(R.home("lib"), c("libgfortran.5.dylib", "libquadmath.0.dylib", "libgcc_s.1.1.dylib"))
uses_missing_opt_gfortran <- any(grepl("/opt/gfortran", configured_flibs, fixed = TRUE)) && !dir.exists("/opt/gfortran")
toolchain_workaround_applied <- uses_missing_opt_gfortran && all(file.exists(bundled_fortran))
temporary_makevars <- NULL
old_makevars <- Sys.getenv("R_MAKEVARS_USER", unset = NA_character_)
if (toolchain_workaround_applied) {
  temporary_makevars <- tempfile("harmony-bootstrap-Makevars-")
  writeLines(
    paste("FLIBS =", paste(bundled_fortran, collapse = " ")),
    temporary_makevars
  )
  Sys.setenv(R_MAKEVARS_USER = temporary_makevars)
  on.exit({
    if (is.na(old_makevars)) Sys.unsetenv("R_MAKEVARS_USER") else Sys.setenv(R_MAKEVARS_USER = old_makevars)
    if (!is.null(temporary_makevars) && file.exists(temporary_makevars)) unlink(temporary_makevars)
  }, add = TRUE)
}

archives <- data.frame(
  package = c("RcppProgress", "RhpcBLASctl", "harmony"),
  version = c("0.4.2", "0.23.42", "1.2.3"),
  file = file.path(
    algorithm_dir,
    "vendor",
    c("RcppProgress_0.4.2.tar.gz", "RhpcBLASctl_0.23-42.tar.gz", "harmony_1.2.3.tar.gz")
  ),
  sha256 = c(
    "b1624b21b7aeb1dafb30f092b2a4bef4c3504efd2d6b00b2cdf55dc9df194b48",
    "5c889d5b69e264060b9f1f0383c447f594855b8afc15b7d76d39e4d62b946615",
    "9ab815ab0260a6726b2bb48de3fa4c3cad36acbe9ac6158d515fb16ea66bbddd"
  ),
  stringsAsFactors = FALSE
)

# Authenticate every vendored archive and the complete runtime-lock file before
# any package installation can modify the dedicated library.
validate_source_archives(algorithm_dir)
validate_runtime_lock_file(algorithm_dir, check_versions = FALSE)

current_build_environment <- current_build_environment_record()
build_environment_path <- file.path(local_library, "build_environment.csv")
build_environment_matches <- FALSE
if (file.exists(build_environment_path)) {
  existing_build_environment <- tryCatch(
    utils::read.csv(build_environment_path, stringsAsFactors = FALSE, check.names = FALSE),
    error = function(...) NULL
  )
  build_environment_matches <- is.data.frame(existing_build_environment) &&
    identical(existing_build_environment, current_build_environment)
}

fingerprint_path <- file.path(local_library, "installed_package_fingerprints.csv")
existing_fingerprints_valid <- tryCatch({
  validate_installed_fingerprint_manifest(local_library)
  TRUE
}, error = function(...) FALSE)
existing_source_stamps_valid <- tryCatch({
  validate_source_stamps(local_library)
  TRUE
}, error = function(...) FALSE)
reuse_integrity_ok <- build_environment_matches && existing_fingerprints_valid && existing_source_stamps_valid

for (index in seq_len(nrow(archives))) {
  archive <- archives[index, ]
  installed_path <- find.package(archive$package, lib.loc = local_library, quiet = TRUE)
  installed_version <- installed_version_without_loading(archive$package, required_library = local_library)
  source_stamp <- file.path(local_library, sprintf(".%s_source_sha256", archive$package))
  stamp_ok <- file.exists(source_stamp) && identical(trimws(readLines(source_stamp, warn = FALSE)[1L]), archive$sha256)
  needs_reinstall <- !identical(installed_version, archive$version) || !stamp_ok || !reuse_integrity_ok
  if (needs_reinstall) {
    if (length(installed_path) && nzchar(installed_path)) utils::remove.packages(archive$package, lib = local_library)
    utils::install.packages(
      archive$file,
      repos = NULL,
      type = "source",
      lib = local_library,
      dependencies = FALSE,
      quiet = FALSE
    )
    writeLines(archive$sha256, source_stamp)
  }
  final_version <- installed_version_without_loading(archive$package, required_library = local_library)
  if (!identical(final_version, archive$version)) {
    stop(sprintf("Installed %s version %s instead of %s.", archive$package, final_version, archive$version), call. = FALSE)
  }
}

runtime_lock <- validate_runtime_lock_file(
  algorithm_dir,
  check_versions = TRUE,
  required_library = local_library
)
runtime_packages <- setdiff(runtime_lock$package, "R")
runtime <- data.frame(
  package = runtime_packages,
  version = vapply(
    runtime_packages,
    installed_version_without_loading,
    character(1),
    required_library = local_library
  ),
  library_path = vapply(runtime_packages, function(package) {
    lib_locations <- if (package %in% c("harmony", "RcppProgress", "RhpcBLASctl")) local_library else .libPaths()
    find.package(package, lib.loc = lib_locations)
  }, character(1)),
  stringsAsFactors = FALSE
)
utils::write.csv(runtime, file.path(local_library, "validated_runtime_packages.csv"), row.names = FALSE, quote = TRUE)

fingerprint_relative_files <- expected_installed_fingerprint_files()
fingerprint_files <- file.path(local_library, fingerprint_relative_files)
if (any(!file.exists(fingerprint_files))) {
  stop(sprintf(
    "Installed packages are missing required fingerprint files: %s",
    paste(fingerprint_relative_files[!file.exists(fingerprint_files)], collapse = ", ")
  ), call. = FALSE)
}
fingerprints <- data.frame(
  file = fingerprint_relative_files,
  sha256 = vapply(fingerprint_files, function(path) digest::digest(file = path, algo = "sha256", serialize = FALSE), character(1)),
  stringsAsFactors = FALSE
)
utils::write.csv(
  fingerprints,
  file.path(local_library, "installed_package_fingerprints.csv"),
  row.names = FALSE,
  quote = TRUE
)
utils::write.csv(
  current_build_environment,
  build_environment_path,
  row.names = FALSE,
  quote = TRUE
)
writeLines(capture.output(sessionInfo()), file.path(local_library, "bootstrap_sessionInfo.txt"))
writeLines(
  c(
    sprintf("configured_FLIBS=%s", paste(configured_flibs, collapse = " ")),
    sprintf("temporary_project_makevars_applied=%s", toolchain_workaround_applied),
    sprintf("effective_R_MAKEVARS_USER=%s", Sys.getenv("R_MAKEVARS_USER", unset = "")),
    if (toolchain_workaround_applied) sprintf("temporary_FLIBS=%s", paste(bundled_fortran, collapse = " ")) else "temporary_FLIBS=not_applied"
  ),
  file.path(local_library, "bootstrap_toolchain_record.txt")
)

validate_harmony_install(local_library)
cat(sprintf("PASS: dedicated Harmony %s library is ready at %s\n", HARMONY_EXPECTED_VERSION, local_library))
