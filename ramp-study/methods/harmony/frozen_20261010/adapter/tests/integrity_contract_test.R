#!/usr/bin/env Rscript

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_argument)) stop("Cannot determine script location.", call. = FALSE)
test_path <- sub("^--file=", "", script_argument[[1L]])
test_path <- gsub("~+~", " ", test_path, fixed = TRUE)
test_path <- normalizePath(test_path, mustWork = TRUE)
algorithm_dir <- dirname(dirname(test_path))
local_library <- file.path(algorithm_dir, ".Rlib")
.libPaths(c(local_library, .libPaths()))
source(file.path(algorithm_dir, "harmony_direct_repro.R"), local = FALSE)

if (isNamespaceLoaded("harmony")) {
  stop("Harmony was loaded merely by sourcing the audited wrapper.", call. = FALSE)
}

expect_error <- function(expression, label) {
  failed <- FALSE
  tryCatch(
    force(expression),
    error = function(...) failed <<- TRUE
  )
  if (!failed) stop(sprintf("Expected integrity failure did not occur: %s.", label), call. = FALSE)
  invisible(TRUE)
}

copy_relative_files <- function(source_root, destination_root, relative_files) {
  for (relative_file in relative_files) {
    source_path <- file.path(source_root, relative_file)
    destination_path <- file.path(destination_root, relative_file)
    dir.create(dirname(destination_path), recursive = TRUE, showWarnings = FALSE)
    if (!file.copy(source_path, destination_path, overwrite = TRUE, copy.mode = TRUE)) {
      stop(sprintf("Could not copy integrity fixture: %s.", relative_file), call. = FALSE)
    }
  }
}

test_root <- tempfile("harmony-integrity-contract-")
dir.create(test_root, recursive = TRUE)
on.exit(unlink(test_root, recursive = TRUE, force = TRUE), add = TRUE)

# An exact expected-set manifest passes; a one-row manifest cannot bless an
# unlisted modification, and a listed-file modification is detected.
fixture_library <- file.path(test_root, "fingerprint_library")
dir.create(fixture_library)
fingerprint_files <- expected_installed_fingerprint_files()
copy_relative_files(local_library, fixture_library, fingerprint_files)
manifest_name <- "installed_package_fingerprints.csv"
file.copy(
  file.path(local_library, manifest_name),
  file.path(fixture_library, manifest_name),
  overwrite = TRUE
)
validate_installed_fingerprint_manifest(fixture_library)
manifest <- utils::read.csv(
  file.path(fixture_library, manifest_name),
  stringsAsFactors = FALSE,
  check.names = FALSE
)
utils::write.csv(
  manifest[1L, , drop = FALSE],
  file.path(fixture_library, manifest_name),
  row.names = FALSE,
  quote = TRUE
)
expect_error(
  validate_installed_fingerprint_manifest(fixture_library),
  "truncated fingerprint manifest"
)
utils::write.csv(
  manifest,
  file.path(fixture_library, manifest_name),
  row.names = FALSE,
  quote = TRUE
)
tampered_file <- file.path(fixture_library, "RhpcBLASctl", "NAMESPACE")
writeLines(c(readLines(tampered_file, warn = FALSE), "# integrity-test tamper"), tampered_file)
expect_error(
  validate_installed_fingerprint_manifest(fixture_library),
  "modified installed package file"
)

# The lock is authenticated as a complete file, so a syntactically valid but
# header-only CSV cannot silently disable version checks.
lock_algorithm_dir <- file.path(test_root, "runtime_lock_algorithm")
dir.create(lock_algorithm_dir)
file.copy(
  file.path(algorithm_dir, "runtime_lock.csv"),
  file.path(lock_algorithm_dir, "runtime_lock.csv"),
  overwrite = TRUE
)
validate_runtime_lock_file(lock_algorithm_dir, check_versions = FALSE)
writeLines("package,version,role", file.path(lock_algorithm_dir, "runtime_lock.csv"))
expect_error(
  validate_runtime_lock_file(lock_algorithm_dir, check_versions = FALSE),
  "header-only runtime lock"
)

# All archives authenticate before installation; corruption of any later
# archive is detected independently.
archive_algorithm_dir <- file.path(test_root, "archive_algorithm")
archive_vendor_dir <- file.path(archive_algorithm_dir, "vendor")
dir.create(archive_vendor_dir, recursive = TRUE)
archive_paths <- source_archive_paths(algorithm_dir)
file.copy(archive_paths, archive_vendor_dir, overwrite = TRUE)
validate_source_archives(archive_algorithm_dir)
corrupt_archive <- source_archive_paths(archive_algorithm_dir)[["RhpcBLASctl"]]
writeBin(charToRaw("integrity-test corruption"), corrupt_archive)
expect_error(validate_source_archives(archive_algorithm_dir), "corrupt source archive")

# Most importantly, a corrupted native Harmony library must be rejected before
# requireNamespace() can load it.
preload_algorithm_dir <- file.path(test_root, "preload_algorithm")
preload_library <- file.path(preload_algorithm_dir, ".Rlib")
dir.create(preload_library, recursive = TRUE)
dir.create(file.path(preload_algorithm_dir, "vendor"), recursive = TRUE)
file.copy(
  source_archive_paths(algorithm_dir),
  file.path(preload_algorithm_dir, "vendor"),
  overwrite = TRUE
)
file.copy(
  file.path(algorithm_dir, "runtime_lock.csv"),
  file.path(preload_algorithm_dir, "runtime_lock.csv"),
  overwrite = TRUE
)
copy_relative_files(local_library, preload_library, fingerprint_files)
for (file_name in c(
    "installed_package_fingerprints.csv", "build_environment.csv",
    ".harmony_source_sha256", ".RcppProgress_source_sha256",
    ".RhpcBLASctl_source_sha256")) {
  file.copy(
    file.path(local_library, file_name),
    file.path(preload_library, file_name),
    overwrite = TRUE
  )
}
native_library <- file.path(
  preload_library,
  "harmony",
  "libs",
  paste0("harmony", .Platform$dynlib.ext)
)
writeBin(charToRaw("not a native library"), native_library)
expect_error(
  validate_harmony_install(preload_library),
  "native-library tamper before namespace load"
)
if (isNamespaceLoaded("harmony")) {
  stop("Corrupted Harmony namespace was loaded before integrity rejection.", call. = FALSE)
}

cat(paste0(
  "PASS: Harmony integrity contract; exact fingerprint set; complete runtime lock; ",
  "all archives preverified; native tamper rejected before namespace load\n"
))
