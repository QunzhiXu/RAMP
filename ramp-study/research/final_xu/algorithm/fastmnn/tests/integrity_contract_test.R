#!/usr/bin/env Rscript

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_argument)) stop("Cannot determine test location.", call. = FALSE)
script_path <- normalizePath(sub("^--file=", "", script_argument[[1L]]), mustWork = TRUE)
algorithm_dir <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)
local_library <- file.path(algorithm_dir, ".Rlib")
.libPaths(c(local_library, .libPaths()))
source(file.path(algorithm_dir, "fastmnn_repro.R"), local = FALSE)

assert_true <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
}

installation <- validate_fastmnn_install(local_library)
assert_true(
  identical(installation$package_version, FASTMNN_EXPECTED_VERSION),
  "Validated batchelor version differs from the locked version."
)
assert_true(
  identical(installation$source_sha256, FASTMNN_SOURCE_SHA256),
  "Validated batchelor source differs from the locked source archive."
)

archive <- source_archive_path(algorithm_dir)
assert_true(
  identical(sha256_file(archive), FASTMNN_SOURCE_SHA256),
  "Vendored source archive SHA-256 mismatch."
)
runtime_lock_path <- file.path(algorithm_dir, "runtime_lock.csv")
assert_true(
  identical(sha256_file(runtime_lock_path), FASTMNN_RUNTIME_LOCK_SHA256),
  "Runtime lock SHA-256 mismatch."
)

fingerprint_path <- file.path(local_library, "installed_package_fingerprints.csv")
fingerprints <- utils::read.csv(
  fingerprint_path, stringsAsFactors = FALSE, check.names = FALSE
)
assert_true(
  identical(fingerprints$file, expected_installed_fingerprint_files()),
  "Installed-package fingerprint file list is incomplete."
)
assert_true(
  any(grepl(paste0("batchelor\\", .Platform$dynlib.ext, "$"), fingerprints$file)),
  "The compiled batchelor native library is not fingerprinted."
)
current_fingerprints <- vapply(
  file.path(local_library, fingerprints$file), sha256_file, character(1)
)
assert_true(
  identical(unname(current_fingerprints), fingerprints$sha256),
  "An installed batchelor file differs from its post-build fingerprint."
)

namespace <- asNamespace("batchelor")
actual_function_hashes <- vapply(
  names(FASTMNN_FUNCTION_BODY_SHA256),
  function(function_name) {
    function_object <- get(function_name, envir = namespace, inherits = FALSE)
    digest::digest(
      paste(deparse(body(function_object), width.cutoff = 500L), collapse = "\n"),
      algo = "sha256", serialize = FALSE
    )
  },
  character(1)
)
assert_true(
  identical(actual_function_hashes, FASTMNN_FUNCTION_BODY_SHA256),
  "Core installed function bodies differ from the verified implementation."
)

# Demonstrate that a one-byte source or installed-file change crosses the
# exact hashes enforced by validate_fastmnn_install(), without touching the
# real installation.
temporary_archive <- tempfile(fileext = ".tar.gz")
temporary_native <- tempfile(fileext = .Platform$dynlib.ext)
on.exit(unlink(c(temporary_archive, temporary_native), force = TRUE), add = TRUE)
invisible(file.copy(archive, temporary_archive, overwrite = TRUE))
native_index <- grep(paste0("batchelor\\", .Platform$dynlib.ext, "$"), fingerprints$file)
native_path <- file.path(local_library, fingerprints$file[[native_index]])
invisible(file.copy(native_path, temporary_native, overwrite = TRUE))
archive_connection <- file(temporary_archive, open = "ab")
writeBin(as.raw(0x00), archive_connection)
close(archive_connection)
native_connection <- file(temporary_native, open = "ab")
writeBin(as.raw(0x00), native_connection)
close(native_connection)
assert_true(
  !identical(sha256_file(temporary_archive), FASTMNN_SOURCE_SHA256),
  "Source-archive tamper fixture did not change its SHA-256."
)
assert_true(
  !identical(sha256_file(temporary_native), fingerprints$sha256[[native_index]]),
  "Native-library tamper fixture did not change its SHA-256."
)

cat(sprintf(
  "PASS: integrity contract; source, runtime lock, %d installed files and %d core functions verified.\n",
  nrow(fingerprints), length(actual_function_hashes)
))
