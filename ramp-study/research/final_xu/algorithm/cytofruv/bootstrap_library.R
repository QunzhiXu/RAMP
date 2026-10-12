#!/usr/bin/env Rscript

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1L) stop("Unable to resolve bootstrap_library.R path")
algorithm_dir <- dirname(normalizePath(sub("^--file=", "", script_arg), mustWork = TRUE))
project_library <- file.path(algorithm_dir, ".Rlib")
vendor_dir <- file.path(algorithm_dir, "vendor")

if (!requireNamespace("digest", quietly = TRUE)) {
  stop("The bootstrap requires digest in the host R library")
}

sha256_file <- function(path) {
  digest::digest(file = path, algo = "sha256", serialize = FALSE)
}

archives <- data.frame(
  package = c("ruv", "rsvd"),
  version = c("0.9.7.1", "1.0.5"),
  filename = c("ruv_0.9.7.1.tar.gz", "rsvd_1.0.5.tar.gz"),
  sha256 = c(
    "a0c54e56ba3d8f6ae178ae4d0e417a79295abf5dcb68bbae26c4b874734d98d8",
    "e40686b869acd4f71fdb1e8e7a6c64cd6792fc9d52a78f9e559a7176ab84e21e"
  ),
  stringsAsFactors = FALSE
)

for (i in seq_len(nrow(archives))) {
  archive <- file.path(vendor_dir, archives$filename[[i]])
  if (!file.exists(archive) || !identical(sha256_file(archive), archives$sha256[[i]])) {
    stop("Vendored archive SHA-256 mismatch: ", archives$filename[[i]])
  }
}

# Build both direct numerical dependencies from the locked source archives in a
# fresh library. Nothing derived from a prior installation is re-signed.
build_library <- tempfile(".Rlib.bootstrap-", tmpdir = algorithm_dir)
if (!dir.create(build_library, recursive = FALSE)) stop("Unable to create bootstrap library")
committed <- FALSE
on.exit({
  if (!committed && dir.exists(build_library)) {
    unlink(build_library, recursive = TRUE, force = TRUE)
  }
}, add = TRUE)

host_libraries <- .libPaths()
.libPaths(c(build_library, host_libraries))
for (i in seq_len(nrow(archives))) {
  install.packages(
    file.path(vendor_dir, archives$filename[[i]]),
    repos = NULL,
    type = "source",
    lib = build_library,
    dependencies = FALSE,
    quiet = FALSE
  )
  observed <- as.character(utils::packageVersion(archives$package[[i]], lib.loc = build_library))
  if (!identical(observed, archives$version[[i]])) {
    stop("Installed version mismatch for ", archives$package[[i]])
  }
}

function_lock <- read.csv(
  file.path(algorithm_dir, "runtime_function_lock.csv"),
  stringsAsFactors = FALSE,
  check.names = FALSE
)
if (!identical(names(function_lock), c("package", "symbol", "sha256")) ||
    !nrow(function_lock) || anyNA(function_lock) ||
    anyDuplicated(paste(function_lock$package, function_lock$symbol))) {
  stop("Invalid runtime_function_lock.csv")
}
function_sha256 <- function(fun) {
  canonical <- paste(deparse(fun, width.cutoff = 500L, control = "all"), collapse = "\n")
  digest::digest(canonical, algo = "sha256", serialize = FALSE)
}
for (i in seq_len(nrow(function_lock))) {
  namespace <- loadNamespace(function_lock$package[[i]], lib.loc = build_library)
  symbol <- function_lock$symbol[[i]]
  if (!exists(symbol, envir = namespace, inherits = FALSE)) {
    stop("Missing runtime function: ", function_lock$package[[i]], ":::", symbol)
  }
  observed <- function_sha256(get(symbol, envir = namespace, inherits = FALSE))
  if (!identical(observed, function_lock$sha256[[i]])) {
    stop("Runtime function SHA-256 mismatch: ", function_lock$package[[i]], ":::", symbol)
  }
}

writeLines(capture.output(sessionInfo()), file.path(build_library, "bootstrap_sessionInfo.txt"), useBytes = TRUE)

backup_library <- tempfile(".Rlib.backup-", tmpdir = algorithm_dir)
had_previous <- dir.exists(project_library)
if (had_previous && !file.rename(project_library, backup_library)) {
  stop("Unable to move the previous project library aside")
}
if (!file.rename(build_library, project_library)) {
  if (had_previous) file.rename(backup_library, project_library)
  stop("Unable to activate the freshly built project library")
}
committed <- TRUE
if (had_previous && dir.exists(backup_library)) {
  unlink(backup_library, recursive = TRUE, force = TRUE)
}

cat("PASS: CytofRUV runtime rebuilt from locked sources and function-verified\n")
