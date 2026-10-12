#!/usr/bin/env Rscript
# Offline installation of unmodified, hash-locked upstream source archives.
arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)[1L]
here <- dirname(normalizePath(gsub("~+~", " ", sub("^--file=", "", arg), fixed = TRUE)))
lib <- file.path(here, ".Rlib")
dir.create(lib, showWarnings = FALSE)
.libPaths(c(lib, .libPaths()))
# macOS R bundles the runtime libraries even when the compiler's historical
# /opt/gfortran prefix is absent. Only adjust this installation's linker paths.
if (Sys.info()[["sysname"]] == "Darwin" && !dir.exists("/opt/gfortran/lib")) {
  runtime_libs <- file.path(R.home("lib"), c("libgfortran.5.dylib", "libquadmath.0.dylib"))
  if (all(file.exists(runtime_libs))) {
    makevars <- file.path(lib, "Makevars.local")
    writeLines(paste("FLIBS =", paste(shQuote(runtime_libs), collapse = " ")), makevars)
    Sys.setenv(R_MAKEVARS_USER = makevars)
  }
}
for (p in c("jsonlite", "digest")) if (!requireNamespace(p, quietly = TRUE)) stop("Missing bootstrap dependency: ", p)
lock <- jsonlite::read_json(file.path(here, "upstream_lock.json"), simplifyVector = FALSE)
for (entry in lock$archives) {
  p <- file.path(here, "vendor", entry$file)
  stopifnot(identical(digest::digest(file = p, algo = "sha256"), entry$sha256))
}
order <- c("ALL", "colorRamps", "emdist", "polyclip", "tweenr", "Rtsne",
           "ggnewscale", "ggforce", "ConsensusClusterPlus", "FlowSOM", "CytoNorm")
for (package in order) {
  entry <- Filter(function(x) startsWith(x$file, paste0(package, "_")), lock$archives)[[1L]]
  version <- sub("\\.tar\\.gz$", "", sub(paste0("^", package, "_"), "", entry$file))
  description <- file.path(lib, package, "DESCRIPTION")
  if (file.exists(description) && read.dcf(description)[1, "Version"] == version) next
  install.packages(file.path(here, "vendor", entry$file), repos = NULL, type = "source",
                   lib = lib, INSTALL_opts = c("--no-multiarch", "--no-docs", "--no-html"))
  if (!file.exists(description) || read.dcf(description)[1, "Version"] != version) stop("Installation failed: ", package)
}
for (p in c("CytoNorm", "FlowSOM", "flowCore", "jsonlite", "digest", "data.table")) {
  if (!requireNamespace(p, quietly = TRUE)) stop("Cannot load: ", p)
}
pkgs <- sort(loadedNamespaces())
runtime <- data.frame(package = pkgs,
                      version = vapply(pkgs, function(p) as.character(packageVersion(p)), ""),
                      library = vapply(pkgs, function(p) dirname(find.package(p)), ""))
write.csv(runtime, file.path(lib, "runtime_lock.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(lib, "sessionInfo.txt"))
cat("CytoNorm", as.character(packageVersion("CytoNorm")), "installed in", lib, "\n")
