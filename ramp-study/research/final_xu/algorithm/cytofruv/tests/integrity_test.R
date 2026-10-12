#!/usr/bin/env Rscript

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
test_path <- normalizePath(sub("^--file=", "", script_arg), mustWork = TRUE)
algorithm_dir <- dirname(dirname(test_path))
Sys.setenv(CYTOFRUV_ALGORITHM_DIR = algorithm_dir)
source(file.path(algorithm_dir, "cytofruv_repro.R"), local = globalenv())

validate_cytofruv_source()
cytofruv_validate_dependency_archives()
activate_cytofruv_runtime()
upstream <- load_official_cytofruv_core()
stopifnot(is.function(upstream$run_RUVIII))
stopifnot(is.function(upstream$fastRUVIII))
stopifnot(is.function(upstream$make_residual_mat_several_rep))
function_lock <- cytofruv_validate_runtime_functions()
stopifnot(nrow(function_lock) == 5L)
stopifnot(file.exists(file.path(algorithm_dir, ".Rlib", "bootstrap_sessionInfo.txt")))

cat("PASS: CytofRUV source archives, runtime versions, paths, and reached-function integrity\n")
