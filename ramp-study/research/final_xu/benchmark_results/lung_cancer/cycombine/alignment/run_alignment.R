#!/usr/bin/env Rscript

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_argument)) stop("Cannot determine script location.", call. = FALSE)
script_path <- sub("^--file=", "", script_argument[[1L]])
script_path <- gsub("~+~", " ", script_path, fixed = TRUE)
script_path <- normalizePath(script_path, mustWork = TRUE)
alignment_dir <- dirname(script_path)
method_dir <- dirname(alignment_dir)
final_xu <- normalizePath(file.path(alignment_dir, "..", "..", "..", ".."), mustWork = TRUE)
runner <- file.path(final_xu, "algorithm", "cycombine", "run_cycombine.R")
input <- file.path(method_dir, "preprocessing", "data", "combined_analysis_ready.csv.gz")
markers <- file.path(method_dir, "preprocessing", "markers.txt")
config <- file.path(alignment_dir, "config.json")
output <- file.path(alignment_dir, "data")

required <- c(runner, input, markers, config)
if (any(!file.exists(required))) {
  stop(sprintf("Missing required file(s): %s", paste(required[!file.exists(required)], collapse = ", ")), call. = FALSE)
}
if (file.exists(output) || dir.exists(output)) {
  stop("alignment/data already exists; the canonical runner will not overwrite a completed run.", call. = FALSE)
}

status <- system2(
  "Rscript",
  c("--vanilla", shQuote(runner), "--input", shQuote(input), "--markers", shQuote(markers),
    "--config", shQuote(config), "--output", shQuote(output))
)
if (!identical(status, 0L)) stop(sprintf("cyCombine runner failed with status %s.", status), call. = FALSE)
cat(sprintf("Saved canonical lung-cancer cyCombine alignment to %s\n", output))

