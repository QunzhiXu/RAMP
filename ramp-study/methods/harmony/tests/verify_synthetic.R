# Compare the new entry's output to an independent direct call on identical parsed input.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 7L)
adapter <- args[[1]]; kind <- args[[2]]; input_path <- args[[3]]
markers <- readLines(args[[4]]); result <- args[[5]]; config_path <- args[[6]]; out_report <- args[[7]]
.libPaths(c(file.path(adapter, '.Rlib'), .libPaths()))
source(file.path(adapter, 'harmony_direct_repro.R'))
config <- jsonlite::read_json(config_path, simplifyVector=TRUE)
if (kind == 'technical') {
  input <- read_analysis_table(input_path, character_columns=c('cell_id','dataset','unit_key'))
  datasets <- config$batch_levels
  actual <- readRDS(file.path(result, 'harmony_adjusted_marker_coordinates.rds'))
} else {
  input <- read.csv(gzfile(input_path), check.names=FALSE, stringsAsFactors=FALSE,
                    colClasses=c(rep('character',3),rep('numeric',length(markers))))
  datasets <- config$datasets
  actual <- readRDS(file.path(result, 'fit.rds'))$adjusted_marker_coordinates
}
fit <- fit_harmony_direct(input, markers, 'cell_id', 'dataset', 'unit_key', datasets,
                          required_library=file.path(adapter, '.Rlib'))
stopifnot(identical(actual, fit$adjusted_marker_coordinates))
jsonlite::write_json(list(kind=kind, cells=nrow(input), platforms=length(datasets),
    markers=length(markers), direct_wrapper_identical=TRUE,
    maximum_absolute_difference=max(abs(actual-fit$adjusted_marker_coordinates))),
    out_report, pretty=TRUE, auto_unbox=TRUE)
