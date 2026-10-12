args <- commandArgs(trailingOnly=TRUE)
out <- normalizePath(args[1]); method <- args[2]; algroot <- normalizePath(args[3])
helperroot <- normalizePath(args[4])
name <- switch(method,harmony_direct='harmony',cytonorm='CytoNorm',method)
alg <- file.path(algroot,name); lib <- file.path(alg,'.Rlib')
.libPaths(c(lib,.libPaths()))
markers <- readLines(file.path(out,'markers.txt'))
input <- read.csv(gzfile(file.path(out,'input.csv.gz')),check.names=FALSE,
  colClasses=c(rep('character',3),rep('numeric',length(markers))))
stopifnot(identical(names(input),c('cell_id','dataset','unit_key',markers)))
helper <- switch(method,harmony_direct=file.path(alg,'harmony_direct_repro.R'),
  fastmnn=file.path(alg,'fastmnn_repro.R'),
  cycombine=file.path(helperroot,'vendor/cycombine_repro.R'),
  cytonorm=file.path(helperroot,'vendor/cytonorm_repro.R'))
source(helper)
if(method=='cytonorm') {
  validate_runtime(alg)
  cfg <- jsonlite::read_json(file.path(alg,'config.json')); cfg$cases <- NULL
  result <- fit_cytonorm(input,markers,cfg,out)
  jsonlite::write_json(list(parameters=cfg,result=result),file.path(out,'runtime.json'),auto_unbox=TRUE,pretty=TRUE)
} else {
  fn <- switch(method,harmony_direct=fit_harmony_direct,fastmnn=fit_fastmnn_official,cycombine=fit_cycombine_publication)
  fit <- fn(input,markers,'cell_id','dataset','unit_key',unique(input$dataset),required_library=lib)
  y <- switch(method,harmony_direct=fit$adjusted_marker_coordinates,fastmnn=fit$corrected_embedding,cycombine=fit$corrected_expression)
  y <- as.matrix(y)
  stopifnot(nrow(y)==nrow(input),all(is.finite(y)))
  if(method=='fastmnn') colnames(y)<-paste0('MNN',seq_len(ncol(y))) else colnames(y)<-markers
  data.table::fwrite(data.frame(cell_id=input$cell_id,y,check.names=FALSE),file.path(out,'corrected_expression.csv.gz'),quote=TRUE)
  saveRDS(fit,file.path(out,'fit.rds'))
  jsonlite::write_json(if(method=='harmony_direct') fit$resolved_parameters else fit$parameters,
    file.path(out,'runtime.json'),auto_unbox=TRUE,pretty=TRUE)
}
writeLines(capture.output(sessionInfo()),file.path(out,'sessionInfo.txt'))
