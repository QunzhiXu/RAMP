args <- commandArgs(trailingOnly=TRUE)
here <- normalizePath(args[1]); method <- args[2]
root <- dirname(dirname(here))
algname <- c(harmony='harmony',fastmnn='fastmnn',cycombine='cycombine',cytonorm='CytoNorm')[[method]]
alg <- file.path(root,'algorithm',algname); lib <- file.path(alg,'.Rlib')
.libPaths(c(lib,.libPaths()))
config <- jsonlite::read_json(file.path(here,'run_config.json'),simplifyVector=TRUE)
markers <- readLines(file.path(here,'input','markers.txt'))
inputfile <- file.path(here,'input',if(method=='harmony') 'raw.csv.gz' else 'analysis_ready.csv.gz')
input <- read.csv(gzfile(inputfile),check.names=FALSE,stringsAsFactors=FALSE,colClasses=c(rep('character',3),rep('numeric',length(markers))))
out <- file.path(here,'methods',method)
if (file.exists(file.path(out,'R_COMPLETE.json'))) stop('Already completed; do not overwrite')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
helper <- switch(method,harmony=file.path(alg,'harmony_direct_repro.R'),fastmnn=file.path(alg,'fastmnn_repro.R'),cycombine=file.path(here,'scripts','cycombine_repro.R'),cytonorm=file.path(here,'scripts','cytonorm_repro.R'))
source(helper)
start <- Sys.time()
if(method=='cytonorm') {
 validate_runtime(alg)
 cfg <- jsonlite::read_json(file.path(alg,'config.json')); cfg$cases <- NULL
 result <- fit_cytonorm(input,markers,cfg,out)
 parameters <- cfg
} else {
 fun <- switch(method,harmony=fit_harmony_direct,fastmnn=fit_fastmnn_official,cycombine=fit_cycombine_publication)
 fit <- fun(input,markers,'cell_id','dataset','unit_key',config$datasets,required_library=lib)
 saveRDS(fit,file.path(out,'fit.rds'),compress=TRUE)
 y <- switch(method,harmony=fit$adjusted_marker_coordinates,fastmnn=fit$corrected_embedding,cycombine=fit$corrected_expression)
 y <- as.matrix(y); stopifnot(nrow(y)==nrow(input),all(is.finite(y)))
 if(method=='fastmnn') colnames(y) <- paste0('MNN',seq_len(ncol(y))) else colnames(y) <- markers
 data.table::fwrite(data.frame(cell_id=input$cell_id,y,check.names=FALSE),file.path(out,'corrected_expression.csv.gz'),quote=TRUE)
 for(n in names(fit)) if(is.data.frame(fit[[n]]) && !n %in% c('corrected_expression','input_marker_coordinates','adjusted_marker_coordinates')) data.table::fwrite(fit[[n]],file.path(out,paste0(n,'.csv.gz')),quote=TRUE)
 parameters <- if(method=='harmony') fit$resolved_parameters else fit$parameters
 result <- list(cells=nrow(y),dimensions=ncol(y),row_identity_verified=TRUE,output_type=if(method=='fastmnn') 'embedding' else 'marker_values')
}
writeLines(capture.output(sessionInfo()),file.path(out,'sessionInfo.txt'))
jsonlite::write_json(list(method=method,input_sha256=digest::digest(file=inputfile,algo='sha256'),helper_sha256=digest::digest(file=helper,algo='sha256'),parameters=parameters,result=result,elapsed_seconds=as.numeric(difftime(Sys.time(),start,units='secs')),output_sha256=digest::digest(file=file.path(out,'corrected_expression.csv.gz'),algo='sha256')),file.path(out,'R_COMPLETE.json'),auto_unbox=TRUE,pretty=TRUE,digits=17,null='null')
cat('COMPLETE',method,'\n')
