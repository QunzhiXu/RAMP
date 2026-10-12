# Read-only runtime authentication; no installation or fitting.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2L, args[[2]] %in% c('technical','story'))
adapter <- normalizePath(args[[1]],mustWork=TRUE)
lib <- file.path(adapter,'.Rlib'); .libPaths(c(lib,.libPaths()))
cats <- c('LC_COLLATE','LC_CTYPE','LC_MONETARY','LC_NUMERIC','LC_TIME','LC_MESSAGES')
locale <- setNames(vapply(cats,Sys.getlocale,character(1)),cats)
expected <- setNames(rep('C.UTF-8',length(cats)),cats); expected['LC_NUMERIC'] <- 'C'
stopifnot(identical(locale,expected),identical(unname(Sys.getenv(c('LC_ALL','LANG'))),c('C.UTF-8','C.UTF-8')))
source(file.path(adapter,'fastmnn_repro.R'))
installation <- validate_fastmnn_install(lib)
lock <- validate_runtime_lock(adapter,normalizePath(lib,mustWork=TRUE))
extra <- NULL
if (args[[2]]=='story') {
  stopifnot(as.character(packageVersion('data.table'))=='1.17.8')
  extra <- list(data.table='1.17.8')
}
cat(jsonlite::toJSON(list(locale=as.list(locale),environment=as.list(Sys.getenv(c('LC_ALL','LANG'))),
  installation=installation,runtime_lock=lock,build=current_build_environment_record(),
  extra_serialization=extra),auto_unbox=TRUE,null='null',digits=17), '\n',sep='')
