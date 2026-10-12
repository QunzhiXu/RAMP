#!/usr/bin/env Rscript
# No fit and no installation. Validate the explicit library and frozen versions.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2L)
adapter <- normalizePath(args[1], mustWork=TRUE)
lib <- normalizePath(file.path(adapter,'.Rlib'), mustWork=TRUE)
.libPaths(c(lib,.libPaths()))
source(file.path(adapter,'cycombine_repro.R'))
provenance <- validate_cycombine_install(lib)
locked <- read.csv(args[2], stringsAsFactors=FALSE)
actual <- vapply(locked$Package, function(p) utils::packageDescription(p)$Version, character(1))
stopifnot(identical(unname(actual),locked$Version))
stopifnot(identical(paste(R.version$major,R.version$minor,sep='.'),'4.3.2'))
descriptions <- vapply(locked$Package,function(p) {
  digest::digest(file=system.file('DESCRIPTION',package=p),algo='sha256',serialize=FALSE)
},character(1))
cat(jsonlite::toJSON(list(R=R.version.string, installation=provenance,
    packages=as.list(setNames(unname(actual),locked$Package)),
    description_sha256=as.list(descriptions)),auto_unbox=TRUE,digits=NA),'\n')
