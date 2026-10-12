# Check the additional story dispatcher dependency and the frozen helper variant.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2L)
adapter <- normalizePath(args[1],mustWork=TRUE)
lib <- file.path(adapter,'.Rlib')
.libPaths(c(lib,.libPaths()))
base <- new.env(parent=globalenv()); source(file.path(adapter,'cycombine_repro.R'),local=base)
story <- new.env(parent=globalenv()); source(args[2],local=story)
provenance <- story$validate_cycombine_install(lib)
stopifnot(identical(ls(base),ls(story)))
different <- ls(base)[!vapply(ls(base),function(n) {
  a<-get(n,base);b<-get(n,story)
  if(is.function(a)) identical(formals(a),formals(b)) && identical(body(a),body(b)) else identical(a,b)
},logical(1))]
stopifnot(identical(different,'build_cycombine_input'))
stopifnot(identical(utils::packageDescription('data.table')$Version,'1.17.8'))
cat(jsonlite::toJSON(list(data_table_version=utils::packageDescription('data.table')$Version,
  data_table_description_sha256=digest::digest(file=system.file('DESCRIPTION',package='data.table'),algo='sha256'),
  helper_sha256=digest::digest(file=args[2],algo='sha256'),
  parsed_helper_difference=different,installation=provenance),auto_unbox=TRUE,digits=NA),'\n')
