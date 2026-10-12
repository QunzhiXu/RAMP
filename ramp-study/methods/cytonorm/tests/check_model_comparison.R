# Read-only tests on in-memory copies; no refitting or source writes.
args<-commandArgs(TRUE);stopifnot(length(args)==4L)
.libPaths(c(normalizePath(args[4]),.libPaths()))
invisible(suppressPackageStartupMessages(loadNamespace('CytoNorm')))
invisible(suppressPackageStartupMessages(loadNamespace('FlowSOM')))
invisible(suppressPackageStartupMessages(loadNamespace('flowCore')))
invisible(suppressPackageStartupMessages(loadNamespace('igraph')))
source(args[1]);old<-readRDS(args[2]);fresh<-function()readRDS(args[3])
passed<-character()
check<-function(name,mutate,accept=FALSE) {
  x<-fresh();x<-mutate(x)
  result<-try(strict_model_compare(old,x),silent=TRUE)
  if(accept)stopifnot(is.list(result),identical(result$status,'passed'))
  else stopifnot(inherits(result,'try-error'))
  passed<<-c(passed,name)
}
check('real metadata-only difference accepted',identity,TRUE)
check('map coefficient exactness',function(x){x$fsom$map$codes[1,1]<-x$fsom$map$codes[1,1]+1e-10;x})
check('training call parameter',function(x){x$fsom$info$parameters$seed<-43L;x})
check('flowFrame expression',function(x){f<-x$fsom$info$parameters$input;f@exprs[1,1]<-f@exprs[1,1]+1e-10;x$fsom$info$parameters$input<-f;x})
check('FCS span mismatch',function(x){f<-x$fsom$info$parameters$input;f@description[['$ENDDATA']]<-as.character(as.double(f@description[['$ENDDATA']])+1);x$fsom$info$parameters$input<-f;x})
check('FCS mismatched GUID pair',function(x){f<-x$fsom$info$parameters$input;f@description$GUID<-'fileabc';x$fsom$info$parameters$input<-f;x})
check('FCS arbitrary GUID string',function(x){f<-x$fsom$info$parameters$input;f@description$GUID<-f@description$ORIGINALGUID<-'invalid';x$fsom$info$parameters$input<-f;x})
check('invalid graph UUID',function(x){e<-unclass(x$fsom$MST$graph)[[10]];e$myid<-'invalid';x})
check('graph extra binding',function(x){e<-unclass(x$fsom$MST$graph)[[10]];e$unexpected<-1L;x})
check('invalid timestamp',function(x){x$fsom$info$date<-'invalid';x})
check('extra model value',function(x){x$unexpected<-1L;x})
find_spline<-function(x) {
 for(cl in names(x$clusterRes))for(batch in names(x$clusterRes[[cl]]$splines))for(marker in names(x$clusterRes[[cl]]$splines[[batch]])) {
  f<-x$clusterRes[[cl]]$splines[[batch]][[marker]]
  if(identical(parent.env(environment(f)),asNamespace('stats')))return(c(cl,batch,marker))
 };stop('No stats spline')
}
check('captured spline coefficient',function(x){p<-find_spline(x);e<-environment(x$clusterRes[[p[1]]]$splines[[p[2]]][[p[3]]]);e$m[1]<-e$m[1]+1e-10;x})
check('captured extra spline binding',function(x){p<-find_spline(x);e<-environment(x$clusterRes[[p[1]]]$splines[[p[2]]][[p[3]]]);e$extra<-1L;x})
check('spline body',function(x){p<-find_spline(x);body(x$clusterRes[[p[1]]]$splines[[p[2]]][[p[3]]])<-quote(x+1);x})
check('spline namespace',function(x){p<-find_spline(x);e<-environment(x$clusterRes[[p[1]]]$splines[[p[2]]][[p[3]]]);parent.env(e)<-baseenv();x})
cat(jsonlite::toJSON(list(status='passed',tests=length(passed),names=passed),auto_unbox=TRUE),'\n',sep='')
