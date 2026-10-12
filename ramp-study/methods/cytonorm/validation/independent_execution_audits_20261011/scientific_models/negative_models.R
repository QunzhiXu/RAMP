.libPaths(c('/Users/qxu6/Documents/New project/final_xu/algorithm/CytoNorm/.Rlib',.libPaths()))
source('strict_model_compare.R')
old<-readRDS('/Users/qxu6/Documents/New project/final_xu/benchmark_results/bone_marrow/cytonorm/alignment/data/model.rds')
new<-readRDS('/Users/qxu6/Documents/New project/ramp-study/methods/cytonorm/runs/seven_fit_20261011/bone_marrow/runtime/tree/algorithm_test/cytonorm_execution/bone_marrow/result/model.rds')
copy<-function(x)unserialize(serialize(x,NULL))
fn<-function(b)b$clusterRes[['1']]$splines[['BM CODEX']][['CD117']]
setfn<-function(b,f){b$clusterRes[['1']]$splines[['BM CODEX']][['CD117']]<-f;b}
tests<-list(
  SOM_codebook=function(b){stopifnot(is.matrix(b$fsom$map$codes));b$fsom$map$codes[1,1]<-b$fsom$map$codes[1,1]+1e-12;b},
  reference_quantile=function(b){b$clusterRes[['1']]$refQuantiles[1,1]<-b$clusterRes[['1']]$refQuantiles[1,1]+1e-12;b},
  embedded_input_numeric=function(b){b$fsom$info$parameters$input@exprs[1,1]<-b$fsom$info$parameters$input@exprs[1,1]+1e-12;b},
  call_seed=function(b){b$fsom$info$parameters$seed<-43;b},
  call_extra_parameter=function(b){b$fsom$info$parameters$unknown<-1;b},
  unexpected_model_metadata=function(b){b$fsom$info$unexpected<-1;b},
  FCS_span=function(b){d<-b$fsom$info$parameters$input@description;d[['$ENDDATA']]<-as.character(as.double(d[['$ENDDATA']])+1);b$fsom$info$parameters$input@description<-d;b},
  FCS_count=function(b){d<-b$fsom$info$parameters$input@description;d[['$TOT']]<-as.character(as.double(d[['$TOT']])+1);b$fsom$info$parameters$input@description<-d;b},
  FCS_GUID_pair=function(b){b$fsom$info$parameters$input@description$GUID<-'fileabcdef';b},
  invalid_timestamp=function(b){b$fsom$info$date<-'not-a-date';b},
  graph_extra_binding=function(b){e<-unclass(b$fsom$MST$graph)[[10]];e$unexpected<-1;b},
  graph_bad_UUID=function(b){e<-unclass(b$fsom$MST$graph)[[10]];e$myid<-'not-a-uuid';b},
  spline_captured_numeric=function(b){e<-environment(fn(b));z<-get('x0',e,inherits=FALSE);z[1]<-z[1]+1e-12;assign('x0',z,e);b},
  spline_extra_binding=function(b){assign('unexpected',1,environment(fn(b)));b},
  spline_parent_namespace=function(b){e<-environment(fn(b));parent.env(e)<-baseenv();b},
  spline_function_body=function(b){f<-fn(b);body(f)<-quote({x+1});setfn(b,f)},
  spline_function_formals=function(b){f<-fn(b);formals(f)$deriv<-1L;setfn(b,f)}
)
results<-list()
for(name in names(tests)){
  altered<-tests[[name]](copy(new))
  error<-tryCatch({strict_model_compare(old,altered);NULL},error=function(e)conditionMessage(e))
  stopifnot(is.character(error),length(error)==1L)
  results[[name]]<-list(rejected=TRUE,error=error);cat('PASS rejects',name,'\n')
}
positive<-copy(new)
positive$fsom$info$date<-'2026-10-12 12:34:56.1'
desc<-positive$fsom$info$parameters$input@description
desc$GUID<-'fileabcdef';desc$ORIGINALGUID<-'fileabcdef'
desc[['$BEGINDATA']]<-as.character(as.double(desc[['$BEGINDATA']])+17)
desc[['$ENDDATA']]<-as.character(as.double(desc[['$ENDDATA']])+17)
positive$fsom$info$parameters$input@description<-desc
e<-unclass(positive$fsom$MST$graph)[[10]];e$myid<-'12345678-1234-1234-1234-123456789abc'
stopifnot(strict_model_compare(old,positive)$status=='passed')
jsonlite::write_json(list(status='passed',negative_cases=results,valid_metadata_only_change_accepted=TRUE),
 'negative_models.json',pretty=TRUE,auto_unbox=TRUE)
cat('PASS valid metadata-only changes accepted\n')
