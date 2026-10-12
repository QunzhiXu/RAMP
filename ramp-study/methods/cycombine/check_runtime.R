#!/usr/bin/env Rscript
# Read-only environment + real synthetic numerical checks. Never fit a cohort.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2L)
frozen <- normalizePath(args[1], mustWork=TRUE)
lib <- normalizePath(args[2], mustWork=TRUE)
.libPaths(c(lib, .libPaths()))
base <- new.env(parent=globalenv())
source(file.path(frozen, 'adapter', 'cycombine_repro.R'), local=base)
provenance <- base$validate_cycombine_install(lib)
locked <- read.csv(file.path(frozen, 'environment', 'validated_direct_runtime_packages.csv'), stringsAsFactors=FALSE)
actual <- vapply(locked$Package, function(p) utils::packageDescription(p)$Version, character(1))
stopifnot(identical(unname(actual), locked$Version))
stopifnot(identical(digest::digest(file=file.path(frozen,'adapter/vendor/cyCombine-v0.2.5.tar.gz'),algo='sha256'),base$CYCOMBINE_SOURCE_SHA256))
story <- new.env(parent=globalenv())
source(file.path(frozen, 'study_sources/figure_HNSCC/Comparison/scripts/cycombine_repro.R'), local=story)

# All story copies are identical; verify parsed function bodies against the base.
names_base <- ls(base)
names_story <- ls(story)
stopifnot(identical(names_base,names_story))
different <- names_base[!vapply(names_base,function(n) {
  a<-get(n,base); b<-get(n,story)
  if (is.function(a)) identical(formals(a),formals(b)) && identical(body(a),body(b)) else identical(a,b)
},logical(1))]
stopifnot(identical(different,'build_cycombine_input'))

set.seed(20260830)
batch_levels <- c('platform_A','platform_B','platform_C')
parts <- lapply(seq_along(batch_levels),function(i) {
  n<-900L; latent<-sample(1:4,n,replace=TRUE)
  data.frame(cell_id=sprintf('%s_%05d',batch_levels[i],seq_len(n)),
             dataset=batch_levels[i],unit_key=paste0(batch_levels[i],'_',seq_len(n)%%9L),
             CD3=4+latent*0.5+rnorm(n,0,0.6)+i*0.2,
             CD4=3+latent*0.2+rnorm(n,0,0.5)-i*0.1,
             CD8=2+latent*0.4+rnorm(n,0,0.7)+i*0.3,
             CD45=5+rnorm(n,0,0.4)+i*0.15,
             provider_label=letters[latent],outcome=seq_len(n)%%2L,
             stringsAsFactors=FALSE,check.names=FALSE)
})
markers<-c('CD3','CD4','CD8','CD45')
rng_before<-RNGkind()
checks<-list()
for (nb in c(2L,3L)) {
  fixture<-do.call(rbind,parts[seq_len(nb)]); levels<-batch_levels[seq_len(nb)]
  helper<-if(nb==2L) story else base
  fit<-function(x) helper$fit_cycombine_publication(x,markers,'cell_id','dataset','unit_key',levels,required_library=lib)
  first<-suppressMessages(fit(fixture))
  changed<-fixture;changed$provider_label<-rev(changed$provider_label);changed$outcome<-1L-changed$outcome
  second<-suppressMessages(fit(changed))
  stopifnot(identical(first$corrected_expression,second$corrected_expression),identical(first$som_labels,second$som_labels),identical(first$cell_ids,fixture$cell_id))
  input<-helper$build_cycombine_input(fixture,markers,'cell_id','dataset','unit_key',levels)
  direct<-suppressMessages(cyCombine::batch_correct(df=input,label=NULL,xdim=8,ydim=8,rlen=10,
          parametric=TRUE,seed=840,covar=NULL,anchor=NULL,markers=markers,norm_method='rank',ties.method='average'))
  error<-max(abs(as.matrix(first$corrected_expression)-as.matrix(direct[,markers,drop=FALSE])))
  stopifnot(is.finite(error),error==0L)
  if(nb==3L) {
    adapted<-suppressMessages(story$fit_cycombine_publication(fixture,markers,'cell_id','dataset','unit_key',levels,required_library=lib))
    stopifnot(identical(first$corrected_expression,adapted$corrected_expression),identical(first$som_labels,adapted$som_labels))
  } else {
    rejection<-tryCatch({base$build_cycombine_input(fixture,markers,'cell_id','dataset','unit_key',levels);FALSE},error=function(e)TRUE)
    stopifnot(rejection)
  }
  checks[[as.character(nb)]]<-list(batches=nb,cells=nrow(fixture),markers=length(markers),direct_official_max_abs_error=error,
    deterministic_and_label_invariant=TRUE,ids_preserved=TRUE,finite=all(is.finite(as.matrix(first$corrected_expression))))
}
# Direct calls also use the current default RNG kinds; wrapper restores them.
stopifnot(identical(RNGkind(),rng_before))
result<-list(R=R.version.string,loaded_wrapper=file.path(frozen,'adapter/cycombine_repro.R'),
   loaded_two_or_three_wrapper=file.path(frozen,'study_sources/figure_HNSCC/Comparison/scripts/cycombine_repro.R'),
   installation=provenance,packages=as.list(setNames(unname(actual),locked$Package)),
   parsed_function_differences=different,synthetic_checks=checks,sessionInfo=capture.output(sessionInfo()))
cat('\nCYCOMBINE_AUDIT_JSON=',jsonlite::toJSON(result,auto_unbox=TRUE,digits=17,null='null'), '\n',sep='')
