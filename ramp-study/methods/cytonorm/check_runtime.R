# Synthetic-only direct official-call comparison. All writes under private scratch.
args <- commandArgs(TRUE)
adapter <- normalizePath(args[1]); story_path <- normalizePath(args[2]); output <- normalizePath(args[3])
expected_lock <- normalizePath(args[4])
.libPaths(c(file.path(adapter, '.Rlib'), .libPaths()))
stopifnot(as.character(getRversion()) == '4.3.2')
locale <- sapply(c('LC_COLLATE','LC_CTYPE','LC_MONETARY','LC_NUMERIC','LC_TIME','LC_MESSAGES'), Sys.getlocale)
stopifnot(Sys.getenv('LC_ALL')=='C.UTF-8', Sys.getenv('LANG')=='C.UTF-8', locale[['LC_NUMERIC']]=='C',
          all(locale[names(locale)!='LC_NUMERIC']=='C.UTF-8'))
tech <- new.env();sys.source(file.path(adapter,'cytonorm_repro.R'), tech)
story <- new.env();sys.source(story_path, story)
lock_hash <- tech$sha(expected_lock)
stopifnot(tech$sha(file.path(adapter,'.Rlib/runtime_lock.csv'))==lock_hash)
tech$validate_runtime(adapter)
lock <- read.csv(file.path(adapter,'.Rlib/runtime_lock.csv'))
stopifnot(nrow(lock)>0, all(vapply(seq_len(nrow(lock)), function(i)
  as.character(packageVersion(lock$package[i]))==lock$version[i], logical(1))))
cfg <- jsonlite::read_json(file.path(adapter,'config.json'));cfg$cases<-NULL
# Apart from the explicit batch-count guard, helper source is unchanged.
old <- readLines(file.path(adapter,'cytonorm_repro.R'));new <- readLines(story_path)
changed <- which(old!=new)
stopifnot(length(old)==length(new),length(changed)==1L,
  grepl('Expected exactly three',old[changed],fixed=TRUE),
  grepl('Expected two or three',new[changed],fixed=TRUE))
# Independently verify installed public CytoNorm function bodies against its source archive.
unpacked <- file.path(getwd(),'upstream');dir.create(unpacked)
untar(file.path(adapter,'vendor/CytoNorm_2.0.6.tar.gz'),exdir=unpacked)
definitions <- list()
package_root <- list.dirs(unpacked,recursive=FALSE,full.names=TRUE)
stopifnot(length(package_root)==1L,dir.exists(file.path(package_root,'R')))
for (p in list.files(file.path(package_root,'R'),full.names=TRUE,pattern='\\.[rR]$')) {
  for (node in parse(p,keep.source=FALSE)) {
    if (is.call(node) && as.character(node[[1]]) %in% c('<-','=') && length(node)==3L &&
        is.symbol(node[[2]]) && is.call(node[[3]]) && identical(node[[3]][[1]],as.name('function'))) {
      definitions[[as.character(node[[2]])]] <- eval(node[[3]])
    }
  }
}
checked <- c('CytoNorm.train','CytoNorm.normalize','QuantileNorm.train','QuantileNorm.normalize')
for (name in checked) {
  fun <- get(name,asNamespace('CytoNorm'));original <- definitions[[name]]
  stopifnot(is.function(original),identical(formals(fun),formals(original)),identical(body(fun),body(original)))
}

frame <- function(x) {
  ff<-flowCore::flowFrame(x)
  ff@parameters@data$minRange<-rep(-1e30,ncol(x));ff@parameters@data$maxRange<-rep(1e30,ncol(x))
  ff@parameters@data$range<-rep(1e30,ncol(x));ff
}
official <- function(input,markers,config,dest) {
  dir.create(dest);batches<-unique(input$dataset);x<-as.matrix(input[markers])
  RNGkind('Mersenne-Twister','Inversion','Rejection');set.seed(config$seed)
  selected<-unlist(lapply(unique(input$unit_key),function(u) {
    rows<-which(input$unit_key==u);sort(rows[sample.int(length(rows),min(length(rows),config$aggregate_cells_per_unit))])
  }),use.names=FALSE)
  files<-file.path(dest,sprintf('aggregate_%02d.fcs',seq_along(batches)))
  for(i in seq_along(batches))flowCore::write.FCS(frame(x[selected[input$dataset[selected]==batches[i]],,drop=FALSE]),files[i],what='double')
  fs<-config$flowsom;fs$colsToUse<-markers;fs$compensate<-FALSE;fs$transform<-FALSE
  train<-file.path(dest,'train');dir.create(train)
  model<-CytoNorm::CytoNorm.train(files=files,labels=batches,channels=markers,
    transformList=NULL,outputDir=train,FlowSOM.params=fs,
    normParams=c(config$quantile,list(transformation=FALSE,truncate_max_range=FALSE,min.limit=-Inf)),
    seed=config$seed,recompute=TRUE,clean=TRUE,plot=FALSE,verbose=TRUE,
    transformation=FALSE,truncate_max_range=FALSE,min.limit=-Inf)
  frames<-lapply(batches,function(b) {rows<-which(input$dataset==b);frame(cbind(x[rows,,drop=FALSE],.row_id=as.double(rows)))})
  set<-flowCore::flowSet(frames);flowCore::sampleNames(set)<-sprintf('batch_%02d',seq_along(batches))
  norm<-function(...)CytoNorm::QuantileNorm.normalize(...,transformation=FALSE,truncate_max_range=FALSE,min.limit=-Inf)
  out<-CytoNorm::CytoNorm.normalize(model,set,labels=batches,transformList=NULL,transformList.reverse=NULL,
    outputDir=file.path(dest,'normalize'),normMethod.normalize=norm,clean=TRUE,write=FALSE,verbose=TRUE,
    transformation=FALSE,truncate_max_range=FALSE,min.limit=-Inf)
  y<-do.call(rbind,lapply(seq_along(batches),function(i)flowCore::exprs(out[[i]])))
  stopifnot(!anyDuplicated(y[,'.row_id']),setequal(y[,'.row_id'],seq_len(nrow(input))))
  list(values=y[match(seq_len(nrow(input)),y[,'.row_id']),markers,drop=FALSE],selected=input$cell_id[selected])
}
results<-list()
for(nbatch in c(2L,3L)) {
  set.seed(20261011);n<-2200L*nbatch;markers<-paste0('M',1:6)
  group<-rep(seq_len(nbatch),each=2200L);state<-rep(1:5,length.out=n)
  x<-matrix(rnorm(n*6,0,.7),n,6)+state*1.4+group*.4;x[1,]<-0;colnames(x)<-markers
  input<-data.frame(cell_id=paste0('cell',seq_len(n)),dataset=paste0('platform',group),
    unit_key=paste0('unit',group),x,check.names=FALSE)
  run<-file.path(getwd(),paste0('wrapper',nbatch));dir.create(run)
  helper<-if(nbatch==3L)tech else story
  result<-helper$fit_cytonorm(input,markers,cfg,run)
  ref<-official(input,markers,cfg,file.path(getwd(),paste0('official',nbatch)))
  actual<-as.matrix(data.table::fread(file.path(run,'corrected_expression.csv.gz'))[,..markers])
  selected<-data.table::fread(file.path(run,'aggregate_cells.csv.gz'))$cell_id
  # Compare the exact same public CSV representation, and independently check
  # full binary values before CSV serialization using the wrapper's saved model.
  direct_csv<-file.path(getwd(),paste0('official',nbatch,'.csv.gz'))
  data.table::fwrite(data.frame(cell_id=input$cell_id,ref$values,check.names=FALSE),direct_csv,quote=TRUE,na='NA')
  direct_parsed<-as.matrix(data.table::fread(direct_csv)[,..markers])
  model<-readRDS(file.path(run,'model.rds'));cl<-data.table::fread(file.path(run,'cell_clusters.csv.gz'))$cluster
  f32<-function(z)matrix(readBin(writeBin(as.double(z),raw(),size=4L),'double',n=length(z),size=4L),nrow(z),ncol(z),dimnames=dimnames(z))
  before<-f32(x);replayed<-before
  for(c in sort(unique(cl)))for(b in unique(input$dataset))for(m in markers) {
    rows<-which(cl==c & input$dataset==b)
    replayed[rows,m]<-model$clusterRes[[as.character(c)]]$splines[[b]][[m]](before[rows,m])
  }
  replayed<-f32(replayed)
  stopifnot(identical(selected,ref$selected),nrow(actual)==n,all(is.finite(actual)),
    identical(unname(actual),unname(direct_parsed)),identical(unname(replayed),unname(ref$values)),result$spline_replay_max_absolute_error==0,
    result$source_labels_used==FALSE,result$aggregate_cells==2000L*nbatch)
  results[[length(results)+1L]]<-list(batches=nbatch,cells=n,markers=length(markers),
    aggregate_cells=result$aggregate_cells,maximum_absolute_difference=max(abs(actual-direct_parsed)),
    preserialization_model_replay_maximum_difference=max(abs(replayed-ref$values)),
    csv_serialization_rounding=max(abs(actual-ref$values)),
    spline_replay_max_absolute_error=result$spline_replay_max_absolute_error,
    selected_ids_identical=TRUE,input_negative_values_preserved=any(x<0),zero_cell_retained=TRUE,
    helper=if(nbatch==3L)'technical' else 'story')
}
# Execute the frozen CLI only with a deliberately corrupt hash: it must reject before fitting.
data.table::fwrite(input,file.path(getwd(),'input.csv'));writeLines(markers,file.path(getwd(),'markers.txt'))
bad<-cfg;bad$input_sha256<-paste(rep('0',64),collapse='');bad$markers_sha256<-tech$sha(file.path(getwd(),'markers.txt'))
tech$write_json(bad,file.path(getwd(),'bad_config.json'))
stdout<-file.path(output,'negative_cli.stdout.log');stderr<-file.path(output,'negative_cli.stderr.log')
status<-system2(file.path(R.home('bin'),'Rscript'),c('--vanilla',shQuote(file.path(adapter,'run_cytonorm.R')),
  '--input',shQuote(file.path(getwd(),'input.csv')),'--markers',shQuote(file.path(getwd(),'markers.txt')),
  '--config',shQuote(file.path(getwd(),'bad_config.json')),'--output',shQuote(file.path(getwd(),'must_not_exist'))),stdout=stdout,stderr=stderr)
stopifnot(status!=0L,any(grepl('Prepared input hash mismatch',readLines(stderr),fixed=TRUE)),!file.exists(file.path(getwd(),'must_not_exist')))
tech$validate_runtime(adapter)
stopifnot(tech$sha(file.path(adapter,'.Rlib/runtime_lock.csv'))==lock_hash,tech$sha(expected_lock)==lock_hash)
tech$write_json(list(status='passed',real_cohort_fit=FALSE,parameters=cfg,locale=as.list(locale),
  runtime_lock_sha256=lock_hash,
  runtime_versions=setNames(as.list(lock$version),lock$package),upstream_functions_identical=checked,
  adapter_helper_sha256=tech$sha(file.path(adapter,'cytonorm_repro.R')),
  story_helper_sha256=tech$sha(story_path),story_only_batch_guard_changed=TRUE,
  corrupt_hash_rejected_before_fit=TRUE,results=results),file.path(output,'synthetic.json'))
writeLines(capture.output(sessionInfo()),file.path(output,'sessionInfo.txt'))
cat('PASS direct official CytoNorm parity for two and three batches\n')
