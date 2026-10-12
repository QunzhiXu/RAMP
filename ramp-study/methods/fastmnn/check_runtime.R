# Only synthetic data. The historical helper and independent official calls agree.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2L)
adapter <- normalizePath(args[[1]],mustWork=TRUE)
output <- normalizePath(args[[2]],mustWork=TRUE)
lib <- file.path(adapter,'.Rlib');.libPaths(c(lib,.libPaths()))
cats <- c('LC_COLLATE','LC_CTYPE','LC_MONETARY','LC_NUMERIC','LC_TIME','LC_MESSAGES')
locale <- setNames(vapply(cats,Sys.getlocale,character(1)),cats)
expected <- setNames(rep('C.UTF-8',length(cats)),cats);expected['LC_NUMERIC'] <- 'C'
stopifnot(identical(locale,expected),identical(unname(Sys.getenv(c('LC_ALL','LANG'))),c('C.UTF-8','C.UTF-8')))
source(file.path(adapter,'fastmnn_repro.R'))
install <- validate_fastmnn_install(lib)
stopifnot(as.character(packageVersion('data.table'))=='1.17.8')
lock <- read.csv(file.path(adapter,'runtime_lock.csv'),stringsAsFactors=FALSE)
parameters <- canonical_fastmnn_parameters()
results <- list()
for (nbatch in c(2L,3L)) {
  set.seed(729L+nbatch)
  n <- nbatch*60L;p <- 7L;levels <- paste0('platform_',seq_len(nbatch))
  x <- matrix(rexp(n*p),n,p);x[,1:3] <- x[,1:3]+rep(c(0,1),length.out=n)
  x[1,] <- 0;markers <- paste0('marker_',seq_len(p));colnames(x) <- markers
  input <- data.frame(cell_id=sprintf('synthetic_%04d',seq_len(n)),dataset=rep(levels,each=60L),
    unit_key=paste0(rep(levels,each=60L),'_',rep(rep(1:3,each=20L),nbatch)),x,
    unused_label=rep(c('A','B'),length.out=n),check.names=FALSE)
  input <- input[sample(n),,drop=FALSE];rownames(input) <- NULL
  fit <- fit_fastmnn_official(input,markers,'cell_id','dataset','unit_key',levels,required_library=lib)
  expr <- t(as.matrix(input[,markers]));colnames(expr) <- input$cell_id
  RNGkind(parameters$rng_kind,parameters$normal_kind,parameters$sample_kind);set.seed(parameters$seed)
  direct <- suppressWarnings(suppressMessages(batchelor::fastMNN(expr,batch=factor(input$dataset,levels=levels),
    k=parameters$k,prop.k=parameters$prop_k,restrict=NULL,cos.norm=parameters$cos_norm,ndist=parameters$ndist,
    d=parameters$requested_d,deferred=parameters$deferred,weights=parameters$weights,get.variance=parameters$get_variance,
    merge.order=parameters$merge_order,auto.merge=parameters$auto_merge,min.batch.skip=parameters$min_batch_skip,
    subset.row=parameters$subset_row,correct.all=parameters$correct_all,BSPARAM=BiocSingular::IrlbaParam(),
    BNPARAM=BiocNeighbors::KmknnParam(),BPPARAM=BiocParallel::SerialParam())))
  actual <- as.matrix(SingleCellExperiment::reducedDim(direct,'corrected'))[match(input$cell_id,colnames(direct)),,drop=FALSE]
  maxdiff <- max(abs(unname(fit$corrected_embedding)-unname(actual)))
  stopifnot(maxdiff==0,all(is.finite(actual)),ncol(actual)==6L,identical(rownames(fit$corrected_embedding),input$cell_id))
  input$unused_label <- rev(input$unused_label)
  repeated <- fit_fastmnn_official(input,markers,'cell_id','dataset','unit_key',levels,required_library=lib)
  stopifnot(identical(repeated$corrected_embedding,fit$corrected_embedding))
  results[[length(results)+1L]] <- list(batches=nbatch,cells=n,markers=p,dimensions=ncol(actual),
    maximum_absolute_difference=maxdiff,label_perturbation_identical=TRUE,zero_cell_retained=TRUE)
}
target <- file.path(output,'synthetic.json');stopifnot(!file.exists(target))
jsonlite::write_json(list(status='passed',real_cohort_fit=FALSE,adapter_helper_sha256=sha256_file(file.path(adapter,'fastmnn_repro.R')),
  results=results,parameters=parameters,runtime_lock=lock,installed=install,effective_locale=as.list(locale)),
  target,pretty=TRUE,auto_unbox=TRUE,null='null',digits=17)
writeLines(capture.output(sessionInfo()),file.path(output,'sessionInfo.txt'))
cat('PASS two/three-platform direct official-call parity; maximum differences 0\n')
