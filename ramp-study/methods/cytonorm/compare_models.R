# Read-only exact scientific comparison. Explicit metadata exceptions only.
strict_model_compare <- function(old, new) {
  same <- function(a,b) identical(a,b,num.eq=FALSE,single.NA=FALSE,attrib.as.set=FALSE)
  require <- function(ok,where) if(!isTRUE(ok))stop('Model comparison failed: ',where)
  scalar_string <- function(x) is.character(x)&&length(x)==1L&&!is.na(x)&&is.null(attributes(x))
  integer_text <- function(x) scalar_string(x)&&grepl('^[0-9]+$',x)&&is.finite(as.double(x))&&as.double(x)<=2^53
  metadata <- list()
  graph_path <- 'model$fsom$MST$graph[[10]]'
  input_path <- 'model$fsom$info$parameters$input@description'
  exceptions <- c(paste0(graph_path,'::myid'),paste0(graph_path,'::igraph'),
    'model$fsom$info$date',paste0(input_path,'$GUID'),paste0(input_path,'$ORIGINALGUID'),
    paste0(input_path,'$$BEGINDATA'),paste0(input_path,'$$ENDDATA'))
  require(identical(names(old),c('fsom','clusterRes'))&&identical(names(new),names(old)),'top-level schema')

  # Graph cache exceptions are limited to its UUID and external-pointer address.
  # All nine stored graph fields and the public graph interpretation stay exact.
  graph_info <- function(model,label) {
    graph<-model$fsom$MST$graph;g<-unclass(graph)
    require(identical(class(graph),'igraph')&&is.list(g)&&length(g)==10L,paste(label,'igraph schema'))
    e<-g[[10]];want<-c('.__igraph_version__.','igraph','me','myid')
    require(is.environment(e)&&identical(ls(e,all.names=TRUE,sorted=TRUE),want),paste(label,'igraph environment bindings'))
    require(identical(parent.env(e),baseenv()),paste(label,'igraph parent'))
    require(identical(e$me,graph),paste(label,'igraph self reference'))
    require(identical(e$.__igraph_version__.,4L),paste(label,'igraph format version'))
    require(scalar_string(e$myid)&&grepl('^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',e$myid),paste(label,'igraph UUID'))
    require(typeof(e$igraph)=='externalptr',paste(label,'igraph external pointer'))
    list(graph=graph,g=g,e=e)
  }
  ga<-graph_info(old,'old');gb<-graph_info(new,'new')
  require(same(ga$g[1:9],gb$g[1:9]),'all stored graph topology and numerical attributes')
  require(same(attributes(ga$graph),attributes(gb$graph)),'graph attributes')
  require(same(serialize(ga$e$igraph,NULL,version=3L),serialize(gb$e$igraph,NULL,version=3L)),
          'external pointer serialized tag/protected state')
  require(same(igraph::as_edgelist(ga$graph,names=FALSE),igraph::as_edgelist(gb$graph,names=FALSE)),
          'public graph edge interpretation')
  require(same(igraph::vertex_attr(ga$graph),igraph::vertex_attr(gb$graph))&&
          same(igraph::edge_attr(ga$graph),igraph::edge_attr(gb$graph))&&
          same(igraph::graph_attr(ga$graph),igraph::graph_attr(gb$graph)), 'public graph attributes')

  # These keywords describe the temporary first aggregate FCS representation,
  # not the combined/augmented flowFrame expression matrix (which stays exact).
  fcs_info <- function(model,label) {
    call<-model$fsom$info$parameters
    require(is.call(call)&&sum(names(as.list(call))=='input')==1L,paste(label,'FlowSOM call input'))
    frame<-call$input
    require(isS4(frame)&&identical(class(frame)[1L],'flowFrame'),paste(label,'embedded flowFrame'))
    desc<-slot(frame,'description')
    fields<-c('$BEGINDATA','$ENDDATA','GUID','ORIGINALGUID','$TOT','$PAR','$DATATYPE')
    require(is.list(desc)&&all(vapply(fields,function(n)sum(names(desc)==n)==1L,logical(1))),paste(label,'FCS keyword schema'))
    require(scalar_string(desc$GUID)&&grepl('^file[0-9a-f]+$',desc$GUID)&&same(desc$GUID,desc$ORIGINALGUID),paste(label,'temporary FCS GUID pair'))
    for(n in c('$BEGINDATA','$ENDDATA','$TOT','$PAR'))require(integer_text(desc[[n]]),paste(label,n,'integer text'))
    begin<-as.double(desc[['$BEGINDATA']]);end<-as.double(desc[['$ENDDATA']])
    total<-as.double(desc[['$TOT']]);parameters<-as.double(desc[['$PAR']])
    require(begin>=58&&end>=begin&&total>0&&parameters>0&&identical(desc[['$DATATYPE']],'D'),paste(label,'double FCS representation'))
    require(end-begin+1==total*parameters*8,paste(label,'FCS data span TOT*PAR*8'))
    bits<-paste0('$P',seq_len(parameters),'B')
    require(all(vapply(bits,function(n)sum(names(desc)==n)==1L&&identical(desc[[n]],'64'),logical(1))),paste(label,'64-bit FCS channel metadata'))
    list(begin=begin,end=end,span=end-begin+1,total=total,parameters=parameters)
  }
  fa<-fcs_info(old,'old');fb<-fcs_info(new,'new')
  require(same(fa$span,fb$span)&&same(fa$total,fb$total)&&same(fa$parameters,fb$parameters),'unchanged FCS byte span/counts')
  for(label in c('old','new')) {
    d<-get(label)$fsom$info$date
    require(scalar_string(d)&&grepl('^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}(\\.[0-9]{1,6})?$',d)&&
            !is.na(as.POSIXct(d,format='%Y-%m-%d %H:%M:%OS',tz='UTC')),paste(label,'creation timestamp'))
  }

  # Check every closure, even when the two R functions are already identical.
  spline_counts<-c(stats=0L,identity=0L)
  require(same(names(old$clusterRes),names(new$clusterRes)),'cluster names')
  for(cl in names(old$clusterRes)) {
    aa<-old$clusterRes[[cl]]$splines;bb<-new$clusterRes[[cl]]$splines
    require(same(names(aa),names(bb)),paste('spline batches',cl))
    for(batch in names(aa)) {
      require(same(names(aa[[batch]]),names(bb[[batch]])),paste('spline markers',cl,batch))
      for(marker in names(aa[[batch]])) {
        a<-aa[[batch]][[marker]];b<-bb[[batch]][[marker]];path<-paste(cl,batch,marker)
        require(is.function(a)&&is.function(b)&&same(formals(a),formals(b))&&same(body(a),body(b))&&same(attributes(a),attributes(b)),paste('spline function',path))
        ea<-environment(a);eb<-environment(b)
        if(identical(ea,asNamespace('CytoNorm'))||identical(eb,asNamespace('CytoNorm'))) {
          ref<-get('identityFunction',asNamespace('CytoNorm'))
          require(identical(ea,asNamespace('CytoNorm'))&&identical(eb,ea)&&same(formals(a),formals(ref))&&same(body(a),body(ref)),paste('identity spline namespace/body',path))
          spline_counts['identity']<-spline_counts['identity']+1L
        }else {
          require(identical(parent.env(ea),asNamespace('stats'))&&identical(parent.env(eb),asNamespace('stats')),paste('spline parent stats namespace',path))
          require(identical(ls(ea,all.names=TRUE,sorted=TRUE),c('dx','m','x0','y0'))&&identical(ls(eb,all.names=TRUE,sorted=TRUE),c('dx','m','x0','y0')),paste('spline captured-binding schema',path))
          for(n in c('dx','m','x0','y0'))require(same(get(n,ea,inherits=FALSE),get(n,eb,inherits=FALSE)),paste('exact spline captured values',path,n))
          spline_counts['stats']<-spline_counts['stats']+1L
        }
      }
    }
  }

  # Walk everything else with exact type/value/attribute comparisons. Paired
  # environment visitation preserves cycles/sharing; no whole call is dropped.
  visited_a<-list();visited_b<-list();leaves<-0L
  walk<-function(a,b,path) {
    if(path %in% exceptions) {
      require(identical(typeof(a),typeof(b))&&same(attributes(a),attributes(b)),paste(path,'metadata type/attributes'))
      metadata[[path]]<<-if(typeof(a)=='externalptr')list(kind='external pointer address',serialized_state_identical=TRUE)
                         else list(old=a,new=b,changed=!same(a,b))
      return(invisible(NULL))
    }
    if(same(a,b)){leaves<<-leaves+1L;return(invisible(NULL))}
    require(identical(typeof(a),typeof(b))&&same(class(a),class(b)),paste(path,'type/class'))
    if(is.environment(a)) {
      ia<-which(vapply(visited_a,function(e)identical(e,a),logical(1)))
      ib<-which(vapply(visited_b,function(e)identical(e,b),logical(1)))
      if(length(ia)||length(ib)){
        require(length(ia)==1L&&length(ib)==1L&&identical(ia,ib),paste(path,'environment sharing'))
        return(invisible(NULL))
      }
      visited_a[[length(visited_a)+1L]]<<-a;visited_b[[length(visited_b)+1L]]<<-b
      require(same(environmentIsLocked(a),environmentIsLocked(b)),paste(path,'environment lock'))
      an<-ls(a,all.names=TRUE,sorted=TRUE);bn<-ls(b,all.names=TRUE,sorted=TRUE)
      require(same(an,bn),paste(path,'environment names'))
      for(n in an){
        require(same(bindingIsActive(n,a),bindingIsActive(n,b))&&same(bindingIsLocked(n,a),bindingIsLocked(n,b)),paste(path,n,'binding flags'))
        require(!bindingIsActive(n,a),paste(path,n,'active binding not allowed'))
        walk(get(n,a,inherits=FALSE),get(n,b,inherits=FALSE),paste0(path,'::',n))
      }
      walk(parent.env(a),parent.env(b),paste0(path,'::<parent>'))
    }else if(is.function(a)) {
      require(same(formals(a),formals(b))&&same(body(a),body(b))&&same(attributes(a),attributes(b)),paste(path,'function formals/body/attributes'))
      walk(environment(a),environment(b),paste0(path,'::<environment>'))
    }else if(isS4(a)) {
      an<-slotNames(a);bn<-slotNames(b);require(same(an,bn),paste(path,'slot names'))
      for(n in an)walk(slot(a,n),slot(b,n),paste0(path,'@',n))
      walk(attributes(a)[setdiff(names(attributes(a)),an)],attributes(b)[setdiff(names(attributes(b)),bn)],paste0(path,'::<attributes>'))
    }else if(is.list(a)||is.pairlist(a)||is.call(a)||is.expression(a)) {
      aa<-if(is.list(a))unclass(a) else a;bb<-if(is.list(b))unclass(b) else b
      require(length(aa)==length(bb),paste(path,'length'))
      walk(attributes(a),attributes(b),paste0(path,'::<attributes>'))
      if(length(aa))for(i in seq_along(aa)) {
        label<-if(!is.null(names(a))&&nzchar(names(a)[i]))paste0('$',names(a)[i]) else paste0('[[',i,']]')
        walk(aa[[i]],bb[[i]],paste0(path,label))
      }
    }else stop('Exact model value differs: ',path)
  }
  walk(old,new,'model')
  list(status='passed',comparison='exact values/types/attributes; no tolerance',
       spline_closures=as.list(spline_counts),fcs_data_span=fa$span,
       metadata_exceptions=metadata,allowed_metadata_paths=exceptions,
       exact_substructures_checked=leaves)
}


if (sys.nframe()==0L) {
  args<-commandArgs(TRUE)
  stopifnot(length(args)==3L)
  .libPaths(c(normalizePath(args[3],mustWork=TRUE),.libPaths()))
  invisible(suppressPackageStartupMessages(loadNamespace('CytoNorm')))
  invisible(suppressPackageStartupMessages(loadNamespace('FlowSOM')))
  invisible(suppressPackageStartupMessages(loadNamespace('flowCore')))
  invisible(suppressPackageStartupMessages(loadNamespace('igraph')))
  result<-strict_model_compare(readRDS(args[1]),readRDS(args[2]))
  cat(jsonlite::toJSON(result,auto_unbox=TRUE,digits=17,null='null'),'\n',sep='')
}
