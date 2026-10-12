# Read-only authentication for real execution; no fitting or installation.
args <- commandArgs(TRUE)
stopifnot(length(args)==2L, as.character(getRversion())=='4.3.2')
adapter <- normalizePath(args[1]); expected <- normalizePath(args[2])
.libPaths(c(file.path(adapter,'.Rlib'),.libPaths()))
source(file.path(adapter,'cytonorm_repro.R'))
stopifnot(sha(expected)==sha(file.path(adapter,'.Rlib/runtime_lock.csv')))
validate_runtime(adapter)
lock <- read.csv(expected)
stopifnot(nrow(lock)==69L, !anyDuplicated(lock$package))
versions <- setNames(vapply(lock$package,function(p) as.character(packageVersion(p)),character(1)),lock$package)
stopifnot(identical(unname(versions),lock$version))
cats <- c('LC_COLLATE','LC_CTYPE','LC_MONETARY','LC_NUMERIC','LC_TIME','LC_MESSAGES')
locale <- setNames(vapply(cats,Sys.getlocale,character(1)),cats)
want <- setNames(rep('C.UTF-8',length(cats)),cats);want['LC_NUMERIC'] <- 'C'
stopifnot(identical(locale,want),all(Sys.getenv(c('LC_ALL','LANG'))=='C.UTF-8'))
cat(jsonlite::toJSON(list(R=as.character(getRversion()),locale=as.list(locale),
 environment=as.list(Sys.getenv(c('LC_ALL','LANG'))),runtime_versions=as.list(versions),
 runtime_lock_sha256=sha(expected),upstream_lock_sha256=sha(file.path(adapter,'upstream_lock.json'))),
 auto_unbox=TRUE,digits=17),'\n',sep='')
