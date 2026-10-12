# Read-only check; installation is deliberately a separate operation.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 2L)
adapter <- normalizePath(args[[1]], mustWork=TRUE)
lib <- file.path(adapter, '.Rlib')
.libPaths(c(lib, .libPaths()))
# Match all eleven historical sessionInfo records. Check before the adapter's
# locale-sensitive fingerprint sort; never silently accept an unavailable locale.
locale_categories <- c('LC_COLLATE','LC_CTYPE','LC_MONETARY','LC_NUMERIC','LC_TIME','LC_MESSAGES')
effective_locale <- setNames(vapply(locale_categories, Sys.getlocale, character(1)), locale_categories)
locale_environment <- Sys.getenv(c('LC_ALL','LANG'))
cat('R_LOCALE_JSON=', jsonlite::toJSON(list(environment=as.list(locale_environment),
    effective=as.list(effective_locale)), auto_unbox=TRUE), '\n', sep='')
expected_locale <- setNames(rep('C.UTF-8', length(locale_categories)), locale_categories)
expected_locale[['LC_NUMERIC']] <- 'C'  # R's historical decimal-point convention.
if (!identical(unname(locale_environment), c('C.UTF-8','C.UTF-8')) ||
    !identical(effective_locale, expected_locale)) {
  stop('R locale must be C.UTF-8 (LC_NUMERIC=C); launch with LC_ALL=C.UTF-8 and LANG=C.UTF-8. The requested locale must be available.', call.=FALSE)
}
source(file.path(adapter, 'harmony_direct_repro.R'))
validate_harmony_install(required_library=lib)
lock <- validate_runtime_lock_file(adapter, check_versions=TRUE, required_library=lib)
write.table(lock, stdout(), sep=',', row.names=FALSE, quote=TRUE)
if (args[[2]] == 'story') {
  if (!requireNamespace('data.table', quietly=TRUE) ||
      as.character(packageVersion('data.table')) != '1.17.8') {
    stop('Story serialization requires historical data.table 1.17.8', call.=FALSE)
  }
  cat('data.table,1.17.8,story serialization\n')
}
cat('BUILD_ENVIRONMENT\n')
write.table(current_build_environment_record(), stdout(), sep=',', row.names=FALSE, quote=TRUE)
