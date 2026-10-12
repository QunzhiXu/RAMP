suppressPackageStartupMessages({
  library(SingleCellExperiment); library(data.table); library(jsonlite); library(digest)
})
root <- normalizePath(commandArgs(TRUE)[1])
setwd(root)
stopifnot(basename(root) == 'caf_fullpanel')
stopifnot(startsWith(normalizePath(tempdir()), paste0(root, '/tmp/')))
config <- fromJSON('provenance/EXPORT_REQUEST.json')
stopifnot(digest(file='data/CELL_REQUEST.csv', algo='sha256', serialize=FALSE) == config$request_sha256)
req <- fread('data/CELL_REQUEST.csv', colClasses=list(character=c('patient_id', 'roi_id')))
stopifnot(nrow(req) == 5206, sum(req$provider_cell_type == 'vCAF') == 646)
cat('Reading the original annotated Cords SCE read-only.\n'); flush.console()
sce <- readRDS(config$provider_sce)
cat('SCE dimensions:', dim(sce), '\nAssays:', assayNames(sce), '\n'); flush.console()
rows <- as.integer(req$source_row_id)
stopifnot(all(rows >= 1), all(rows <= ncol(sce)), !anyDuplicated(rows))
fields <- c('Patient_ID', 'RoiID', 'Center_X', 'Center_Y', 'cell_category', 'cell_type', 'cell_subtype')
cd <- as.data.frame(colData(sce)[rows, fields, drop=FALSE])
stopifnot(all(as.character(cd$Patient_ID) == req$patient_id),
          all(as.character(cd$RoiID) == req$roi_id),
          all(as.character(cd$cell_type) == req$provider_cell_type),
          all(as.character(cd$cell_subtype) == req$cell_subtype),
          all(as.character(cd$cell_category) == req$cell_category))
coord_error <- max(abs(cd$Center_X - req$x), abs(cd$Center_Y - req$y))
stopifnot(coord_error < 1e-8)
markers <- rownames(sce)
stopifnot(length(markers) == 43, all(c('CD146', 'CD34') %in% markers))
fwrite(data.table(provider_marker=markers,
                  used_as_protein=!(markers %in% c('Iridium_191', 'Iridium_193'))),
       'data/ALL_MARKERS.csv')
records <- list()
for (name in assayNames(sce)) {
  values <- t(assay(sce, name)[, rows, drop=FALSE])
  storage.mode(values) <- 'double'
  stopifnot(identical(dim(values), c(5206L, 43L)), all(is.finite(values)))
  colnames(values) <- markers
  file <- paste0('data/ASSAY_', gsub('[^A-Za-z0-9_]', '_', name), '.csv.gz')
  fwrite(cbind(req[, .(cell_key, source_row_id)], as.data.table(values)), file, compress='gzip')
  records[[name]] <- list(file=file, min=min(values), max=max(values),
                         sha256=digest(file=file, algo='sha256', serialize=FALSE))
  cat('Exported', name, 'range', range(values), '\n'); flush.console()
}
cat('Verifying source checksum.\n'); flush.console()
observed <- digest(file=config$provider_sce, algo='sha256', serialize=FALSE)
stopifnot(observed == config$provider_sce_sha256)
write_json(list(status='PASS', n_cells=nrow(req), n_channels=43,
                n_protein_channels=41, n_vCAF=sum(req$provider_cell_type == 'vCAF'),
                all_ids_labels_coordinates_match=TRUE, coordinate_max_error=coord_error,
                source_row_indexing='1-based columns in the provider SCE',
                source_sha256=observed, assays=records),
           'provenance/EXPORT_VERIFICATION.json', pretty=TRUE, auto_unbox=TRUE)
writeLines(capture.output(sessionInfo()), 'provenance/R_SESSION.txt')
cat('Full-panel export complete.\n'); flush.console()
