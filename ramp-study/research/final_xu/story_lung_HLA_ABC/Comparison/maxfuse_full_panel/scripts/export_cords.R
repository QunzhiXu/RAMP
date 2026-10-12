suppressPackageStartupMessages({library(SingleCellExperiment);library(data.table);library(jsonlite);library(digest)})
a<-commandArgs(TRUE);source<-normalizePath(a[1]);out<-normalizePath(a[2])
req<-fread(file.path(out,'requested_cells.csv'),colClasses=list(character=c('cell_key','patient_id','source_cell_id')))
cat('Loading exact source SCE read-only\n');flush.console()
sce<-readRDS(source)
pos<-as.integer(req$source_row_id)
stopifnot(length(pos)==28000L,!anyDuplicated(pos),min(pos)>=1L,max(pos)<=ncol(sce))
stopifnot(all(as.character(colData(sce)$Patient_ID[pos])==req$patient_id))
features<-rownames(sce)
writeLines(features,file.path(out,'raw_feature_names.txt'))
for (which in c('counts','c_counts_asinh')) {
 x<-as.matrix(assay(sce,which)[,pos,drop=FALSE]);storage.mode(x)<-'double'
 stopifnot(all(is.finite(x)))
 writeBin(as.double(x),file.path(out,paste0(which,'.float64.bin')),size=8,endian='little')
}
fwrite(data.table(cell_id=req$cell_key,source_row_id=pos,source_cell_id=as.character(colnames(sce)[pos]),patient_id=req$patient_id),file.path(out,'source_identity.csv'))
write_json(list(source=source,source_sha256=digest(file=source,algo='sha256',serialize=FALSE),cells=length(pos),features=length(features),row_index_base=1,assays=c('counts','c_counts_asinh'),dtype='float64 little endian C-order cells by markers',patient_identity_verified=TRUE),file.path(out,'EXPORT_COMPLETE.json'),pretty=TRUE,auto_unbox=TRUE)
cat('Cords exact-cell full-panel export COMPLETE\n')
