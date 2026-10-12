# Load only two named function expressions from the frozen sensitivity script.
# No top-level legacy paths, 20um arm, selection or new endpoint is executed.
suppressPackageStartupMessages({library(survival); library(jsonlite)})
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 3L)
env <- list(R=paste(R.version$major,R.version$minor,sep="."),
            survival=as.character(packageVersion("survival")),
            jsonlite=as.character(packageVersion("jsonlite")))
stopifnot(identical(env,list(R="4.3.2",survival="3.5.7",jsonlite="2.0.0")))
exprs <- parse(args[[1]])
selected <- character()
for (e in exprs) {
  if (is.call(e) && identical(e[[1]],as.name("<-")) &&
      is.symbol(e[[2]]) && as.character(e[[2]]) %in% c("safe_cox","analyze_feature")) {
    stopifnot(is.call(e[[3]]),identical(e[[3]][[1]],as.name("function")))
    eval(e, envir=.GlobalEnv)
    selected <- c(selected, as.character(e[[2]]))
  }
}
stopifnot(identical(selected,c("safe_cox","analyze_feature")))
features <- read.csv(args[[2]],check.names=FALSE,stringsAsFactors=FALSE)
stopifnot(nrow(features)==30L, !anyDuplicated(features$patient))
result <- analyze_feature("observed_30um_per_all",30)
write_json(result,file.path(args[[3]],"survival_30um.json"),auto_unbox=TRUE,pretty=TRUE,digits=16,na="null")
write_json(env,file.path(args[[3]],"environment.json"),auto_unbox=TRUE,pretty=TRUE)
writeLines(capture.output(sessionInfo()),file.path(args[[3]],"sessionInfo.txt"))
