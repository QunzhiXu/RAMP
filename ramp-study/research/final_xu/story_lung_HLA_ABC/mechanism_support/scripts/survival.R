suppressPackageStartupMessages({library(data.table);library(survival);library(jsonlite)})
root<-normalizePath(commandArgs(TRUE)[1]);setwd(root)
stopifnot(basename(root)=='mechanism_support',startsWith(normalizePath(tempdir()),paste0(root,'/tmp/')))
d<-fread('results/SURVIVAL_INPUTS.csv');stopifnot(nrow(d)==280,sum(d$event)==96)
d[,sex:=factor(sex)];d[,stage_code:=factor(Stage)];d[,TMA:=factor(TMA)]
for(pair in list(c('caf_fraction','caf_z'),c('old_global_CAF_score','global_z'),c('vcaf_fraction','abundance_z'),c('all_mean','hla_z'))){d[,(pair[2]):=as.numeric(scale(get(pair[1])))]}
terms<-c('age','sex','stage_code','caf_z','global_z','abundance_z')
formula_for<-function(extra)as.formula(paste('Surv(time_years,event)~',paste(c(terms,extra),collapse='+')))
issues<-list();rows<-list();ph<-list()
fit_safe<-function(f,dat,id){tryCatch(withCallingHandlers(coxph(f,data=dat,ties='efron',x=TRUE,na.action=na.fail),warning=function(w){issues[[length(issues)+1]]<<-data.table(id=id,message=conditionMessage(w));invokeRestart('muffleWarning')}),error=function(e){issues[[length(issues)+1]]<<-data.table(id=id,message=conditionMessage(e));NULL})}
ref<-fit_safe(formula_for('hla_z'),d,'original_HLA')
old<-fread('../survival/results/COX_TERMS.csv')[config=='RAMP_k30'&state=='mean'&cohort=='all280'&model=='matched_full'&term=='state_z']
error<-abs(coef(ref)['hla_z']-old$coef);stopifnot(error<1e-10)
fwrite(data.table(original_HLA_coef_error=error),'results/COX_REPRODUCTION.csv')
for(marker in c('CD73','IDO')){
 d[,marker_z:=as.numeric(scale(get(paste0(marker,'_mean_asinh'))))]
 for(batch in c(FALSE,TRUE))for(joint in c(FALSE,TRUE)){
  extra<-c('marker_z',if(joint)'hla_z',if(batch)'TMA')
  id<-paste(marker,batch,joint,sep=':');fit<-fit_safe(formula_for(extra),d,id)
  if(is.null(fit))next
  s<-summary(fit)
  for(term in c('marker_z',if(joint)'hla_z'))rows[[length(rows)+1]]<-data.table(marker=marker,adjust_TMA=batch,with_HLA=joint,n=nrow(d),events=sum(d$event),term=term,coef=s$coef[term,'coef'],HR=s$conf.int[term,'exp(coef)'],CI_low=s$conf.int[term,'lower .95'],CI_high=s$conf.int[term,'upper .95'],p=s$coef[term,'Pr(>|z|)'])
  z<-tryCatch(cox.zph(fit)$table,error=function(e)NULL)
  if(!is.null(z))for(term in rownames(z))ph[[length(ph)+1]]<-data.table(marker=marker,adjust_TMA=batch,with_HLA=joint,term=term,p=z[term,'p'])
 }
}
r<-rbindlist(rows);r[,q_bh:=p.adjust(p,method='BH'),by=.(adjust_TMA,with_HLA,term)]
fwrite(r,'results/MEASURED_MARKER_DFS_COX.csv')
fwrite(rbindlist(ph),'results/COX_PH_DIAGNOSTICS.csv')
fwrite(if(length(issues))rbindlist(issues) else data.table(id=character(),message=character()),'results/COX_ISSUES.csv')
p<-fread('results/PATIENT_MARKER_MEANS.csv');p[,TMA:=factor(TMA)]
out<-list()
for(pop in c('all_available','DFS_280_members')){
 a<-if(pop=='all_available')copy(p[n_vcaf>=15]) else copy(p[patient_id%in%d$patient_id])
 a[,h_rank:=rank(HLA_mean,ties.method='average')]
 for(marker in c('CD73','IDO')){
  a[,m_rank:=rank(get(paste0(marker,'_mean_asinh')),ties.method='average')]
  hr<-residuals(lm(h_rank~TMA,data=a));mr<-residuals(lm(m_rank~TMA,data=a))
  ct<-cor.test(hr,mr,method='pearson')
  out[[length(out)+1]]<-data.table(population=pop,marker=marker,n=nrow(a),partial_spearman_TMA=as.numeric(ct$estimate))
 }
}
fwrite(rbindlist(out),'results/BETWEEN_PATIENT_TMA_ADJUSTED.csv')
writeLines(capture.output(sessionInfo()),'results/R_SURVIVAL_SESSION.txt')
print(r[adjust_TMA==TRUE])
