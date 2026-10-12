suppressPackageStartupMessages({library(data.table);library(survival)})
root<-normalizePath(commandArgs(TRUE)[1]);setwd(root)
stopifnot(startsWith(normalizePath(tempdir()),paste0(root,'/tmp/')))
d<-fread('results/SURVIVAL_INPUTS.csv');stopifnot(nrow(d)==280,sum(d$event)==96)
d[,sex:=factor(sex)];d[,stage_code:=factor(Stage)]
for(pair in list(c('caf_fraction','caf_z'),c('old_global_CAF_score','global_z'),c('vcaf_fraction','abundance_z'),c('all_mean','hla_z'))){d[,(pair[2]):=as.numeric(scale(get(pair[1])))]}
terms<-c('age','sex','stage_code','caf_z','global_z','abundance_z')
f<-function(extra)as.formula(paste('Surv(time_years,event)~',paste(c(terms,extra),collapse='+')))
rows<-list();issues<-list();tests<-list();ph<-list()
states<-grep('^(r(20|40|80)__|niche__)',names(d),value=TRUE)
extract<-function(fit,term){a<-summary(fit);data.table(term=term,coef=a$coef[term,'coef'],HR=a$conf.int[term,'exp(coef)'],CI_low=a$conf.int[term,'lower .95'],CI_high=a$conf.int[term,'upper .95'],p=a$coef[term,'Pr(>|z|)'])}
fit_safe<-function(formula,dat,id){tryCatch(withCallingHandlers(coxph(formula,data=dat,ties='efron',x=TRUE,na.action=na.fail),warning=function(w){issues[[length(issues)+1]]<<-data.table(id=id,message=conditionMessage(w));invokeRestart('muffleWarning')}),error=function(e){issues[[length(issues)+1]]<<-data.table(id=id,message=conditionMessage(e));NULL})}
ref<-fit_safe(f('hla_z'),d,'reference_HLA');old<-fread('../survival/results/COX_TERMS.csv')[config=='RAMP_k30'&state=='mean'&cohort=='all280'&model=='matched_full'&term=='state_z']
err<-abs(coef(ref)['hla_z']-old$coef);stopifnot(err<1e-10);fwrite(data.table(reference_coef_error=err),'results/COX_REPRODUCTION.csv')
for(state in states){
 sub<-copy(d[is.finite(get(state))]);sub[,spatial_z:=as.numeric(scale(get(state)))];primary<-grepl('^(r40__|niche__)',state)
 for(model in c('spatial_only','with_HLA')){
   vars<-if(model=='spatial_only')'spatial_z' else c('spatial_z','hla_z');fit<-fit_safe(f(vars),sub,paste(state,model))
   if(is.null(fit))next
   for(term in vars)rows[[length(rows)+1]]<-cbind(data.table(state=state,primary_state=primary,model=model,n=nrow(sub),events=sum(sub$event)),extract(fit,term))
   zz<-tryCatch(cox.zph(fit)$table,error=function(e)NULL)
   if(!is.null(zz))for(term in rownames(zz))ph[[length(ph)+1]]<-data.table(state=state,model=model,term=term,chisq=zz[term,'chisq'],df=zz[term,'df'],p=zz[term,'p'])
 }
}
a<-rbindlist(rows);a[,q_bh:=p.adjust(p,method='BH'),by=.(model,term,primary_state)];fwrite(a,'results/SPATIAL_DFS_COX.csv')
fwrite(rbindlist(ph),'results/PH_DIAGNOSTICS.csv')
fwrite(if(length(issues))rbindlist(issues) else data.table(id=character(),message=character()),'results/MODEL_ISSUES.csv')
writeLines(capture.output(sessionInfo()),'results/R_SESSION_INFO.txt')
print(a[primary_state&model=='spatial_only'][order(p)])
