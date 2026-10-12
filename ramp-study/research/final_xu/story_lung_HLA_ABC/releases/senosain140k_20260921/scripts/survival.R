suppressPackageStartupMessages({library(data.table);library(survival)})
root=normalizePath(commandArgs(TRUE)[1]);setwd(root)
stopifnot(startsWith(normalizePath(tempdir()),paste0(root,'/tmp/')))
input=fread('analysis/SURVIVAL_INPUTS.csv',colClasses=list(character='patient_id'))
stopifnot(nrow(input)==2240)
rows=list();phrows=list();kmrows=list();fiverows=list();curves=list();problems=list()
for(ver in c('old','new')) for(conf in unique(input$config)) for(st in c('mean','high')) for(cohort in c('all280','new_donors_only')) {
 d=copy(input[version==ver & config==conf]);stopifnot(nrow(d)==280,sum(d$event)==96)
 if(cohort=='new_donors_only')d=d[reference_donor==FALSE]
 d[,state_raw:=get(paste0('all_',st))];cut=median(d$state_raw)
 d[,group:=factor(ifelse(state_raw>=cut,'Higher','Lower'),levels=c('Lower','Higher'))]
 d[,sex:=factor(sex)];d[,stage_code:=factor(Stage)]
 for(col in c('state_raw','caf_fraction','vcaf_fraction','current_global_CAF_median'))d[,(paste0(col,'_z')):=as.numeric(scale(get(col)))]
 d[,state_z:=state_raw_z];d[,abundance_z:=vcaf_fraction_z];d[,global_raw_z:=current_global_CAF_median_z]
 f=Surv(time_years,event)~age+sex+caf_fraction_z+global_raw_z+stage_code+abundance_z+state_z
 warns=character()
 fit=withCallingHandlers(coxph(f,data=d,ties='efron',x=TRUE,y=TRUE,na.action=na.fail),warning=function(w){warns<<-c(warns,conditionMessage(w));invokeRestart('muffleWarning')})
 stopifnot(fit$n==nrow(d),all(is.finite(coef(fit))))
 su=summary(fit)
 for(term in rownames(su$coef))rows[[length(rows)+1]]=data.table(version=ver,config=conf,state=st,cohort=cohort,term=term,n=nrow(d),events=sum(d$event),coef=su$coef[term,'coef'],se=su$coef[term,'se(coef)'],hr=su$conf.int[term,'exp(coef)'],ci_low=su$conf.int[term,'lower .95'],ci_high=su$conf.int[term,'upper .95'],p=su$coef[term,'Pr(>|z|)'])
 ph=cox.zph(fit)
 for(term in rownames(ph$table))phrows[[length(phrows)+1]]=data.table(version=ver,config=conf,state=st,cohort=cohort,term=term,p=ph$table[term,'p'])
 km=survfit(Surv(time_years,event)~group,data=d);lr=survdiff(Surv(time_years,event)~group,data=d)
 kmrows[[length(kmrows)+1]]=data.table(version=ver,config=conf,state=st,cohort=cohort,median_cut=cut,lower_n=sum(d$group=='Lower'),higher_n=sum(d$group=='Higher'),logrank_p=pchisq(lr$chisq,1,lower.tail=FALSE))
 curves[[length(curves)+1]]=data.table(version=ver,config=conf,state=st,cohort=cohort,time=km$time,survival=km$surv,lower=km$lower,upper=km$upper,group=rep(sub('group=','',names(km$strata)),km$strata))
 for(g in c('Lower','Higher')){
  sub=d[group==g];five=summary(survfit(Surv(time_years,event)~1,data=sub),times=5,extend=FALSE)
  fiverows[[length(fiverows)+1]]=data.table(version=ver,config=conf,state=st,cohort=cohort,group=g,five_year_survival=as.numeric(five$surv),lower=as.numeric(five$lower),upper=as.numeric(five$upper),n_risk_5y=sum(sub$time_years>=5))
 }
 if(length(warns))problems[[length(problems)+1]]=data.table(version=ver,config=conf,state=st,cohort=cohort,message=paste(unique(warns),collapse='; '))
}
fwrite(rbindlist(rows),'analysis/COX_TERMS.csv');fwrite(rbindlist(phrows),'analysis/PH_DIAGNOSTICS.csv')
fwrite(rbindlist(kmrows),'analysis/KM_SUMMARY.csv');fwrite(rbindlist(fiverows),'analysis/FIVE_YEAR_DFS.csv');fwrite(rbindlist(curves),'analysis/KM_CURVES.csv')
fwrite(if(length(problems))rbindlist(problems) else data.table(message=character()),'analysis/SURVIVAL_WARNINGS.csv')
writeLines(capture.output(sessionInfo()),'analysis/R_SESSION_INFO.txt')
print(rbindlist(rows)[config=='RAMP_k30' & cohort=='all280' & term=='state_z'])
