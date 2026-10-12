suppressPackageStartupMessages({library(data.table);library(survival);library(splines)})
root <- normalizePath(commandArgs(TRUE)[1]);setwd(root)
stopifnot(startsWith(normalizePath(tempdir()),paste0(root,'/tmp/')))
dir.create('results/km',showWarnings=FALSE)
inp <- fread('results/PATIENT_INPUTS.csv'); scores <- fread('results/ALL_PATIENT_COMPARTMENT_SCORES.csv')
ref <- fread('../survival/results/COX_TERMS.csv')
terms <- c('age','sex','stage_code','caf_z','global_z','abundance_z')
f <- function(extra=character())as.formula(paste('Surv(time_years,event) ~',paste(c(terms,extra),collapse='+')))
prep <- function(d){
 d<-copy(d);d[,sex:=factor(sex)];d[,stage_code:=factor(Stage)]
 for(pair in list(c('caf_fraction','caf_z'),c('old_global_CAF_score','global_z'),c('vcaf_fraction','abundance_z'),c('all_mean','state_z'))){
   d[,(pair[2]):=(get(pair[1])-mean(get(pair[1])))/sd(get(pair[1]))]
 };d
}
rows<-list();tests<-list();curve<-list();quart<-list();km<-list();coefs<-list();specific<-list();paired<-list();repro<-list();correlations<-list();issues<-list()
fit_safe<-function(formula,d,id){
 tryCatch(withCallingHandlers(coxph(formula,data=d,ties='efron',x=TRUE,na.action=na.fail),warning=function(w){issues[[length(issues)+1]]<<-data.table(id=id,message=conditionMessage(w));invokeRestart('muffleWarning')}),error=function(e){issues[[length(issues)+1]]<<-data.table(id=id,message=conditionMessage(e));NULL})
}
extract<-function(fit,term){
 a<-summary(fit);data.table(term=term,coef=a$coef[term,'coef'],HR=a$conf.int[term,'exp(coef)'],CI_low=a$conf.int[term,'lower .95'],CI_high=a$conf.int[term,'upper .95'],p=a$coef[term,'Pr(>|z|)'])
}
savekm<-function(d,by,id){
 fit<-survfit(as.formula(paste('Surv(time_years,event)~',by)),data=d)
 a<-data.table(time=fit$time,survival=fit$surv,lower=fit$lower,upper=fit$upper,n_risk=fit$n.risk,n_event=fit$n.event,n_censor=fit$n.censor,group=rep(sub(paste0(by,'='),'',names(fit$strata)),fit$strata))
 fwrite(a,paste0('results/km/',id,'_curve.csv'))
 for(g in levels(d[[by]])){
   sub<-d[get(by)==g];if(nrow(sub)==0)next
   at<-summary(survfit(Surv(time_years,event)~1,data=sub),times=5,extend=FALSE)
   km[[length(km)+1]]<<-data.table(id=id,group=g,n=nrow(sub),events=sum(sub$event),at_risk_5=sum(sub$time_years>=5),DFS5=if(length(at$surv))at$surv else NA_real_,CI_low=if(length(at$lower))at$lower else NA_real_,CI_high=if(length(at$upper))at$upper else NA_real_)
 }
}
for(conf in c('RAMP_k30','RAMP_k100')){
 d<-prep(inp[config==conf]);stopifnot(nrow(d)==280,sum(d$event)==96)
 base<-fit_safe(f(),d,paste(conf,'base'));linear<-fit_safe(f('state_z'),d,paste(conf,'linear'))
 old<-ref[config==conf&state=='mean'&cohort=='all280'&model=='matched_full'&term=='state_z']
 err<-abs(coef(linear)['state_z']-old$coef);stopifnot(err<1e-10)
 repro[[length(repro)+1]]<-data.table(config=conf,coef_error=err)
 for(t in names(coef(linear)))coefs[[length(coefs)+1]]<-cbind(data.table(config=conf,model='fully_adjusted'),extract(linear,t))
 correlations[[length(correlations)+1]]<-data.table(config=conf,rho=cor(d$all_mean,d$vcaf_fraction,method='spearman'))
 # Independent information beyond abundance; also estimate the interaction without changing splits.
 inter<-fit_safe(f(c('state_z','state_z:abundance_z')),d,paste(conf,'interaction'))
 tests[[length(tests)+1]]<-data.table(config=conf,test='state_addition',df=1,LR=2*(linear$loglik[2]-base$loglik[2]),p=pchisq(2*(linear$loglik[2]-base$loglik[2]),1,lower.tail=FALSE))
 tests[[length(tests)+1]]<-data.table(config=conf,test='state_abundance_interaction',df=1,LR=2*(inter$loglik[2]-linear$loglik[2]),p=pchisq(2*(inter$loglik[2]-linear$loglik[2]),1,lower.tail=FALSE))
 cut_state<-median(d$all_mean);cut_abund<-median(d$vcaf_fraction)
 d[,state_group:=factor(ifelse(all_mean>=cut_state,'High HLA','Low HLA'),levels=c('Low HLA','High HLA'))]
 d[,abundance_group:=factor(ifelse(vcaf_fraction>=cut_abund,'High abundance','Low abundance'),levels=c('Low abundance','High abundance'))]
 d[,joint_group:=factor(paste(abundance_group,state_group,sep=' | '),levels=c('Low abundance | Low HLA','Low abundance | High HLA','High abundance | Low HLA','High abundance | High HLA'))]
 fwrite(d[,.(patient_id,all_mean,vcaf_fraction,state_group,abundance_group,joint_group,time_years,event)],paste0('results/',conf,'_STATE_ABUNDANCE.csv'))
 savekm(d,'joint_group',paste0(conf,'_joint'))
 for(gr in levels(d$abundance_group)){
   sub<-d[abundance_group==gr];fit<-fit_safe(f('state_z'),sub,paste(conf,gr));lr<-survdiff(Surv(time_years,event)~state_group,data=sub)
   if(!is.null(fit))rows[[length(rows)+1]]<-cbind(data.table(config=conf,abundance_group=gr,n=nrow(sub),events=sum(sub$event),state_cut=cut_state,abundance_cut=cut_abund,logrank_p=pchisq(lr$chisq,1,lower.tail=FALSE)),extract(fit,'state_z'))
   savekm(sub,'state_group',paste0(conf,'_',gsub(' ','_',gr)))
 }
 # Fixed score quartiles, with categorical rather than forced monotonic effects.
 qq<-quantile(d$all_mean,c(.25,.5,.75));d[,score_quartile:=cut(all_mean,c(-Inf,qq,Inf),labels=paste0('Q',1:4),right=FALSE)]
 qfit<-fit_safe(f('score_quartile'),d,paste(conf,'quartiles'))
 for(gr in levels(d$score_quartile)){
   sub<-d[score_quartile==gr]
   estimate<-if(gr=='Q1')data.table(term='reference',coef=0,HR=1,CI_low=1,CI_high=1,p=NA_real_) else extract(qfit,paste0('score_quartile',gr))
   quart[[length(quart)+1]]<-cbind(data.table(config=conf,quartile=gr,n=nrow(sub),events=sum(sub$event),mean_score=mean(sub$all_mean),min_score=min(sub$all_mean),max_score=max(sub$all_mean)),estimate)
 }
 # Natural cubic spline: one internal knot plus fixed 10th/90th boundaries (2 df).
 bounds<-quantile(d$all_mean,c(.1,.9));knot<-median(d$all_mean)
 B<-ns(d$all_mean,knots=knot,Boundary.knots=bounds)
 d[,ns1:=B[,1]];d[,ns2:=B[,2]]
 sp<-fit_safe(f(c('ns1','ns2')),d,paste(conf,'spline'))
 tests[[length(tests)+1]]<-data.table(config=conf,test='nonlinearity',df=1,LR=2*(sp$loglik[2]-linear$loglik[2]),p=pchisq(max(0,2*(sp$loglik[2]-linear$loglik[2])),1,lower.tail=FALSE))
 grid<-seq(quantile(d$all_mean,.05),quantile(d$all_mean,.95),length.out=160)
 delta<-sweep(predict(B,grid),2,as.numeric(predict(B,knot)),'-');idx<-match(c('ns1','ns2'),names(coef(sp)))
 loghr<-as.vector(delta%*%coef(sp)[idx]);se<-sqrt(rowSums((delta%*%vcov(sp)[idx,idx]) * delta))
 curve[[length(curve)+1]]<-data.table(config=conf,score=grid,reference=knot,HR=exp(loghr),CI_low=exp(loghr-1.96*se),CI_high=exp(loghr+1.96*se),boundary_low=bounds[1],boundary_high=bounds[2])
 # Cell-compartment associations on matched subsets; no algorithm baseline.
 for(comp in unique(scores$compartment)){
   a<-copy(scores[compartment==comp&n_cells>=15]);a[,comp_score:=get(conf)]
   sub<-merge(inp[config==conf],a[,.(patient_id,comp_score,n_cells)],by='patient_id');sub<-prep(sub)
   sub[,comp_z:=(comp_score-mean(comp_score))/sd(comp_score)]
   fit<-fit_safe(f('comp_z'),sub,paste(conf,comp,'separate'))
   vfit<-fit_safe(f('state_z'),sub,paste(conf,comp,'matched_vcaf'))
   if(is.null(fit)||is.null(vfit))next
   specific[[length(specific)+1]]<-cbind(data.table(config=conf,compartment=comp,n=nrow(sub),events=sum(sub$event),rho_with_vcaf=cor(sub$all_mean,sub$comp_score,method='spearman')),extract(fit,'comp_z'))
   if(comp=='vCAF')next
   joint<-fit_safe(f(c('state_z','comp_z')),sub,paste(conf,comp,'joint'))
   if(is.null(joint))next
   vv<-vcov(joint);delta<-coef(joint)['state_z']-coef(joint)['comp_z'];delta_se<-sqrt(vv['state_z','state_z']+vv['comp_z','comp_z']-2*vv['state_z','comp_z'])
   for(t in c('state_z','comp_z'))paired[[length(paired)+1]]<-cbind(data.table(config=conf,compartment=comp,n=nrow(sub),events=sum(sub$event),model='joint',coefficient_difference=delta,difference_p=2*pnorm(-abs(delta/delta_se))),extract(joint,t))
 }
}
outputs<-list(ABUNDANCE_STRATIFIED_COX=rows,ADJUSTED_COX_TERMS=coefs,MODEL_TESTS=tests,QUARTILE_COX=quart,CONTINUOUS_HR=curve,KM_5YEAR=km,COMPARTMENT_COX=specific,JOINT_COMPARTMENT_COX=paired,REPRODUCTION=repro,STATE_ABUNDANCE_CORRELATION=correlations)
for(name in names(outputs)){
 a<-rbindlist(outputs[[name]],fill=TRUE)
 if(name=='COMPARTMENT_COX')a[,q_bh:=p.adjust(p,method='BH'),by=config]
 fwrite(a,paste0('results/',name,'.csv'))
}
fwrite(if(length(issues))rbindlist(issues) else data.table(id=character(),message=character()),'results/MODEL_ISSUES.csv')
writeLines(capture.output(sessionInfo()),'results/R_SESSION_INFO.txt')
print(rbindlist(rows));print(rbindlist(specific));print(rbindlist(quart));print(rbindlist(km)[grepl('k30_joint',id)])
