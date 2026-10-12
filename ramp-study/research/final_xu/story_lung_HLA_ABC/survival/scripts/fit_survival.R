suppressPackageStartupMessages({library(data.table); library(survival)})
args <- commandArgs(TRUE)
root <- normalizePath(args[1]); setwd(root)
stopifnot(startsWith(normalizePath(tempdir()), paste0(root, '/tmp/')))
folder <- 'results'
stopifnot(file.exists(file.path(folder, 'PREPARATION_VERIFICATION.json')))
dir.create(file.path(folder, 'km'), showWarnings=FALSE)
input <- fread(file.path(folder, 'PATIENT_INPUTS.csv'), colClasses=list(character='patient_id'))
stopifnot(nrow(input)==1120, all(input$event %in% c(0,1)), all(input$time_years>0))
writeLines(c('Primary: RAMP k30, original 280 patients; k100 and KMEANSLOW baseline are fixed comparisons.',
 'Patient features: vCAF prediction mean and fraction above the frozen new cell-level threshold.',
 'KM: patient median split; no outcome-based cutoff selection.',
 'Matched full Cox: age, sex, stage, original CAF fraction, original global CAF score, vCAF fraction, state.',
 'Updated global Cox: same, replacing original global CAF score by new global CAF median.',
 'Sensitivity: exclude the original Cords reference donors; repeat the same models.',
 'All continuous scores/fractions standardized within analysis cohort; age remains in years.',
 'Reproduce four original adjusted state coefficients before accepting the analysis.'),
 file.path(folder,'SURVIVAL_DESIGN.txt'))
model_rows <- list(); term_rows <- list(); ph_rows <- list(); nested_rows <- list()
km_rows <- list(); five_rows <- list(); problems <- list(); scalings <- list(); replication <- list()

save_km <- function(d, config, state, cohort) {
  d <- copy(d); value <- d[[paste0('all_',state)]]; cut <- median(value)
  id <- paste(config,state,cohort,sep='__')
  d[,group:=factor(ifelse(value>=cut,'Higher','Lower'), levels=c('Lower','Higher'))]
  fit <- survfit(Surv(time_years,event)~group, data=d)
  lr <- survdiff(Surv(time_years,event)~group, data=d)
  fwrite(data.table(time=fit$time,survival=fit$surv,lower=fit$lower,upper=fit$upper,
    n_risk=fit$n.risk,n_event=fit$n.event,n_censor=fit$n.censor,
    group=rep(sub('group=','',names(fit$strata)),fit$strata)),file.path(folder,'km',paste0(id,'_curve.csv')))
  fwrite(d[,.(patient_id,time_years,event,group)],file.path(folder,'km',paste0(id,'_patients.csv')))
  km_rows[[length(km_rows)+1]] <<- data.table(id,config,state,cohort,n=nrow(d),events=sum(d$event),
    median_cut=cut,lower_n=sum(d$group=='Lower'),higher_n=sum(d$group=='Higher'),
    logrank_p=pchisq(lr$chisq,1,lower.tail=FALSE))
  for(g in c('Lower','Higher')) {
    sub <- d[group==g]; stopifnot(max(sub$time_years)>5)
    at5 <- summary(survfit(Surv(time_years,event)~1,data=sub), times=5,extend=FALSE)
    five_rows[[length(five_rows)+1]] <<- data.table(id,config,state,cohort,group=g,n=nrow(sub),events=sum(sub$event),
      at_risk_5y=sum(sub$time_years>=5),five_year_survival=as.numeric(at5$surv),
      lower=as.numeric(at5$lower),upper=as.numeric(at5$upper))
  }
}

prepare <- function(d,state_col,global_col) {
  d <- copy(d); d[,sex:=factor(sex)]; d[,stage_code:=factor(Stage)]
  d[,state_raw:=get(state_col)]; d[,global_raw:=get(global_col)]
  for(col in c('caf_fraction','global_raw','vcaf_fraction','state_raw')) {
    mu <- mean(d[[col]]); sig <- sd(d[[col]])
    stopifnot(is.finite(sig),sig>1e-12)
    d[,(paste0(col,'_z')):=(get(col)-mu)/sig]
  }
  d[,state_z:=state_raw_z]; d[,abundance_z:=vcaf_fraction_z]
  d
}
base_terms <- c('age','sex','caf_fraction_z','global_raw_z','stage_code','abundance_z')
formula_full <- as.formula(paste('Surv(time_years,event)~',paste(c(base_terms,'state_z'),collapse='+')))
formula_base <- as.formula(paste('Surv(time_years,event)~',paste(base_terms,collapse='+')))

old_terms <- fread('reference/LEGACY_COX_TERMS.csv')
for(kk in c(30,100)) for(st in c('mean','high')) {
  d <- prepare(input[config==paste0('RAMP_k',kk)],paste0('old_',st),'old_global_CAF_score')
  fit <- coxph(formula_full,data=d,ties='efron',na.action=na.fail)
  old <- old_terms[k==kk & state==st & model=='stage_abundance_state' & term=='state_z']
  stopifnot(nrow(old)==1,abs(unname(coef(fit)['state_z'])-old$coef)<1e-10)
  replication[[length(replication)+1]] <- data.table(k=kk,state=st,old_coef=old$coef,
    reproduced_coef=unname(coef(fit)['state_z']),absolute_error=abs(unname(coef(fit)['state_z'])-old$coef))
}

for(conf in unique(input$config)) for(st in c('mean','high')) for(cohort_name in c('all280','new_donors_only')) {
  original <- copy(input[config==conf])
  stopifnot(nrow(original)==280,sum(original$event)==96,!anyDuplicated(original$patient_id))
  if(cohort_name=='new_donors_only') original <- original[reference_donor==FALSE]
  save_km(original,conf,st,cohort_name)
  for(model_name in c('matched_full','updated_global')) {
    global <- if(model_name=='matched_full') 'old_global_CAF_score' else 'current_global_CAF_median'
    d <- prepare(original,paste0('all_',st),global)
    id <- paste(conf,st,cohort_name,model_name,sep='__')
    for(col in c('caf_fraction','global_raw','vcaf_fraction','state_raw'))
      scalings[[length(scalings)+1]] <- data.table(id,feature=col,mean=mean(d[[col]]),sd=sd(d[[col]]))
    warn <- character()
    fit <- tryCatch(withCallingHandlers(coxph(formula_full,data=d,ties='efron',x=TRUE,y=TRUE,model=TRUE,na.action=na.fail),
      warning=function(w){warn<<-c(warn,conditionMessage(w));invokeRestart('muffleWarning')}),error=function(e)e)
    if(inherits(fit,'error')) {problems[[length(problems)+1]]<-data.table(id,kind='fit_error',message=conditionMessage(fit));next}
    stopifnot(fit$n==nrow(d),fit$nevent==sum(d$event),all(is.finite(coef(fit))))
    su <- summary(fit)
    model_rows[[length(model_rows)+1]] <- data.table(id,config=conf,state=st,cohort=cohort_name,model=model_name,
      n=fit$n,events=fit$nevent,df=length(coef(fit)),loglik=fit$loglik[2],AIC=AIC(fit),
      apparent_c=unname(su$concordance[1]),c_se=unname(su$concordance[2]),warnings=paste(unique(warn),collapse=' | '))
    for(term in rownames(su$coef)) {
      ci <- su$conf.int[term,]
      term_rows[[length(term_rows)+1]] <- data.table(id,config=conf,state=st,cohort=cohort_name,model=model_name,term,
        coef=su$coef[term,'coef'],se=su$coef[term,'se(coef)'],hr=ci['exp(coef)'],ci_low=ci['lower .95'],
        ci_high=ci['upper .95'],p=su$coef[term,'Pr(>|z|)'])
    }
    ph <- tryCatch(cox.zph(fit),error=function(e)e)
    if(!inherits(ph,'error')) {
      for(term in rownames(ph$table)) ph_rows[[length(ph_rows)+1]] <- data.table(id,config=conf,state=st,
        cohort=cohort_name,model=model_name,term,chisq=ph$table[term,'chisq'],df=ph$table[term,'df'],p=ph$table[term,'p'])
    } else problems[[length(problems)+1]] <- data.table(id,kind='PH_error',message=conditionMessage(ph))
    reduced <- coxph(formula_base,data=d,ties='efron',na.action=na.fail)
    stat <- 2*(fit$loglik[2]-reduced$loglik[2]);stopifnot(stat>=-1e-7)
    nested_rows[[length(nested_rows)+1]] <- data.table(id,config=conf,state=st,cohort=cohort_name,model=model_name,
      LR_chisq=stat,p=pchisq(max(0,stat),1,lower.tail=FALSE),delta_AIC=AIC(fit)-AIC(reduced),
      delta_apparent_C=unname(su$concordance[1]-summary(reduced)$concordance[1]))
    if(length(warn))problems[[length(problems)+1]] <- data.table(id,kind='warning',message=paste(unique(warn),collapse=' | '))
  }
}
outputs <- list(COX_MODELS=model_rows,COX_TERMS=term_rows,PH_DIAGNOSTICS=ph_rows,NESTED_TESTS=nested_rows,
  KM_SUMMARY=km_rows,FIVE_YEAR_DFS=five_rows,PREDICTOR_SCALING=scalings,OLD_MODEL_REPRODUCTION=replication)
for(name in names(outputs))fwrite(rbindlist(outputs[[name]]),file.path(folder,paste0(name,'.csv')))
if(length(problems))fwrite(rbindlist(problems),file.path(folder,'MODEL_ISSUES.csv')) else
  fwrite(data.table(id=character(),kind=character(),message=character()),file.path(folder,'MODEL_ISSUES.csv'))
print(rbindlist(term_rows)[cohort=='all280' & term=='state_z'])
print(rbindlist(five_rows)[cohort=='all280'])
stopifnot(length(model_rows)==32,length(km_rows)==16,length(replication)==4)
writeLines(capture.output(sessionInfo()),file.path(folder,'R_SESSION_INFO.txt'))
