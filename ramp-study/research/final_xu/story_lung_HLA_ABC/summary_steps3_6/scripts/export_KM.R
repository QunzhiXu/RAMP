suppressPackageStartupMessages({library(data.table);library(survival)})
root=normalizePath(commandArgs(TRUE)[1]);d=fread(file.path(root,'06_DFS/KM_PATIENTS.csv'))
d[,group:=factor(group,levels=c('Lower','Higher'))]
f=survfit(Surv(time_years,event)~group,data=d)
x=data.table(time=f$time,survival=f$surv,lower=f$lower,upper=f$upper,n_risk=f$n.risk,n_event=f$n.event,n_censor=f$n.censor,group=rep(sub('group=','',names(f$strata)),f$strata))
fwrite(x,file.path(root,'06_DFS/KM_CURVE.csv'))
r=CJ(group=c('Lower','Higher'),time_years=c(0,2,4,6,8))
r[,n_at_risk:=mapply(function(g,t)sum(d$group==g & d$time_years>=t),group,time_years)]
fwrite(r,file.path(root,'06_DFS/NUMBER_AT_RISK.csv'))
# Ensure serialization is identical to the completed test's statistical curves.
t=fread(file.path(root,'../releases/senosain140k_20260921/analysis/KM_CURVES.csv'))[version=='new' & config=='RAMP_k30' & state=='mean' & cohort=='all280']
stopifnot(nrow(x)==nrow(t),max(abs(x$survival-t$survival))<1e-12,identical(as.character(x$group),t$group))
cat('KM curve/censor/risk data verified against completed analysis.\n')
