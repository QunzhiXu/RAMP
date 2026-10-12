"""Verify reproduced DFS models and present the fixed vCAF story."""
import package_runtime as rt
from package_runtime import BASE
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=BASE/'results'
FIG=OUT/'figures'
COLORS={'Lower':'#24758C','Higher':'#CF6848'}


def verify_statistics():
    models=pd.read_csv(OUT/'COX_MODELS.csv');km=pd.read_csv(OUT/'KM_SUMMARY.csv')
    terms=pd.read_csv(OUT/'COX_TERMS.csv');five=pd.read_csv(OUT/'FIVE_YEAR_DFS.csv')
    issues=pd.read_csv(OUT/'MODEL_ISSUES.csv')
    assert len(models)==32 and len(km)==16 and len(five)==32 and issues.empty
    old=pd.read_csv(BASE/'reference/PREVIOUS_COX_TERMS.csv')
    joined=terms.merge(old,on=['id','term'],validate='one_to_one',suffixes=('','_old'))
    assert len(joined)==len(terms)==len(old)
    errors={c:float(np.abs(joined[c]-joined[c+'_old']).max()) for c in ['coef','se','hr','ci_low','ci_high','p']}
    assert max(errors.values())<1e-9,errors
    previous_five=pd.read_csv(BASE/'reference/PREVIOUS_FIVE_YEAR_DFS.csv')
    joined=five.merge(previous_five,on=['id','group'],validate='one_to_one',suffixes=('','_old'))
    probability_error=float(np.abs(joined.five_year_survival-joined.five_year_survival_old).max())
    assert len(joined)==32 and probability_error<1e-12
    previous_km=pd.read_csv(BASE/'reference/PREVIOUS_KM_SUMMARY.csv')
    joined_km=km.merge(previous_km,on='id',validate='one_to_one',suffixes=('','_old'))
    logrank_error=float(np.abs(joined_km.logrank_p-joined_km.logrank_p_old).max())
    assert len(joined_km)==16 and logrank_error<1e-12
    independent=[]
    for _,r in five.iterrows():
        d=pd.read_csv(OUT/'km'/f'{r.id}_patients.csv');g=d[d.group.eq(r.group)]
        survival=1.
        for t in sorted(g.loc[g.event.eq(1)&g.time_years.le(5),'time_years'].unique()):
            events=int((g.event.eq(1)&g.time_years.eq(t)).sum());risk=int(g.time_years.ge(t).sum())
            survival*=1-events/risk
        error=abs(survival-r.five_year_survival)
        assert error<1e-12 and len(g)==r.n and int(g.time_years.ge(5).sum())==r.at_risk_5y
        independent.append({'id':r.id,'group':r.group,'absolute_error':error})
    pd.DataFrame(independent).to_csv(OUT/'INDEPENDENT_KM_VERIFICATION.csv',index=False)
    legacy=pd.read_csv(OUT/'OLD_MODEL_REPRODUCTION.csv')
    assert len(legacy)==4 and legacy.absolute_error.max()<1e-10
    rt.save_json(OUT/'STATISTICAL_REPRODUCTION.json',{'status':'PASS','time_utc':rt.now(),'n_Cox_models':32,
        'n_KM_analyses':16,'n_five_year_probabilities':32,'Cox_max_errors':errors,
        'KM_probability_max_error':probability_error,'logrank_p_max_error':logrank_error,
        'independent_KM_max_error':max(r['absolute_error'] for r in independent),'model_warnings_or_errors':0,
        'same_previously_selected_analysis':True})
    return models,km,terms,five


def draw_km(k,km,five):
    fig,axes=plt.subplots(1,2,figsize=(13,6.8))
    fig.subplots_adjust(left=.075,right=.97,top=.79,bottom=.32,wspace=.28)
    for ax,state,title in zip(axes,['mean','high'],['Mean predicted HLA-ABC rank','High-state vCAF fraction']):
        ident=f'RAMP_k{k}__{state}__all280'
        curve=pd.read_csv(OUT/'km'/f'{ident}_curve.csv')
        patients=pd.read_csv(OUT/'km'/f'{ident}_patients.csv')
        stat=km[km.id.eq(ident)].iloc[0];fs=five[five.id.eq(ident)].set_index('group')
        for group,color in COLORS.items():
            c=curve[curve.group.eq(group)]
            x=np.r_[0,c.time];y=np.r_[1,c.survival]
            n=int(patients.group.eq(group).sum())
            ax.step(x,y,where='post',color=color,lw=2.1,label=f'{group} (n={n})')
            ax.fill_between(x,np.r_[1,c.lower],np.r_[1,c.upper],step='post',color=color,alpha=.13,lw=0)
            censored=c[c.n_censor.gt(0)]
            ax.scatter(censored.time,censored.survival,marker='+',s=17,color=color,linewidths=.7)
        ax.set(xlim=(0,8),ylim=(0,1.04),xticks=[0,2,4,6,8],xlabel='Years',ylabel='Disease-free survival',title=title)
        ax.legend(loc='lower left',frameon=False,fontsize=9)
        ax.text(.04,.16,f'5-year DFS: {fs.loc["Lower","five_year_survival"]:.1%} / {fs.loc["Higher","five_year_survival"]:.1%}\nLog-rank p = {stat.logrank_p:.3g}',
                transform=ax.transAxes,fontsize=10,bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
        ax.text(0,-.21,'Number at risk',transform=ax.transAxes,fontsize=9)
        for i,(group,color) in enumerate(COLORS.items()):
            ax.text(-.035,-.30-i*.09,group,transform=ax.transAxes,ha='right',color=color,fontsize=9)
            for t in [0,2,4,6,8]:
                n=int((patients.group.eq(group)&patients.time_years.ge(t)).sum())
                ax.text(t/8,-.30-i*.09,str(n),transform=ax.transAxes,ha='center',color=color,fontsize=9)
    fig.suptitle(f'Predicted vCAF HLA-ABC state and DFS | k = {k}',fontsize=17,y=.97)
    fig.text(.5,.9,'280 patients · 96 DFS events · all author-annotated vCAFs · patient median split',ha='center',fontsize=11)
    fig.text(.5,.035,'Shading: 95% CI. Plus signs: censoring. Low/high groups refer to patient scores; no survival-based cutpoint search.',ha='center',fontsize=9,color='#53606b')
    fig.savefig(FIG/f'vCAF_DFS_k{k}.png',dpi=240);fig.savefig(FIG/f'vCAF_DFS_k{k}.svg');plt.close(fig)


def draw_cox(terms):
    cases=[('RAMP k30 | primary','RAMP_k30','matched_full','all280'),
           ('RAMP k30 | updated global CAF','RAMP_k30','updated_global','all280'),
           ('Baseline k30 | matched covariates','baseline_k30','matched_full','all280'),
           ('Baseline k30 | updated global CAF','baseline_k30','updated_global','all280'),
           ('RAMP k30 | 258 non-reference patients','RAMP_k30','matched_full','new_donors_only')]
    fig,axes=plt.subplots(1,2,figsize=(14.4,5.7))
    fig.subplots_adjust(left=.32,right=.97,top=.78,bottom=.17,wspace=.25)
    for ax,state,title in zip(axes,['mean','high'],['Mean predicted HLA-ABC rank','High-state vCAF fraction']):
        for i,(label,config,model,cohort) in enumerate(cases):
            r=terms[terms.config.eq(config)&terms.state.eq(state)&terms.model.eq(model)&terms.cohort.eq(cohort)&terms.term.eq('state_z')].iloc[0]
            color='#167d8d' if config.startswith('RAMP') else '#b58654'
            ax.errorbar(r.hr,i,xerr=[[r.hr-r.ci_low],[r.ci_high-r.hr]],fmt='o',color=color,capsize=3,lw=1.5)
            ax.text(.98,i-.23,f'{r.hr:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}], p={r.p:.3g}',
                    ha='right',fontsize=8.5,transform=ax.get_yaxis_transform())
        ax.axvline(1,color='#949ca5',ls='--',lw=1)
        ax.set(ylim=(4.6,-.7),xlim=(.85,2.2),yticks=range(5),yticklabels=[c[0] for c in cases] if state=='mean' else ['']*5,
               xlabel='Adjusted hazard ratio per 1 SD (95% CI)',title=title)
    fig.suptitle('Adjusted DFS association | fixed k=30 analyses',fontsize=17,y=.96)
    fig.text(.5,.87,'Age, sex, provider Stage, CAF fraction, global CAF score and vCAF fraction included',ha='center',fontsize=11)
    fig.text(.5,.045,'Continuous patient scores; exploratory association. All prespecified configurations are retained in the result tables.',ha='center',fontsize=10,color='#53606b')
    fig.savefig(FIG/'vCAF_adjusted_DFS_k30.png',dpi=240);fig.savefig(FIG/'vCAF_adjusted_DFS_k30.svg');plt.close(fig)


def write_report(models,km,terms,five):
    rows=[]
    for config in ['RAMP_k30','RAMP_k100','baseline_k30','baseline_k100']:
        for state in ['mean','high']:
            ident=f'{config}__{state}__all280';fs=five[five.id.eq(ident)].set_index('group')
            ks=km[km.id.eq(ident)].iloc[0]
            r=terms[terms.config.eq(config)&terms.state.eq(state)&terms.cohort.eq('all280')&terms.model.eq('matched_full')&terms.term.eq('state_z')].iloc[0]
            rows.append({'config':config,'state':state,'n':int(ks.n),'events':int(ks.events),
                         'lower_DFS5':fs.loc['Lower','five_year_survival'],'higher_DFS5':fs.loc['Higher','five_year_survival'],
                         'patient_median_cut':ks.median_cut,'logrank_p':ks.logrank_p,'adjusted_HR':r.hr,
                         'CI_low':r.ci_low,'CI_high':r.ci_high,'cox_p':r.p})
    summary=pd.DataFrame(rows);summary.to_csv(OUT/'MAIN_RESULTS.csv',index=False)
    design=json.loads((OUT/'DESIGN.json').read_text());cut=design['cell_thresholds']['RAMP_k30']
    lines=['# Cords vCAF 预测 HLA-ABC 状态与 DFS','',
           '已复现固定的全部源细胞 → 全部目标细胞方案，并从 527 位 Cords 患者重新应用纳入条件，得到同一 280 人、96 个 DFS 事件。细胞预测来自 ../data_cords；没有重新对齐或按生存结果调参。','',
           '## 状态与队列定义','',
           'vCAF 使用作者原始 cell_type=vCAF 注释。主指标是每位患者全部 vCAF 的预测 HLA-ABC 秩均值；伴随指标是这些 vCAF 中高状态细胞的比例。预测源与 Zheng 验证一致：全部 28,000 个 Senosain 参考细胞、共享九标记、源端 StandardScaler、距离加权 kNN，k=30 为主、k=100 为敏感性。','',
           f'RAMP k30 的细胞高状态阈值为 **{cut:.12f}**，等于固定的原 150,024 个抽样 CAF 的预测分数中位数。它与每个 KM 面板使用的患者分数中位数是两个不同层次的阈值，也与 Zheng C30 图的颜色范围无关。','',
           '纳入条件：作者早期 stage_group=I-II；至少 15 个 vCAF、50 个全量 CAF；DFS 时间为正、事件为 0/1；年龄、性别、原 CAF 比例和原 global-CAF 分数完整。Stage 作为作者分类编码进入 Cox。没有 T 细胞距离、近／远 T 数量或 SMA-CAF 条件。COHORT_FLOW.csv 保留完整筛选过程。','',
           '## 结果','',
           '| 配置 | 患者指标 | 5 年 DFS：低／高组 | 调整 HR（95% CI） | Cox P |',
           '| --- | --- | ---: | ---: | ---: |']
    for _,r in summary.iterrows():
        name='均值' if r.state=='mean' else '高状态比例'
        lines.append(f'| {r.config} | {name} | {r.lower_DFS5:.1%} / {r.higher_DFS5:.1%} | {r.adjusted_HR:.3f} ({r.CI_low:.3f}–{r.CI_high:.3f}) | {r.cox_p:.4g} |')
    lines += ['',
              'HR 对应患者分数每增加 1 个标准差。主 Cox 调整年龄、性别、Stage、原 CAF 比例、原 global-CAF 分数以及 vCAF/全部 CAF 的比例。每种状态分别建模；KM 为未经协变量调整的患者中位数分组，图含置信区间、删失标记与风险人数。','',
              '## 固定敏感性与解释','',
              '保留四种配置（RAMP/baseline × k30/k100）、两种状态、两种 global-CAF 调整，以及排除 22 位参考患者后的 258 人分析，共 32 个 Cox 模型和 16 组 KM。全部结果见 COX_TERMS.csv、COX_MODELS.csv、PH_DIAGNOSTICS.csv 和 NESTED_TESTS.csv。比例风险诊断结果如实保留；没有根据 P 值筛选模型。','',
              '结果表示较高的预测 vCAF HLA-ABC 状态与较差 DFS 相关，不说明 HLA-ABC 导致复发。Baseline 同样存在关联，不能归为 RAMP 独有优势；Zheng 的 HLA-A 验证支持相关蛋白信号的转移，但不是 Cords vCAF 实测 HLA-ABC 或生存关联的独立验证。这里复现的是已经探索过的固定分析。','',
              '## 可复现文件','',
              '- figures/vCAF_DFS_k30.png / .svg：主 KM 图，左为均值、右为高状态比例。',
              '- figures/vCAF_DFS_k100.png / .svg：k100 敏感性图。',
              '- figures/vCAF_adjusted_DFS_k30.png / .svg：主调整模型与固定敏感性结果。',
              '- PATIENT_INPUTS.csv：全部 4 × 280 行模型输入；ALL_PATIENTS.csv：527 人各配置纳入状态。',
              '- MAIN_RESULTS.csv：主比较数字；km/：全部曲线及患者分组。',
              '- COHORT_REPRODUCTION.csv：患者、分数、协变量、结局与原分析一致。',
              '- STATISTICAL_REPRODUCTION.json：32 个模型与既有结果的比较；32 个五年 KM 概率另作独立乘积限重算。','']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines))
    rt.save_json(OUT/'SUMMARY.json',{'n_initial_patients':527,'n_survival_patients':280,'events':96,
                                  'primary_config':'RAMP_k30','primary_results':summary[summary.config.eq('RAMP_k30')].to_dict('records')})


def main():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    models,km,terms,five=verify_statistics()
    draw_km(30,km,five);draw_km(100,km,five);draw_cox(terms);write_report(models,km,terms,five)
    design=json.loads((OUT/'DESIGN.json').read_text())
    assert all(rt.sha(p)==h for p,h in design['input_sha256'].items())
    files=[p for folder in ['data','results'] for p in (BASE/folder).rglob('*') if p.is_file()
           and p.name not in {'RUN_MANIFEST.json','RUN_STATUS.json','RUNNING.lock','FIGURE_QA.json'}]
    rt.save_json(OUT/'RUN_MANIFEST.json',{'time_utc':rt.now(),'inputs':design['input_sha256'],
        'scripts':{str(p.relative_to(BASE)):rt.sha(p) for p in (BASE/'scripts').glob('*') if p.is_file()},
        'reference':{str(p.relative_to(BASE)):rt.sha(p) for p in (BASE/'reference').glob('*') if p.is_file()},
        'outputs':{str(p.relative_to(BASE)):rt.sha(p) for p in files}})
    rt.save_json(OUT/'RUN_STATUS.json',{'phase':'complete','time_utc':rt.now(),'visual_review':'pending'})
    print(json.dumps(json.loads((OUT/'SUMMARY.json').read_text()),indent=2),flush=True)


if __name__=='__main__':main()
