"""Compare whole-patient CD8 expression using the unchanged survival groups."""
from pathlib import Path
import os,sys,json,hashlib,datetime
BASE=Path(__file__).resolve().parent
MECH=BASE.parent
STORY=MECH.parent
sys.dont_write_bytecode=True
def allowed(p):
    if isinstance(p,int) or p is None:return
    p=Path(os.fsdecode(p)).resolve()
    if p==Path('/dev/null'):return
    if not p.is_relative_to(BASE):raise PermissionError(f'Write outside patient_cd8: {p}')
def audit(e,a):
    if e=='open':
        p,m,f=a
        if (isinstance(m,str) and any(c in m for c in 'wax+')) or (isinstance(f,int) and f&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):allowed(p)
    elif e in {'os.remove','os.rmdir','os.mkdir','os.chmod','os.utime','os.truncate'}:allowed(a[0])
    elif e in {'os.rename','os.replace'}:allowed(a[0]);allowed(a[1])
    elif e in {'os.link','os.symlink'}:allowed(a[1])
sys.addaudithook(audit)
for key,p in {'TMPDIR':BASE/'tmp','MPLCONFIGDIR':BASE/'cache/matplotlib','XDG_CACHE_HOME':BASE/'cache/xdg'}.items():p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
OUT=BASE/'results';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
INPUTS={'measured':MECH/'data/MEASURED_MARKERS.csv.gz','clinical':MECH/'results/SURVIVAL_INPUTS.csv','km':STORY/'survival/results/KM_SUMMARY.csv','metadata':STORY/'data_cords/data/CELL_METADATA.csv.gz'}
COLORS={'Lower':'#588299','Higher':'#cb506a'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
SEED=20260909
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def bh(p):
    p=np.asarray(p,float);ix=np.argsort(p);q=np.empty(len(p));q[ix]=np.minimum(1,np.minimum.accumulate((p[ix]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]);return q

def freeze():
    hashes={key:sha(p) for key,p in INPUTS.items()}
    clinical=pd.read_csv(INPUTS['clinical'],float_precision='round_trip')
    km=pd.read_csv(INPUTS['km'],float_precision='round_trip')
    row=km[(km.config=='RAMP_k30')&(km.state=='mean')&(km.cohort=='all280')].iloc[0]
    cut=float(row.median_cut)
    assert len(clinical)==280 and clinical.patient_id.is_unique
    # R's KM summary stores fewer decimal digits than the prediction table.
    assert abs(cut-float(clinical.all_mean.median()))<1e-12
    assert np.array_equal(clinical.all_mean>=cut,clinical.all_mean>=clinical.all_mean.median())
    clinical['group']=np.where(clinical.all_mean>=cut,'Higher','Lower')
    assert clinical.group.value_counts().to_dict()=={'Higher':140,'Lower':140} or clinical.group.value_counts().to_dict()=={'Lower':140,'Higher':140}
    clinical['high_HLA']=clinical.group.eq('Higher').astype(int)
    if (OUT/'DESIGN.json').exists():
        old=json.loads((OUT/'DESIGN.json').read_text());assert old['input_sha256']==hashes
    else:
        save(OUT/'DESIGN.json',{'time_utc':now(),'input_sha256':hashes,'patient_cutoff':cut,
          'population':'The exact original 280 DFS patients; fixed original RAMP k30 mean-vCAF-HLA median split, 140 lower and 140 higher. Group membership never recomputed after sensitivity exclusions.',
          'primary_endpoints':['mean of asinh(provider counts PD1) over all CD8 of each patient','mean of asinh(provider counts TCF1) over all CD8 of each patient'],
          'sampling':'Every author-annotated CD8 cell in the available tissue cores of these patients, regardless of proximity to vCAF. Each patient has equal statistical weight. No CD8 values normalized to ranks within patient.',
          'primary_test':'Mann-Whitney U on patient means, two-sided; BH across PD1 and TCF1. Report mean difference, 3000 group-stratified patient-bootstrap CI and rank-biserial effect. These are independent groups, not paired.',
          'adjusted_test':'Patient-level OLS with HC3 robust SE; high-HLA group plus TMA fixed effects, age, sex and Stage categorical effects. Report group coefficient standardized by the endpoint SD. Continuous HLA score companion uses the same covariates.',
          'sensitivity':['Keep only patients with >=20 CD8 (fixed group labels)','Keep only patients with >=50 CD8 (fixed group labels)','Median asinh marker per patient','Equal weight to each ROI mean instead of each cell within patient','Full original CAF covariates additionally adjusted','CD8 abundance and cell-count imbalance described separately'],
          'predicted_directions':{'PD1':'higher in high-HLA patients','TCF1':'lower in high-HLA patients'},
          'scope':'Whole-patient CD8 marker associations; does not demonstrate a local vCAF-CD8 interaction or T-cell dysfunction. Primary model and HLA predictions unchanged.',
          'seed':SEED})
    return hashes,clinical

def ols_hc3(y,x,term):
    x=np.asarray(x,float);y=np.asarray(y,float)
    inv=np.linalg.pinv(x.T@x);beta=inv@x.T@y
    residual=y-x@beta;hat=np.einsum('ij,jk,ik->i',x,inv,x)
    w=residual**2/np.maximum(1-hat,1e-8)**2
    cov=inv@(x.T@(w[:,None]*x))@inv
    se=np.sqrt(max(cov[term,term],0));df=len(y)-np.linalg.matrix_rank(x)
    z=beta[term]/se;pv=float(2*stats.t.sf(abs(z),df));critical=stats.t.ppf(.975,df)
    return dict(coef=float(beta[term]),se=float(se),ci_low=float(beta[term]-critical*se),ci_high=float(beta[term]+critical*se),p=pv,df=int(df),n=len(y),max_leverage=float(hat.max()))

def design_matrix(g,exposure='high_HLA',full=False):
    x=pd.DataFrame({'intercept':np.ones(len(g))},index=g.index)
    e=g[exposure].astype(float)
    if exposure=='all_mean':e=(e-e.mean())/e.std(ddof=1)
    x['exposure']=e
    x['age']=(g.age-g.age.mean())/g.age.std(ddof=1)
    cat=pd.get_dummies(g[['TMA','sex','Stage']].astype(str),drop_first=True,dtype=float)
    x=pd.concat([x,cat],axis=1)
    if full:
        for col in ['caf_fraction','old_global_CAF_score','vcaf_fraction']:x[col]=(g[col]-g[col].mean())/g[col].std(ddof=1)
    return x

def compare(g,column):
    a=g.loc[g.group.eq('Lower'),column].to_numpy();b=g.loc[g.group.eq('Higher'),column].to_numpy()
    test=stats.mannwhitneyu(b,a,alternative='two-sided',method='asymptotic')
    rng=np.random.default_rng(SEED)
    difference=b[rng.integers(len(b),size=(3000,len(b)))].mean(1)-a[rng.integers(len(a),size=(3000,len(a)))].mean(1)
    lo,hi=np.quantile(difference,[.025,.975])
    pooled=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    return dict(n_lower=len(a),n_higher=len(b),lower_mean=float(a.mean()),higher_mean=float(b.mean()),lower_median=float(np.median(a)),higher_median=float(np.median(b)),difference=float(b.mean()-a.mean()),ci_low=float(lo),ci_high=float(hi),p=float(test.pvalue),rank_biserial=float(2*test.statistic/(len(a)*len(b))-1),standardized_difference=float((b.mean()-a.mean())/pooled))

def analyze(clinical):
    m=pd.read_csv(INPUTS['measured'],float_precision='round_trip')
    t=m[m.cell_type.eq('CD8')&m.patient_id.isin(clinical.patient_id)].copy()
    assert t.source_row_id.is_unique and t.patient_id.nunique()==280
    for marker in ['PD1','TCF1']:t[marker+'_asinh']=np.arcsinh(t[marker])
    a=t.groupby('patient_id',as_index=False).agg(n_cd8=('source_row_id','size'),n_cd8_rois=('roi_id','nunique'),PD1_mean=('PD1_asinh','mean'),TCF1_mean=('TCF1_asinh','mean'),PD1_median=('PD1_asinh','median'),TCF1_median=('TCF1_asinh','median'),PD1_raw_mean=('PD1','mean'),TCF1_raw_mean=('TCF1','mean'))
    roi=t.groupby(['patient_id','roi_id'],as_index=False)[['PD1_asinh','TCF1_asinh']].mean()
    re=roi.groupby('patient_id')[['PD1_asinh','TCF1_asinh']].mean().rename(columns={'PD1_asinh':'PD1_roi_equal','TCF1_asinh':'TCF1_roi_equal'})
    a=a.merge(re,on='patient_id',validate='one_to_one')
    md=pd.read_csv(INPUTS['metadata'],usecols=['patient_id'])
    total=md.groupby('patient_id').size().rename('n_all_cells')
    a=a.merge(total,on='patient_id',validate='one_to_one');a['CD8_fraction']=a.n_cd8/a.n_all_cells
    d=clinical.merge(a,on='patient_id',validate='one_to_one')
    assert len(d)==280 and not d[['PD1_mean','TCF1_mean','n_cd8']].isna().any().any()
    d.to_csv(OUT/'PATIENT_CD8.csv',index=False)
    t[['source_row_id','patient_id','roi_id','cell_subtype','PD1','TCF1']].to_csv(OUT/'ALL_CD8_USED.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    cov=[]
    for group,g in d.groupby('group'):
        cov.append(dict(group=group,n_patients=len(g),n_cd8_cells=int(g.n_cd8.sum()),median_cd8=float(g.n_cd8.median()),min_cd8=int(g.n_cd8.min()),max_cd8=int(g.n_cd8.max()),n_below20=int(g.n_cd8.lt(20).sum()),n_below50=int(g.n_cd8.lt(50).sum()),median_CD8_fraction=float(g.CD8_fraction.median())))
    pd.DataFrame(cov).to_csv(OUT/'GROUP_COVERAGE.csv',index=False)
    pd.crosstab(d.TMA,d.group).to_csv(OUT/'TMA_BY_GROUP.csv')
    cases=[('all280_mean',d,'mean'),('min20_mean',d[d.n_cd8>=20],'mean'),('min50_mean',d[d.n_cd8>=50],'mean'),('all280_median',d,'median'),('all280_roi_equal',d,'roi_equal')]
    raw=[];adj=[];corr=[]
    for name,g,agg in cases:
        for marker in ['PD1','TCF1']:
            col=marker+'_'+agg
            raw.append({'analysis':name,'marker':marker,'column':col,**compare(g,col)})
            sd=float(g[col].std(ddof=1))
            y=(g[col]-g[col].mean())/sd
            for full in ([False,True] if name=='all280_mean' else [False]):
                for exposure in ['high_HLA','all_mean']:
                    x=design_matrix(g,exposure,full)
                    row=ols_hc3(y,x,1)
                    adj.append({'analysis':name,'marker':marker,'full_CAF_covariates':full,'exposure':exposure,'n_lower':int(g.group.eq('Lower').sum()),'n_higher':int(g.group.eq('Higher').sum()),'endpoint_sd':sd,**row})
            test=stats.spearmanr(g.all_mean,g[col])
            corr.append(dict(analysis=name,marker=marker,n=len(g),rho=float(test.statistic),p=float(test.pvalue)))
    raw=pd.DataFrame(raw);adj=pd.DataFrame(adj);corr=pd.DataFrame(corr)
    for table,keys in [(raw,['analysis']),(adj,['analysis','full_CAF_covariates','exposure']),(corr,['analysis'])]:
        table['q_bh']=np.nan
        for _,ix in table.groupby(keys).groups.items():assert len(ix)==2;table.loc[ix,'q_bh']=bh(table.loc[ix,'p'])
    raw.to_csv(OUT/'GROUP_COMPARISONS.csv',index=False);adj.to_csv(OUT/'ADJUSTED_COMPARISONS.csv',index=False);corr.to_csv(OUT/'CONTINUOUS_ASSOCIATIONS.csv',index=False)
    abundance=pd.DataFrame([dict(endpoint=col,**compare(d,col)) for col in ['n_cd8','CD8_fraction']])
    abundance.to_csv(OUT/'CD8_ABUNDANCE_COMPARISONS.csv',index=False)
    batch=[]
    for tma,g in d.groupby('TMA'):
        for marker in ['PD1','TCF1']:
            batch.append(dict(TMA=tma,marker=marker,**compare(g,marker+'_mean')))
    pd.DataFrame(batch).to_csv(OUT/'TMA_STRATIFIED_COMPARISONS.csv',index=False)
    return d,t,raw,adj,corr

def plot(d,raw,adj):
    fig,axes=plt.subplots(1,2,figsize=(11.8,6.4))
    fig.subplots_adjust(left=.085,right=.97,bottom=.20,top=.73,wspace=.28)
    for ax,marker,label in zip(axes,['PD1','TCF1'],['PD-1','TCF1']):
        row=raw[(raw.analysis=='all280_mean')&(raw.marker==marker)].iloc[0]
        r=adj[(adj.analysis=='all280_mean')&(adj.marker==marker)&~adj.full_CAF_covariates&(adj.exposure=='high_HLA')].iloc[0]
        for i,group in enumerate(['Lower','Higher']):
            g=d[d.group.eq(group)];vals=g[marker+'_mean'].to_numpy()
            x=np.random.default_rng(SEED).uniform(i-.18,i+.18,len(vals))
            ax.scatter(x,vals,s=19,c=COLORS[group],alpha=.62,lw=0)
            lo,med,hi=np.quantile(vals,[.25,.5,.75]);ax.plot([i-.23,i+.23],[med,med],c='#283c4c',lw=2.5)
            ax.plot([i,i],[lo,hi],c='#283c4c',lw=1.7)
        ax.set(xticks=[0,1],xticklabels=['Low vCAF HLA\n140 patients','High vCAF HLA\n140 patients'],xlim=(-.4,1.4),ylim=(0,float(d[marker+'_mean'].max())*1.08),ylabel=f'Patient mean CD8 {label}\nasinh(provider intensity)')
        ax.text(.5,1.17,f'Measured {label} in all CD8 T cells',transform=ax.transAxes,ha='center',fontsize=12)
        ax.grid(axis='y',alpha=.13)
        ax.text(.5,1.055,f'Group comparison q = {row.q_bh:.3g}\nAdjusted difference = {r.coef:+.2f} SD; q = {r.q_bh:.3g}',transform=ax.transAxes,ha='center',fontsize=10)
    fig.suptitle('Whole-patient CD8 expression in the original vCAF HLA survival groups',fontsize=15,y=.967)
    fig.text(.5,.865,'One dot per patient | All available CD8 cells | Same fixed 140 / 140 patient split',ha='center',color='#4e6070',fontsize=11)
    fig.text(.5,.073,'Bars: patient median and interquartile range. Adjusted model: TMA, age, sex and stage; HC3 robust standard errors.',ha='center',fontsize=9.5,color='#4e6070')
    fig.text(.5,.032,'No distance cutoff or nearest-vCAF assignment. PD-1 and TCF1 are measured protein readouts.',ha='center',fontsize=10,color='#4e6070')
    fig.savefig(FIG/'whole_patient_CD8_groups.png',dpi=230);fig.savefig(FIG/'whole_patient_CD8_groups.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.7));fig.subplots_adjust(left=.18,right=.98,top=.80,bottom=.17,wspace=.25)
    case_labels={'all280_mean':'All 280: cell mean','min20_mean':'At least 20 CD8','min50_mean':'At least 50 CD8','all280_median':'All 280: cell median','all280_roi_equal':'All 280: equal ROI means'}
    for ax,marker,label in zip(axes,['PD1','TCF1'],['PD-1','TCF1']):
        for i,name in enumerate(case_labels):
            row=adj[(adj.analysis==name)&(adj.marker==marker)&~adj.full_CAF_covariates&(adj.exposure=='high_HLA')].iloc[0]
            ax.errorbar(row.coef,i,xerr=[[row.coef-row.ci_low],[row.ci_high-row.coef]],fmt='o',color='#167d8d',capsize=3)
        ax.set(yticks=np.arange(5),yticklabels=list(case_labels.values()) if marker=='PD1' else [],xlabel='Adjusted high − low difference (SD)',title=label)
        ax.invert_yaxis();ax.axvline(0,color='#999fa6',ls='--',lw=1);ax.grid(axis='x',alpha=.12)
    fig.suptitle('Patient-level CD8 comparisons across count and summary sensitivities',fontsize=15,y=.95)
    fig.text(.5,.055,'Original patient labels retained in every analysis. Models adjust for TMA, age, sex and stage; 95% HC3 confidence intervals.',ha='center',fontsize=10,color='#4e6070')
    fig.savefig(FIG/'whole_patient_CD8_sensitivity.png',dpi=230);fig.savefig(FIG/'whole_patient_CD8_sensitivity.svg');plt.close(fig)

def report(d,raw,adj,corr):
    lines=['# 按原生存分组比较整位患者的 CD8 表达','',
      '问题：vCAF 平均预测 HLA-ABC 得分较高的患者，其全部 CD8 T 细胞的实测 PD-1／TCF1 是否不同？这里不使用空间距离或最近 vCAF 分配。原最近邻分析保持不变。','',
      '## 固定设计','',
      '- 保留原来的 280 位 DFS 患者，阈值 0.666938735739868；低组 140、高组 140。所有患者都有作者标注的 CD8。','- 读取原作者 counts，逐个细胞作 asinh(x)，再对患者全部可用 core 的全部 CD8 求均值。每位患者一个观测，组间统计患者等权。','- 不使用患者内 CD8 百分位秩来做患者均值，因为那会将每个人的平均值固定在约 0.5。','- 主要终点仅 PD-1 和 TCF1；双侧 Mann–Whitney U，并在两个终点间作 BH 校正。','- 调整模型：TMA 固定效应、年龄、性别、分期；OLS 的 HC3 稳健标准误。差异按终点患者间标准差表示。原 CAF 协变量进一步调整、连续 HLA 分数、细胞中位数和 ROI 等权均值均保留为伴随分析。','',
      '## 主要结果','',
      '| 标记 | 低组平均值 | 高组平均值 | 未调整 q | 调整后高−低差异（SD）及 95% CI | 调整 q |','|---|---:|---:|---:|---|---:|']
    for marker in ['PD1','TCF1']:
        a=raw[(raw.analysis=='all280_mean')&(raw.marker==marker)].iloc[0]
        b=adj[(adj.analysis=='all280_mean')&(adj.marker==marker)&~adj.full_CAF_covariates&(adj.exposure=='high_HLA')].iloc[0]
        lines.append(f'| {marker} | {a.lower_mean:.4f} | {a.higher_mean:.4f} | {a.q_bh:.4g} | {b.coef:+.3f} ({b.ci_low:+.3f}–{b.ci_high:+.3f}) | {b.q_bh:.4g} |')
    lines+=['','均值单位为 asinh(provider counts)，不是荧光阳性比例。','',
      '本轮的主要发现是高 vCAF HLA 组的 CD8 TCF1 较低，调整后差异约为 −0.70 SD。在至少 20／50 个 CD8、用细胞中位数、对 ROI 等权平均和进一步调整原 CAF 协变量时，TCF1 差异保持同方向。六个 TMA 内 TCF1 高−低差异也均为负。','',
      'PD-1 同样较低，方向与预设的 PD-1 升高假设相反；其差异在至少 50 个 CD8 的敏感性分析中不明确。因此结果支持“较低 TCF1 的患者整体 CD8 状态”，不支持直接描述为 PD-1 高／TCF1 低的典型耗竭表型。','',
      '## CD8 数量与敏感性','',
      '| 分组 | 患者数 | CD8 细胞总数 | 每人 CD8 中位数 | 少于 20 个 CD8 的患者 |','|---|---:|---:|---:|---:|']
    coverage=pd.read_csv(OUT/'GROUP_COVERAGE.csv')
    for _,r in coverage.iterrows():lines.append(f'| {r.group} | {r.n_patients} | {r.n_cd8_cells} | {r.median_cd8:.0f} | {r.n_below20} |')
    lines+=['','下列分析只改变 CD8 表达估计的最低细胞数或汇总方式，始终保留原患者高低组标签，不重新切分：','',
      '| 分析 | 标记 | 低／高组人数 | 调整差异（SD） | 95% CI | q |','|---|---|---|---:|---|---:|']
    for _,r in adj[(adj.exposure=='high_HLA')&~adj.full_CAF_covariates].iterrows():lines.append(f'| {r.analysis} | {r.marker} | {r.n_lower}/{r.n_higher} | {r.coef:+.3f} | {r.ci_low:+.3f}–{r.ci_high:+.3f} | {r.q_bh:.4g} |')
    lines+=['','## 解释范围','',
      '最近 vCAF 的唯一归属是一种局部几何模型：二维质心最近不保证实际接触，也不能涵盖多个 CAF 的共同作用。当前患者分组比较绕开了这个假设，回答患者整体 CD8 状态差异。两种分析的层次不同，不能根据一种结果否定或替代另一种结果。','',
      'PD-1 和 TCF1 都不能单独证明耗竭或功能损伤。尤其不能将一项方向符合而另一项不符合的结果组合成“已证明 CD8 耗竭”。相关背景：[肺癌 CD8 功能与 PD-1／TCF1 研究](https://pubmed.ncbi.nlm.nih.gov/32332901/)。','',
      '数据覆盖患者可用的组织芯片 core，不能代表整块肿瘤的全部细胞。TMA 调整用于处理不同切片批次之间的差异，不能保证移除所有患者间测量差异。','',
      '## 文件','',
      '- results/PATIENT_CD8.csv：每患者的 HLA 分组、CD8 汇总、细胞数和协变量。','- results/GROUP_COMPARISONS.csv：未调整组间比较。','- results/ADJUSTED_COMPARISONS.csv：主调整、原 CAF 协变量进一步调整及连续 HLA 的全部模型。','- results/TMA_STRATIFIED_COMPARISONS.csv：每个 TMA 内的方向和差异。','- results/figures/whole_patient_CD8_groups.png / .svg：原 140／140 分组的患者散点图。','- results/figures/whole_patient_CD8_sensitivity.png / .svg：最低细胞数与汇总方式敏感性。','- results/DESIGN.json / VERIFICATION.json：方案、分组、原始读数和文件哈希核对。','',
      '复现：`/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -B analyze.py`。只在本 patient_cd8 子目录写入，前一轮分子与邻域分析及其全部输入只读。','']
    (BASE/'README_ZH.md').write_text('\n'.join(lines))

def verify(hashes,d,t,raw,adj):
    assert hashes=={key:sha(p) for key,p in INPUTS.items()}
    # Independent exact group reconstruction and three direct patient summaries.
    clinical=pd.read_csv(INPUTS['clinical'],float_precision='round_trip').set_index('patient_id')
    cut=json.loads((OUT/'DESIGN.json').read_text())['patient_cutoff']
    assert np.array_equal(d.high_HLA.to_numpy(),(clinical.loc[d.patient_id,'all_mean'].to_numpy()>=cut).astype(int))
    for pid in d.sample(n=3,random_state=42).patient_id:
        g=t[t.patient_id.eq(pid)];row=d[d.patient_id.eq(pid)].iloc[0]
        for marker in ['PD1','TCF1']:assert abs(np.arcsinh(g[marker]).sum()/len(g)-row[marker+'_mean'])<1e-12
    # Reconstruct one HC3 variance using the SVD pseudoinverse formulation.
    x=design_matrix(d).to_numpy();y=(d.PD1_mean-d.PD1_mean.mean())/d.PD1_mean.std(ddof=1)
    pinv=np.linalg.pinv(x);b=pinv@y;res=y-x@b;hat=np.sum(x*pinv.T,axis=1)
    cov=(pinv*(res/(1-hat)).to_numpy()[None,:])@(pinv*(res/(1-hat)).to_numpy()[None,:]).T
    a=adj[(adj.analysis=='all280_mean')&(adj.marker=='PD1')&~adj.full_CAF_covariates&(adj.exposure=='high_HLA')].iloc[0]
    assert abs(a.coef-b[1])<1e-10 and abs(a.se-np.sqrt(cov[1,1]))<1e-10
    figs={}
    for p in FIG.glob('*.png'):
        with Image.open(p) as im:assert np.asarray(im.convert('RGB')).std()>10;figs[p.name]={'size':list(im.size),'nonblank':True}
    save(OUT/'VERIFICATION.json',{'status':'PASS','time_utc':now(),'source_hashes_unchanged':True,'same_original_140_140_groups':True,'no_nearest_assignment_used':True,'all_CD8_cells_in_280_patients':len(t),'patient_mean_and_HC3_independently_recomputed':True,'figures':figs,'input_sha256':hashes})
    files=[p for p in BASE.rglob('*') if p.is_file() and not any(part in {'cache','tmp','logs','__pycache__'} for part in p.relative_to(BASE).parts) and p.name!='MANIFEST.json']
    save(OUT/'MANIFEST.json',{'time_utc':now(),'output_sha256':{str(p.relative_to(BASE)):sha(p) for p in files}})

def main():
    hashes,clinical=freeze();d,t,raw,adj,corr=analyze(clinical)
    print('Unadjusted patient comparisons:\n',raw[raw.analysis=='all280_mean'].to_string(index=False),flush=True)
    print('Adjusted patient comparisons:\n',adj[(adj.exposure=='high_HLA')&~adj.full_CAF_covariates].to_string(index=False),flush=True)
    plot(d,raw,adj);report(d,raw,adj,corr);verify(hashes,d,t,raw,adj)
    print('Completed patient CD8 comparison and verification.',flush=True)
if __name__=='__main__':main()
