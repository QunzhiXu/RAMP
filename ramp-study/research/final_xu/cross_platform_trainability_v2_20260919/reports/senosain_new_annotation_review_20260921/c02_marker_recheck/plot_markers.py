"""Direct provider-expression review; writes only beside this script."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/senosain_marker_recheck_mpl')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import hashlib,json,html
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pypdf import PdfReader
import fitz

HERE=Path(__file__).resolve().parent
PREV=HERE.parent
FX=HERE.parents[3]
ANN=FX/'raw_data/lung_cancer/Senosain_2025_CyTOF_and_omics/annotation'
HASH={}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def track(p):HASH[str(p)]=sha(p);return p
def save(name,d):d.to_csv(HERE/name,index=False)
def sx(v):return np.arcsinh(np.asarray(v)/5.)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,
    'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False})
a=pd.read_csv(track(ANN/'rough_annotation/senosain_140k_final_cell_annotations.csv.gz'),dtype={'patient_id':str},keep_default_na=False)
m=pd.read_csv(track(ANN/'senosain_140k_markers35.csv'));markers=m.marker.tolist();mi={v:i for i,v in enumerate(markers)}
x=np.load(track(ANN/'senosain_140k_raw35.npy'),mmap_mode='r')
meta=pd.read_csv(track(ANN/'senosain_140k_cell_metadata.csv.gz'),dtype={'patient_id':str})
assert a.source_cell_id.equals(meta.source_cell_id) and len(a)==140000 and x.shape==(140000,35)
a['cluster']=a.parent_cluster+np.where(a.subcluster.ne(''),'-'+a.subcluster,'')
donors=pd.read_csv(track(FX/'story_lung_HLA_ABC/data_umap/data/SELECTED_DONORS.csv'),dtype={'patient_id':str})
story=set(donors.loc[donors.dataset.eq('Senosain CyTOF')&donors.selected.astype(str).str.lower().eq('true'),'patient_id'])
a['original_story_patient']=a.patient_id.isin(story)
th=pd.read_csv(track(PREV/'DIAGNOSTIC_THRESHOLDS.csv'),dtype={'patient_id':str})
th=th[th.alpha.eq(.5)].set_index(['patient_id','marker'])
support=pd.read_csv(track(PREV/'PATIENT_CLUSTER_COEXPRESSION.csv'),dtype={'patient_id':str})
support=support[(support.alpha==.5)&support.cluster.eq('C02')].sort_values(['Myeloid_support','patient_id'])
selected=support[support.patient_id.isin(story)].reset_index(drop=True)
chosen=[]
for q in [.1,.5,.9]:
    r=selected.iloc[round(q*(len(selected)-1))]
    chosen.append(dict(patient_id=r.patient_id,quantile_in_original28=q,C02_cells=int(r.n),old_gate_support=float(r.Myeloid_support)))
save('DISPLAY_PATIENT_SELECTION.csv',pd.DataFrame(chosen))

mask=a.parent_cluster.eq('C02').to_numpy()
raw=pd.DataFrame(np.asarray(x[mask]),columns=markers)
raw=pd.concat([a.loc[mask,['source_cell_id','patient_id','cluster','final_cell_type','annotation_confidence','qc_recommendation','original_story_patient']].reset_index(drop=True),raw],axis=1)
assert len(raw)==14963
save('C02_ALL_CELLS_RAW35.csv.gz',raw)
df=pd.DataFrame(np.asarray(x),columns=markers)
df['cluster']=a.cluster;df['patient_id']=a.patient_id
med=df.groupby('cluster')[markers].median()
save('ALL_CLUSTER_NATIVE_MEDIANS.csv',med.reset_index())
pm=df.groupby(['patient_id','cluster'])[markers].median()
pm.insert(0,'n',df.groupby(['patient_id','cluster']).size())
save('PATIENT_CLUSTER_NATIVE_MEDIANS.csv',pm.reset_index())

# Whole-panel heat map: include CD90 in the table, but mask its nonuniform measurement in the image.
order=['CD45','CD3','CD4','CD8','CD11B','CD56','CD31','EPCAM','CYTOKERATIN','CK7','VIMENTIN','CD24',
       'HLA_DR','HLA_ABC','PD_L1','KI67','CCASP3','TP53','THIOREDOXIN','B_CATENIN','HER2','P_STAT6',
       'P_STAT5','TTF1','P_AKT','P_STAT3','MDM2','MET','TP63','EGFR','CD44','P_ERK','P_S6','HISTONE_H3','CD90']
assert set(order)==set(markers)
balanced=pd.DataFrame(sx(pm[markers]),index=pm.index,columns=markers).groupby(level='cluster').median()
# Each patient contributes one cluster median regardless of cluster size.
zz=(balanced-balanced.mean())/balanced.std(ddof=0)
zz['CD90']=np.nan
labels={k:g.final_cell_type.iloc[0] for k,g in a.groupby('cluster')}
fig,ax=plt.subplots(figsize=(17,9),layout='constrained');cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#d3d3d3')
im=ax.imshow(zz[order],aspect='auto',vmin=-2.5,vmax=2.5,cmap=cmap)
ax.set_yticks(range(len(zz)),[f'{c}  {labels[c]}' for c in zz.index]);ax.set_xticks(range(35),[k.replace('CYTOKERATIN','CK') for k in order],rotation=65,ha='right')
y=zz.index.tolist().index('C02');ax.add_patch(Rectangle((-.5,y-.5),35,1,fill=False,color='#111111',lw=2.5))
ax.set_title('All clusters: full-panel expression context (C02 outlined)',loc='left',fontsize=14)
ax.set_xlabel('Within-marker standardized asinh medians; patients equally weighted. CD90 is gray because measurement was not uniform.')
fig.colorbar(im,ax=ax,shrink=.65,label='Marker-wise z score of cluster medians')
fig.savefig(HERE/'ALL_CLUSTER_MARKER_PROFILES.png',dpi=190,bbox_inches='tight');plt.close(fig)

GROUPS={'C02':['C02'],'C09':['C09'],'T cells':['C00','C03'],'Epithelial':['C01','C05'],'Endothelial':['C11']}
COL={'C02':'#CE7D15','C09':'#087D73','T cells':'#4875B4','Epithelial':'#BC5673','Endothelial':'#80719B'}
KEY=['CD45','CD3','CD11B','HLA_DR','CD56','VIMENTIN','EPCAM','CYTOKERATIN','CK7','CD31','CD4','CD8']
tick_candidates=np.array([0,5,50,500,5000,50000,500000],float)
native_group_medians=[]
for name,cl in GROUPS.items():
    v=np.asarray(x[a.parent_cluster.isin(cl)])
    for mark in markers:
        z=v[:,mi[mark]]
        native_group_medians.append(dict(group=name,marker=mark,n=len(v),q10=np.quantile(z,.1),q25=np.quantile(z,.25),median=np.median(z),q75=np.quantile(z,.75),q90=np.quantile(z,.9)))
save('REFERENCE_GROUP_RAW_DISTRIBUTIONS.csv',pd.DataFrame(native_group_medians))

def ecdf_figure(idx,path,title,draw_gate=False):
    fig,axes=plt.subplots(3,4,figsize=(13.5,9),layout='constrained')
    gg=a.iloc[idx]
    for ax,mark in zip(axes.flat,KEY):
        maxv=0
        for name,clusters in GROUPS.items():
            rows=gg.index[gg.parent_cluster.isin(clusters)].to_numpy()
            if len(rows)==0:continue
            vals=np.sort(np.asarray(x[rows,mi[mark]],dtype=float));r=np.linspace(0,len(vals)-1,min(len(vals),700)).round().astype(int)
            maxv=max(maxv,float(vals[-1]));ax.plot(sx(vals[r]),(r+1)/len(vals),c=COL[name],lw=2 if name=='C02' else 1.3,alpha=1 if name=='C02' else .8)
        if draw_gate:
            patient=gg.patient_id.iloc[0];t=th.loc[(patient,mark)]
            native=float(t.cofactor*np.sinh(t.threshold_arcsinh));ax.axvline(sx(native),c='#777777',ls=':',lw=1)
        ticks=tick_candidates[tick_candidates<=maxv]
        if len(ticks)>7:ticks=ticks[np.linspace(0,len(ticks)-1,7).round().astype(int)]
        ax.set_xticks(sx(ticks),[f'{v:g}' for v in ticks],rotation=35,ha='right',fontsize=8)
        ax.set_xlim(0,sx(maxv)*1.02);ax.set_ylim(0,1.02);ax.set_title(mark.replace('CYTOKERATIN','CK'))
        ax.grid(alpha=.15);ax.set_ylabel('Cumulative cell fraction')
    handles=[Line2D([0],[0],c=COL[g],lw=2,label=f'{g} (n={gg.parent_cluster.isin(cl).sum():,})') for g,cl in GROUPS.items()]
    if draw_gate:handles.append(Line2D([0],[0],c='#777777',ls=':',label='Previous diagnostic cutoff'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,1.02),ncol=3,frameon=False,fontsize=9)
    fig.suptitle(title,y=1.12,fontweight='bold',fontsize=13)
    fig.supxlabel('Provider intensity; asinh(value / 5) display. Curves use all cells in each group.',fontsize=10)
    fig.savefig(path,dpi=145,bbox_inches='tight');plt.close(fig)

ecdf_figure(np.arange(len(a)),HERE/'C02_RAW_MARKER_DISTRIBUTIONS.png','C02 versus lineage reference groups: no hard positivity gates')

# Continuous comparisons and raw median differences within every patient.
rows=[]
for patient,g in a.groupby('patient_id'):
    for name,clusters in list(GROUPS.items())[1:]:
        ps=g.parent_cluster.eq('C02').to_numpy();ns=g.parent_cluster.isin(clusters).to_numpy();use=ps|ns
        if min(ps.sum(),ns.sum())<20:continue
        ind=g.index.to_numpy()[use]
        for marker in KEY:
            v=np.asarray(x[ind,mi[marker]],float)
            rows.append(dict(patient_id=patient,original_story_patient=patient in story,reference=name,marker=marker,
                n_C02=int(ps.sum()),n_reference=int(ns.sum()),
                auc_C02_higher=roc_auc_score(ps[use],v),median_C02=np.median(v[ps[use]]),median_reference=np.median(v[ns[use]])))
ev=pd.DataFrame(rows);save('ALL_PATIENT_CONTINUOUS_COMPARISONS.csv',ev)
ss=ev.groupby(['reference','marker']).agg(patients=('patient_id','size'),median_auc=('auc_C02_higher','median'),
    min_auc=('auc_C02_higher','min'),max_auc=('auc_C02_higher','max'),fraction_patients_C02_higher=('auc_C02_higher',lambda z:(z>.5).mean())).reset_index()
save('CONTINUOUS_COMPARISON_SUMMARY.csv',ss)
fig,axes=plt.subplots(2,3,figsize=(13,7.5),layout='constrained')
tests=[('CD45','Epithelial'),('CD3','T cells'),('CD11B','T cells'),('HLA_DR','T cells'),('EPCAM','Epithelial'),('CK7','Epithelial')]
for ax,(mark,ref) in zip(axes.flat,tests):
    z=ev[(ev.marker==mark)&ev.reference.eq(ref)]
    ax.scatter(sx(z.median_reference),sx(z.median_C02),c=np.where(z.original_story_patient,'#CE7D15','#9CABB8'),s=25,alpha=.85,edgecolors='white',linewidths=.3)
    hi=float(max(z.median_C02.max(),z.median_reference.max()))*1.15;lo=0
    ax.plot([0,sx(hi)],[0,sx(hi)],c='#aaaaaa',ls=':',lw=1)
    ticks=tick_candidates[tick_candidates<=hi]
    if len(ticks)>6:ticks=ticks[np.linspace(0,len(ticks)-1,6).round().astype(int)]
    ax.set_xticks(sx(ticks),[f'{v:g}' for v in ticks]);ax.set_yticks(sx(ticks),[f'{v:g}' for v in ticks])
    ax.set_xlim(0,sx(hi));ax.set_ylim(0,sx(hi));ax.set_xlabel(f'{ref}: patient median');ax.set_ylabel('C02: patient median')
    ax.set_title(f'{mark}: {len(z)} patients');ax.grid(alpha=.12)
fig.legend(handles=[Line2D([0],[0],marker='o',c='#CE7D15',lw=0,label='Original 28 patients'),Line2D([0],[0],marker='o',c='#9CABB8',lw=0,label='Other patients')],loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2,frameon=False)
fig.suptitle('Patient-matched native expression: reference choice matters',y=1.09,fontweight='bold',fontsize=14)
fig.supxlabel('Axes show provider values on an asinh(value / 5) display; each dot is one patient.')
fig.savefig(HERE/'PATIENT_MATCHED_RAW_MEDIANS.png',dpi=180,bbox_inches='tight');plt.close(fig)

# Direct coexpression: deterministic low/median/high support examples, not best performance examples.
fig,axes=plt.subplots(3,3,figsize=(12,11),layout='constrained')
axes_markers=[('CD45','CD11B'),('CD3','HLA_DR'),('EPCAM','CK7')]
for rr,c in enumerate(chosen):
    patient=c['patient_id'];g=a[a.patient_id.eq(patient)]
    for ax,(mx,my) in zip(axes[rr],axes_markers):
        xx=[];yy=[]
        for name in ['Epithelial','Endothelial','T cells','C09','C02']:
            ind=g.index[g.parent_cluster.isin(GROUPS[name])].to_numpy()
            vx=np.asarray(x[ind,mi[mx]],float);vy=np.asarray(x[ind,mi[my]],float);xx.extend(vx);yy.extend(vy)
            ax.scatter(sx(vx),sx(vy),c=COL[name],s=11 if name=='C02' else 6,alpha=.7 if name=='C02' else .3,lw=0,rasterized=True)
        for mark,axis in [(mx,'x'),(my,'y')]:
            t=th.loc[(patient,mark)];v=float(t.cofactor*np.sinh(t.threshold_arcsinh))
            (ax.axvline if axis=='x' else ax.axhline)(sx(v),c='#555555',ls=':',lw=.9)
        for vals,axis in [(xx,'x'),(yy,'y')]:
            hi=max(vals)*1.05;ticks=tick_candidates[tick_candidates<=hi]
            if len(ticks)>6:ticks=ticks[np.linspace(0,len(ticks)-1,6).round().astype(int)]
            getattr(ax,f'set_{axis}ticks')(sx(ticks),[f'{v:g}' for v in ticks],fontsize=8)
            getattr(ax,f'set_{axis}lim')(0,sx(hi))
        ax.set_xlabel(mx);ax.set_ylabel(my);ax.set_title(f"Patient {patient} | C02 n={c['C02_cells']}",fontsize=10)
fig.legend(handles=[Line2D([0],[0],marker='o',c=COL[g],lw=0,label=g) for g in GROUPS],loc='lower center',bbox_to_anchor=(.5,1.02),ncol=5,frameon=False)
fig.suptitle('Individual cells: C02 compared with lineage references',y=1.08,fontweight='bold',fontsize=14)
fig.supxlabel('Native-value axes with asinh display. Dotted lines: prior automatic diagnostic cutoffs, not validated gates.')
fig.savefig(HERE/'C02_PATIENT_COEXPRESSION.png',dpi=170,bbox_inches='tight');plt.close(fig)

# Read-only paper inspection: methods page and Figure 2.
paper=track(ANN.parent/'Senosain_2023.pdf');reader=PdfReader(str(paper));d=fitz.open(paper)
pdest=HERE/'paper_reference';pdest.mkdir(exist_ok=True)
for page in [2,5]:
    (pdest/f'page_{page+1:02d}.txt').write_text(reader.pages[page].extract_text())
    d[page].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(str(pdest/f'page_{page+1:02d}.png'))

dest=HERE/'patients';dest.mkdir(exist_ok=True)
index=[]
for n,(patient,g) in enumerate(a.groupby('patient_id',sort=True),1):
    path=dest/f'patient_{patient}.png'
    ecdf_figure(g.index.to_numpy(),path,f'Patient {patient}: raw marker distributions',True)
    sr=support[support.patient_id.eq(patient)].iloc[0]
    index.append(dict(patient_id=patient,original_story_patient=patient in story,C02_cells=int(sr.n),old_gate_support=float(sr.Myeloid_support),image=f'patients/{path.name}'))
    if n%10==0:print(f'Patient gallery: {n}/70',flush=True)
save('PATIENT_GALLERY_INDEX.csv',pd.DataFrame(index))
ht=['<!doctype html><meta charset="utf-8"><title>C02 marker review</title><style>body{font:16px system-ui;margin:30px;max-width:1450px}img{max-width:100%}td,th{padding:8px;text-align:left}tr:nth-child(even){background:#f4f6f8}table{border-collapse:collapse}</style>',
    '<h1>C02 marker review</h1><p>Read-only expression review. Every patient has a distribution plot. Dotted lines are previous diagnostic cutoffs, not validated positive/negative gates.</p>']
for f,title in [('ALL_CLUSTER_MARKER_PROFILES.png','All-cluster panel'),('C02_RAW_MARKER_DISTRIBUTIONS.png','Raw expression distributions'),('PATIENT_MATCHED_RAW_MEDIANS.png','Patient-matched native medians'),('C02_PATIENT_COEXPRESSION.png','Cell-level coexpression')]:
    ht.append(f'<h2>{title}</h2><a href="{f}"><img src="{f}"></a>')
ht.append('<h2>All 70 patients</h2><table><tr><th>Patient</th><th>Original 28</th><th>C02 cells</th><th>Old diagnostic support</th><th>Raw expression</th></tr>')
for r in index:ht.append(f'<tr><td>{r["patient_id"]}</td><td>{r["original_story_patient"]}</td><td>{r["C02_cells"]}</td><td>{r["old_gate_support"]:.1%}</td><td><a href="{r["image"]}">12 markers</a></td></tr>')
ht.append('</table>');(HERE/'index.html').write_text('\n'.join(ht))
assert all(sha(p)==h for p,h in HASH.items())
(HERE/'MANIFEST.json').write_text(json.dumps(dict(n_cells=len(a),C02_cells=int(mask.sum()),patients=a.patient_id.nunique(),
    inputs_sha256=HASH,script_sha256=sha(__file__),source_files_unchanged=True,labels_changed=False,
    display_transform='arcsinh(provider_value/5)',models_fitted=False),indent=2)+'\n')
print('Complete; original data and labels unchanged.',flush=True)
