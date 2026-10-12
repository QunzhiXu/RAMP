"""Plot current-model spatial patterns, with fixed scores and unchanged coordinates."""
import runtime as rt
from runtime import BASE
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from pathlib import Path

OUT=BASE/'results'; FIG=OUT/'figures'; FIG.mkdir(exist_ok=True)
MAP=FIG/'maps';MAP.mkdir(exist_ok=True)
PRIMARY=['T cell','B cell','Myeloid','Neutrophil','Tumour','Vessel','Other CAF']
HIGH='#d36645';LOW='#217992'
PALETTE={'Tumour':'#cf5748','T cell':'#258671','B cell':'#6686cd','Myeloid':'#d0a02e',
         'Neutrophil':'#a78a50','Other CAF':'#c35199','Vessel':'#46aab6','vCAF':'#383d49','Other':'#c8ccd1'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'svg.fonttype':'none','pdf.fonttype':42})

def save(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=220,facecolor='white')
    fig.savefig(path.with_suffix('.svg'),facecolor='white')
    plt.close(fig)

def quantitative():
    s=pd.read_csv(OUT/'HIGH_LOW_SUMMARY.csv')
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.4))
    order=['T cell','CD8 T cell','B cell','Myeloid','Neutrophil','Tumour','Vessel','Other CAF'][::-1]
    for ax,pop,title in zip(axes,['all_available','DFS_280_members'],['All available patients','Within the DFS cohort']):
        for k,col,dy in [(30,'#277d8e',.09),(100,'#9c9fa4',-.09)]:
            a=s[(s.population==pop)&(s.config==f'RAMP_k{k}')&(s.metric=='proximity_fraction')&(s.radius==40)].set_index('target')
            for j,t in enumerate(order):
                row=a.loc[t];ax.errorbar(row['mean']*100,j+dy,xerr=[[100*(row['mean']-row.ci_low)],[100*(row.ci_high-row['mean'])]],fmt='o',color=col,ms=5,capsize=2,lw=1.4)
            if k==30:n=int(a.loc['T cell','n_patients'])
        ax.axvline(0,color='.4',ls='--',lw=.8);ax.grid(axis='x',alpha=.15)
        ax.set_yticks(range(len(order)),order);ax.set_xlim(-6,6);ax.set_xlabel('High minus low proximity (percentage points)')
        ax.set_title(f'{title} | k30: {n} patients',fontsize=11)
    axes[0].legend(handles=[Line2D([],[],marker='o',color='#277d8e',ls='',label='k30'),Line2D([],[],marker='o',color='#9c9fa4',ls='',label='k100')],frameon=False,loc='lower right')
    fig.suptitle('Neighborhood differences of high- and low-state vCAFs',fontsize=15,y=.97)
    fig.subplots_adjust(left=.1,right=.98,bottom=.2,top=.82,wspace=.37)
    fig.text(.5,.045,'Nearby = centroid distance <40 provider units, within the same ROI. Fixed HLA-state cutoff.\nROI contrasts averaged within patient; patients weighted equally. Bars: 95% patient-bootstrap CI.',ha='center',fontsize=9,color='.35')
    save(fig,FIG/'vCAF_neighborhood_contrasts')

    p=pd.read_csv(OUT/'PATIENT_CLUSTERING.csv');r=pd.read_csv(OUT/'ROI_CLUSTERING.csv')
    p=p[(p.population=='all_available')&(p.config=='RAMP_k30')];r=r[r.config=='RAMP_k30']
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.6))
    axes[0].hist(p.moran_excess,bins=np.linspace(-.5,.85,28),color='#428b9d',edgecolor='white',lw=.6)
    axes[0].axvline(0,color='.3',ls='--',lw=1)
    axes[0].set(xlabel='Spatial correlation above random expectation',ylabel='Patients',title=f'Continuous HLA state | {len(p)} patients')
    axes[0].text(.98,.96,f'{(p.moran_excess>0).mean():.1%} above expectation',ha='right',va='top',transform=axes[0].transAxes)
    sub=p[p.same_state_excess.notna()]
    axes[1].hist(sub.same_state_excess*100,bins=np.linspace(-25,50,26),color='#cf876a',edgecolor='white',lw=.6)
    axes[1].axvline(0,color='.3',ls='--',lw=1)
    axes[1].set(xlabel='Excess same-state neighbors (percentage points)',ylabel='Patients',title=f'Fixed high/low states | {len(sub)} patients')
    axes[1].text(.98,.96,f'Mean +{100*sub.same_state_excess.mean():.1f} pp',ha='right',va='top',transform=axes[1].transAxes)
    fig.suptitle('Spatial organization of predicted HLA-ABC in vCAFs',fontsize=14,y=.99)
    fig.subplots_adjust(left=.07,right=.98,bottom=.24,top=.82,wspace=.27)
    fig.text(.5,.04,'Current three-cohort transfer, k=30. Six-nearest-vCAF graph restricted to <40 provider units.\nROIs averaged within patient. Random expectation preserves the number of cells in each state.',ha='center',fontsize=8.5,color='.35')
    save(fig,FIG/'vCAF_spatial_clustering')

def display_category(c):
    lab=np.full(len(c),'Other',dtype=object)
    for cat in ['Tumour','T cell']:lab[c.cell_category.eq(cat)]=cat
    lab[c.cell_category.eq('Fibroblast')]='Other CAF';lab[c.cell_category.eq('vessel')]='Vessel'
    lab[c.cell_type.eq('Bcell')]='B cell';lab[c.cell_type.eq('Myeloid')]='Myeloid';lab[c.cell_type.eq('Neutrophil')]='Neutrophil'
    lab[c.cell_type.eq('vCAF')]='vCAF'
    return lab

def clean_axis(ax,c):
    pad=12
    ax.set_xlim(c.x.min()-pad,c.x.max()+pad);ax.set_ylim(c.y.max()+pad,c.y.min()-pad)
    ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
    for sp in ax.spines.values():sp.set_visible(False)
    x=c.x.min()+10;y=c.y.max()-10
    ax.plot([x,x+100],[y,y],color='#38404b',lw=2)
    ax.text(x,y-7,'100 coordinate units',fontsize=7,va='bottom',color='#38404b')

def map_one(c,v,row,path):
    fig,axes=plt.subplots(1,3,figsize=(16,6.1))
    labels=display_category(c)
    for lab in ['Other','Tumour','Other CAF','Myeloid','Neutrophil','T cell','B cell','Vessel','vCAF']:
        mask=labels==lab
        axes[0].scatter(c.loc[mask,'x'],c.loc[mask,'y'],s=6 if lab!='vCAF' else 20,c=PALETTE[lab],lw=0,alpha=.75 if lab!='vCAF' else 1,rasterized=True)
    for ax in axes[1:]:ax.scatter(c.x,c.y,s=3,c='#d9dde2',alpha=.5,lw=0,rasterized=True)
    sc=axes[1].scatter(v.x,v.y,c=v.RAMP_k30,cmap='viridis',vmin=0,vmax=1,s=26,edgecolors='white',linewidths=.3,rasterized=True)
    for state,col,mark in [(True,HIGH,'o'),(False,LOW,'^')]:
        q=v[v.high_k30==state]
        axes[2].scatter(q.x,q.y,s=30,c=col,marker=mark,edgecolors='white',linewidths=.3,rasterized=True)
    for ax,title in zip(axes,['Author cell types','Predicted HLA-ABC in vCAFs','High- and low-state vCAFs']):
        clean_axis(ax,c);ax.set_title(title,fontsize=12,pad=12)
    axes[0].legend(handles=[Line2D([],[],marker='o',ls='',color=PALETTE[t],label=t,markersize=5) for t in ['Tumour','T cell','B cell','Myeloid','Neutrophil','Other CAF','Vessel','vCAF']],
                   loc='upper center',bbox_to_anchor=(.5,-.04),ncol=4,frameon=False,fontsize=7.7,columnspacing=.8,handletextpad=.3)
    axes[2].legend(handles=[Line2D([],[],marker='o',ls='',color=HIGH,label=f'High (n={int(row.n_high)})'),Line2D([],[],marker='^',ls='',color=LOW,label=f'Low (n={int(row.n_low)})')],
                   loc='upper center',bbox_to_anchor=(.5,-.04),ncol=2,frameon=False,fontsize=9)
    ca=fig.add_axes([.40,.16,.2,.022]);cb=fig.colorbar(sc,cax=ca,orientation='horizontal',ticks=[0,.25,.5,.75,1]);cb.set_label('Transferred relative HLA-ABC score',fontsize=8);cb.ax.tick_params(labelsize=8)
    fig.suptitle(f'{row.patient_id} | {row.roi_id} | {int(row.n_vcaf)} vCAFs | Moran I = {row.moran_I:.3f}',fontsize=16,y=.96)
    fig.subplots_adjust(left=.025,right=.985,top=.83,bottom=.23,wspace=.08)
    fig.text(.5,.025,'Same ROI, cells and coordinates. Current three-dataset RAMP k30 prediction; no spatial smoothing.\nHigh state: predicted score >=0.43957. This is a cell-state threshold, separate from the patient DFS grouping.',ha='center',fontsize=9,color='.35')
    save(fig,path)

def main():
    quantitative()
    selected=pd.read_csv(OUT/'MAP_SELECTION.csv',dtype={'patient_id':str,'roi_id':str})
    ctx=pd.read_csv(OUT/'MAP_CONTEXT.csv.gz',dtype={'patient_id':str,'roi_id':str})
    v=pd.read_csv(OUT/'VCAF_CURRENT_PREDICTIONS_AND_DISTANCES.csv.gz',dtype={'patient_id':str,'roi_id':str})
    names=[]
    for row in selected.drop_duplicates(['patient_id','roi_id']).itertuples():
        c=ctx[(ctx.patient_id==row.patient_id)&(ctx.roi_id==row.roi_id)]
        q=v[(v.patient_id==row.patient_id)&(v.roi_id==row.roi_id)]
        assert len(q)==row.n_vcaf and np.array_equal(np.sort(c[c.cell_type=='vCAF'].source_row_id),np.sort(q.source_row_id))
        name=f'{row.patient_id}__{row.roi_id}'.replace(',','_')
        map_one(c,q,row,MAP/name);names.append((name,row))
    # A small gallery preserves every selected map and the reason it was selected.
    html=['<!doctype html><html><meta charset="utf-8"><title>Current-model vCAF spatial patterns</title><body style="font-family:Arial;max-width:1500px;margin:auto"><h1>Current three-dataset vCAF HLA spatial patterns</h1><p>k30, fixed predictions and original coordinates. Examples selected for balanced cell coverage or strong Moran I, as recorded below.</p>']
    for name,row in names:html.append(f'<h2>{row.patient_id} / {row.roi_id}</h2><p>{row.selection}</p><img style="width:100%" src="figures/maps/{name}.png">')
    html.append('</body></html>');(OUT/'GALLERY.html').write_text('\n'.join(html))
    allfig=list(FIG.rglob('*.png'))
    from PIL import Image
    sizes={}
    for p in allfig:
        with Image.open(p) as im:im.load();sizes[str(p.relative_to(BASE))]=list(im.size)
    rt.save_json(OUT/'PLOT_VERIFICATION.json',dict(png_count=len(allfig),png_decoding='pass',sizes=sizes,manual_visual_review='pending',script_sha256=rt.sha(__file__)))
    print(f'Created {len(allfig)} figures plus SVGs and GALLERY.html',flush=True)
if __name__=='__main__':main()
