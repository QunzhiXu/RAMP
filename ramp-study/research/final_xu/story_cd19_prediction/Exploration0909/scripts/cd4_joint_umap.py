"""Three-platform UMAP of the same CD4-excluded alignment; no NN refit."""
from runtime import BASE,STORY,save,sha,now
import cd4_nn_umap_validation as validation
import numpy as np,pandas as pd,json,time
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PARENT=validation.OUT
OUT=PARENT/'joint_umap'
OUT.mkdir(exist_ok=True)

def main():
    start=time.time()
    d,md=validation.load_inputs()
    q=md.dataset.eq('BM CODEX').to_numpy()
    ap=OUT/'ramp13_all.npy'
    if ap.exists(): aligned=np.load(ap)
    else:
        print('Reproduce the same 13-marker alignment, retaining all platforms',flush=True)
        fit=validation.ramp.fit_transform(d[validation.FEATURES].to_numpy(float),
            md[['dataset','unit_key']],validation.FEATURES,datasets=validation.DATASETS,
            family='bm_nn_cd4_permanent_holdout')
        assert fit.strict_rank_inversions==0
        aligned=fit.aligned
        np.save(ap,aligned)
    previous=np.load(PARENT/'ramp13_codex.npy')
    assert np.array_equal(aligned[q],previous),'Joint map must use exactly the same alignment'
    md.to_csv(OUT/'all_metadata.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    up=OUT/'umap_all.npy'
    if up.exists(): xy=np.load(up)
    else:
        import umap
        print('Fit joint UMAP on all 88031 cells; fixed seed 42, CD4/CD19 excluded',flush=True)
        init=np.random.default_rng(42).normal(0,1e-4,size=(len(md),2)).astype(np.float32)
        x=StandardScaler().fit_transform(aligned).astype(np.float32)
        xy=umap.UMAP(n_neighbors=30,min_dist=.15,metric='euclidean',random_state=42,
            transform_seed=42,n_jobs=1,low_memory=True,init=init).fit_transform(x).astype(np.float32)
        np.save(up,xy)
    assert xy.shape==(88031,2) and np.isfinite(xy).all()
    # Coordinates are now frozen; attach labels and fixed predictions for display.
    all_labels=pd.read_csv(STORY/'alignment/output/cell_metadata.csv.gz')
    all_labels=md.merge(all_labels,on=validation.ID,validate='one_to_one',sort=False)
    assert all_labels[validation.ID].astype(str).equals(md.astype(str))
    cells=pd.read_csv(PARENT/'cell_predictions_and_umap.csv.gz')
    assert cells[validation.ID].astype(str).equals(md.loc[q].reset_index(drop=True).astype(str))
    cells[['UMAP1','UMAP2']]=xy[q]
    cells.to_csv(OUT/'codex_predictions_on_joint_umap.csv.gz',index=False,
                 compression={'method':'gzip','mtime':0})
    metrics=json.loads((PARENT/'METRIC_SUMMARY.json').read_text())
    validation.plot(cells,metrics,output_dir=OUT,embedding_label='Joint 3-platform UMAP')
    overview(xy,all_labels)
    analyze_regions(cells,aligned[q],md.loc[q].reset_index(drop=True))
    save(OUT/'AUDIT.json',{'time_utc':now(),'seconds':time.time()-start,
        'n_embedding_cells':88031,'n_displayed_validation_cells':30000,
        'features':validation.FEATURES,'same_codex_aligned_values_bitwise':True,
        'NN_refit':False,'prediction_changed':False,'prediction_sha256':sha(PARENT/'CD4_prediction.npz'),
        'joint_umap_sha256':sha(up),'single_fixed_seed':42,'CD4_or_CD19_used_in_alignment_or_umap':False,
        'labels_used_to_fit':False})
    print(f'Joint UMAP and diagnostics finished in {time.time()-start:.1f}s',flush=True)

def overview(xy,md):
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','font.size':9})
    colors={'BM CODEX':'#3E70B2','Triana AbSeq':'#E28B35','BM CyTOF':'#55A868'}
    names={'BM CODEX':'CODEX','Triana AbSeq':'AbSeq','BM CyTOF':'CyTOF'}
    fig,axes=plt.subplots(1,2,figsize=(9.1,4.6))
    # Interleave draw order so no dataset systematically hides another.
    order=np.random.default_rng(42).permutation(len(md))
    point_colors=md.dataset.map(colors).to_numpy()
    axes[0].scatter(xy[order,0],xy[order,1],c=point_colors[order],s=.45,alpha=.6,lw=0,rasterized=True)
    axes[1].scatter(xy[:,0],xy[:,1],c='#D6DBDF',s=.4,alpha=.24,lw=0,rasterized=True)
    for ds in ['Triana AbSeq','BM CyTOF','BM CODEX']:
        keep=md.dataset.eq(ds)&md.celltype.eq('CD4 T')
        axes[1].scatter(xy[keep,0],xy[keep,1],s=1.1,c=colors[ds],alpha=.8,lw=0,rasterized=True)
    for ax,title in zip(axes,['All cells: platform','CD4 T cells: platform']):
        ax.axis('off'); ax.set_aspect('equal'); ax.set_title(title,fontsize=12,fontweight='bold')
    fig.legend([Line2D([0],[0],marker='o',ls='',color=colors[d],ms=5) for d in colors],
        [names[d] for d in colors],loc='lower center',bbox_to_anchor=(.5,.055),ncol=3,frameon=False,fontsize=10)
    fig.suptitle('Three-platform alignment without CD4',fontsize=15,fontweight='bold',ha='left',x=.03,y=.98)
    fig.text(.03,.017,'88,031 cells · RAMP on 13 shared markers · CD4 and CD19 excluded · Fixed UMAP seed',fontsize=8.5)
    fig.subplots_adjust(left=.02,right=.98,top=.87,bottom=.15,wspace=.07)
    for ext in ('png','svg'): fig.savefig(OUT/f'joint_alignment_overview.{ext}',dpi=280,facecolor='white')
    plt.close(fig)

def analyze_regions(cells,aligned,md):
    # Keep the two descriptive regions defined on the previous CODEX-only map.
    # They are used for diagnosis, never for fitting/alignment or subtyping.
    region_path=PARENT/'codex_only_CD4_regions.csv'
    if not region_path.exists():
        original_cells=pd.read_csv(PARENT/'cell_predictions_and_umap.csv.gz')
        region=original_cells[original_cells.celltype.eq('CD4 T')].copy()
        km=KMeans(n_clusters=2,n_init=20,random_state=42).fit(region[['UMAP1','UMAP2']])
        region['region']=km.labels_
        region.to_csv(region_path,index=False)
    region=pd.read_csv(region_path)
    match=cells.reset_index().merge(region[['cell_id','region']],on='cell_id',validate='one_to_one')
    region_centers=match.groupby('region')[['UMAP1','UMAP2']].median()
    region_centers.to_csv(OUT/'old_regions_on_joint_umap.csv')
    rows=match['index'].to_numpy(int)
    df=pd.DataFrame(aligned[rows],columns=validation.FEATURES)
    df['region']=match.region.to_numpy(); df['donor']=match.donor.to_numpy()
    comparison=df.groupby(['donor','region'])[validation.FEATURES].median().reset_index()
    comparison.to_csv(OUT/'CD4_region_donor_marker_medians.csv',index=False)
    ra=comparison.pivot(index='donor',columns='region',values='CD45RA')
    ra['region0_minus_region1']=ra[0]-ra[1]
    ra.to_csv(OUT/'CD45RA_difference_by_donor.csv')
    fig,axes=plt.subplots(1,2,figsize=(9.1,4.6))
    old=np.load(PARENT/'umap_coordinates.npy')
    regmap=region.set_index('cell_id').region
    rr=cells.cell_id.map(regmap).to_numpy()
    for ax,coords,title in zip(axes,[old,cells[['UMAP1','UMAP2']].to_numpy()],
            ['CODEX-only UMAP','Joint 3-platform UMAP']):
        ax.scatter(coords[:,0],coords[:,1],s=.55,color='#D2D8DD',alpha=.4,lw=0,rasterized=True)
        for g,color in [(0,'#9A65B4'),(1,'#286BB0')]:
            keep=rr==g
            ax.scatter(coords[keep,0],coords[keep,1],s=2,color=color,alpha=.9,lw=0,rasterized=True)
        ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title,fontsize=12,fontweight='bold')
    fig.legend([Line2D([0],[0],marker='o',ls='',color=c,ms=5) for c in ['#9A65B4','#286BB0']],
        ['Original lower CD4 T region (325 cells)','Original right CD4 T region (1,101 cells)'],
        loc='lower center',bbox_to_anchor=(.5,.06),ncol=2,frameon=False,fontsize=8.5)
    fig.suptitle('Tracking the same two CD4 T regions',fontsize=15,fontweight='bold',ha='left',x=.03,y=.98)
    fig.text(.03,.017,'Regions describe the original layout; they are not new biological subtype annotations.',fontsize=8.5)
    fig.subplots_adjust(left=.02,right=.98,top=.87,bottom=.15,wspace=.07)
    for ext in ('png','svg'): fig.savefig(OUT/f'CD4_regions_old_vs_joint.{ext}',dpi=280,facecolor='white')
    plt.close(fig)
    print('CD45RA aligned median differences, original lower minus right region:',flush=True)
    print(ra.to_string(),flush=True)

if __name__=='__main__': main()
