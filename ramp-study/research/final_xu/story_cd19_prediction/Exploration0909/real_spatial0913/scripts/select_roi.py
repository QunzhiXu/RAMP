"""Prespecified grid search on all H14 atlas cells; no threshold optimization."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'deps'))
import tifffile

def main():
    df=pd.read_csv(BASE/'data/H14_full_predictions.csv.gz')
    with tifffile.TiffFile(BASE/'downloads/H14_Scan1.qptiff') as t:
        p=t.pages[0];height,width=p.shape
        xres=p.tags['XResolution'].value
        assert int(p.tags['ResolutionUnit'].value)==3
        um_per_pixel=10000*xres[1]/xres[0]
    xy=df[['x.coord','y.coord']].to_numpy(); y=df.annotated_b.to_numpy(bool);positive=df.predicted_high.to_numpy(bool)
    entries=[]
    for y0 in range(0,height-1024+1,256):
        for x0 in range(0,width-1024+1,256):
            inside=(xy[:,0]>=x0)&(xy[:,0]<x0+1024)&(xy[:,1]>=y0)&(xy[:,1]<y0+1024)
            yy=y[inside];pp=positive[inside];n=int(inside.sum());b=int(yy.sum());high=int(pp.sum())
            if n<300 or b<25 or high<25: continue
            tp=int((yy&pp).sum());fp=high-tp;fn=b-tp
            precision=tp/high;recall=tp/b
            entries.append(dict(x0=x0,y0=y0,x1=x0+1024,y1=y0+1024,n_cells=n,n_b=b,n_high=high,tp=tp,fp=fp,fn=fn,
                                precision=precision,recall=recall,f1=2*tp/(high+b),iou=tp/(tp+fp+fn),
                                min_precision_recall=min(precision,recall)))
    ranking=pd.DataFrame(entries).sort_values(['f1','min_precision_recall','n_b','y0','x0'],ascending=[False,False,False,True,True]).reset_index(drop=True)
    assert len(ranking)>0
    ranking.insert(0,'presentation_rank',np.arange(1,len(ranking)+1))
    ranking.to_csv(BASE/'results/roi_ranking_all_eligible.csv',index=False)
    top=ranking.iloc[0].to_dict();top.update(donor='H14',image_shape_yx=[height,width],pixel_size_um=um_per_pixel,
        roi_side_pixels=1024,roi_side_um=1024*um_per_pixel,n_eligible=len(ranking),stride_pixels=256,
        threshold=1.4,reference='author cluster_anno_l1 == B-Cells',selection='best local illustrative ROI within H14, ranked by F1',
        includes_all_atlas_cells_in_roi=True)
    (BASE/'results/selected_roi.json').write_text(json.dumps(top,indent=2)+'\n')
    m=(xy[:,0]>=top['x0'])&(xy[:,0]<top['x1'])&(xy[:,1]>=top['y0'])&(xy[:,1]<top['y1'])
    df.loc[m].to_csv(BASE/'data/selected_roi_cells.csv.gz',index=False)
    print(json.dumps(top,indent=2),flush=True)
    print(ranking.head(10).to_string(index=False))

if __name__=='__main__':main()
