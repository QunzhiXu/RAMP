"""Verify public downloads, save provenance, and locate ROI on original DAPI."""
from pathlib import Path
import os,sys,json,hashlib,xml.etree.ElementTree as ET
BASE=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(BASE/'cache/matplotlib')
sys.path.insert(0,str(BASE/'deps'));sys.dont_write_bytecode=True
import numpy as np,pandas as pd,tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def main():
    downloads=[]
    for metafile in ['figshare_images.json','figshare_masks.json']:
        meta=json.loads((BASE/'metadata'/metafile).read_text())
        for f in meta['files']:
            p=BASE/'downloads'/f['name']
            if not p.exists():continue
            assert p.stat().st_size==f['size'],f'{p.name} not fully downloaded'
            with p.open('rb') as stream: md5=hashlib.file_digest(stream,'md5').hexdigest()
            assert md5==f['computed_md5'],f'{p.name} checksum mismatch'
            downloads.append(dict(name=p.name,url=f['download_url'],bytes=f['size'],md5=md5,verified_against_author=True,
                                  article_url=meta['url_public_html'],doi=meta['doi']))
            print('Verified',p.name,f['size'],md5,flush=True)
    (BASE/'metadata/download_manifest.json').write_text(json.dumps(downloads,indent=2)+'\n')
    roi=json.loads((BASE/'results/selected_roi.json').read_text())
    with tifffile.TiffFile(BASE/'downloads/H14_Scan1.qptiff') as t:
        pages=[]
        for i,p in enumerate(t.pages):
            try:
                xml=ET.fromstring(p.description)
                if xml.findtext('Biomarker')=='DAPI' and xml.findtext('ImageType')=='ReducedResolution':pages.append(i)
            except ET.ParseError: pass
        i=min(pages,key=lambda i:t.pages[i].shape[0]);page=t.pages[i];overview=page.asarray()
        scale=t.pages[0].shape[0]/overview.shape[0]
    hi=float(np.percentile(overview[overview>0],99.5))
    z=np.clip(overview/hi,0,1)**.8;rgb=z[:,:,None]*np.array([.32,.45,.82])
    fig,ax=plt.subplots(figsize=(6,6.8));ax.imshow(rgb,interpolation='nearest');ax.set_axis_off()
    ax.add_patch(Rectangle((roi['x0']/scale,roi['y0']/scale),1024/scale,1024/scale,fill=False,edgecolor='#FFB52E',linewidth=1.8))
    ax.text(roi['x1']/scale+20,roi['y0']/scale,'Selected ROI',color='#FFB52E',fontsize=11,fontweight='bold')
    ax.set_title('H14 · original DAPI image',fontweight='bold',fontsize=15,pad=10)
    ax.text(.5,-.035,f'Best F1 among {roi["n_eligible"]:,} eligible H14 windows',ha='center',transform=ax.transAxes,fontsize=11)
    fig.subplots_adjust(left=.01,right=.99,top=.95,bottom=.07)
    fig.savefig(BASE/'figures/H14_original_DAPI_ROI_location.png',dpi=240,facecolor='white');plt.close(fig)
    pd.DataFrame(downloads).to_csv(BASE/'metadata/download_manifest.csv',index=False)
    print('Downloads verified and overview exported',flush=True)

if __name__=='__main__':main()
