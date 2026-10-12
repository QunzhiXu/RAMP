"""Extract acquired CD19 pixels at the exact coordinates already used in G."""
from pathlib import Path
import sys, json, hashlib, xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
BASE=Path(__file__).resolve().parents[1]
REAL=BASE/'real_spatial0913'
OUT=BASE/'figure_layout_v30_polish'
sys.path.insert(0,str(REAL/'deps'))
import tifffile

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def tile_crop(handle,page,roi):
    x0,y0,x1,y1=roi
    out=np.zeros((y1-y0,x1-x0),dtype=page.dtype)
    th,tw=page.tilelength,page.tilewidth
    nx=(page.shape[1]+tw-1)//tw
    for row in range(y0//th,(y1-1)//th+1):
        for col in range(x0//tw,(x1-1)//tw+1):
            i=row*nx+col
            handle.filehandle.seek(page.dataoffsets[i])
            segment,_,_=page.decode(handle.filehandle.read(page.databytecounts[i]),i)
            tile=segment[0,:,:,0]
            top,left=row*th,col*tw
            ya,yb=max(y0,top),min(y1,top+th)
            xa,xb=max(x0,left),min(x1,left+tw)
            out[ya-y0:yb-y0,xa-x0:xb-x0]=tile[ya-top:yb-top,xa-left:xb-left]
    return out

def main():
    for d in [OUT,OUT/'data',OUT/'qa']:
        d.mkdir(exist_ok=True)
    roi=(13440,3904,13952,4416)
    source=REAL/'downloads/H14_Scan1.qptiff'
    prior=json.loads((REAL/'results/render_audit.json').read_text())
    manifest=json.loads((REAL/'metadata/download_manifest.json').read_text())[0]
    assert source.stat().st_size==manifest['bytes']
    crops={};pages={}
    with tifffile.TiffFile(source) as f:
        for marker in ['DAPI','CD19']:
            i=prior['source_channel_pages'][marker]
            page=f.pages[i];xml=ET.fromstring(page.description)
            assert xml.findtext('Biomarker')==marker
            assert xml.findtext('ImageType')=='FullResolution'
            assert page.shape==(21600,19200)
            crops[marker]=tile_crop(f,page,roi);pages[marker]=i
            (OUT/'data'/f'{marker}_source_metadata.xml').write_text(page.description)
    with np.load(REAL/'data/selected_roi_measured_channels.npz') as z:
        for marker in crops:
            assert np.array_equal(crops[marker],z[marker][320:832,384:896])
    for marker,a in crops.items():
        assert a.dtype==np.uint8 and a.shape==(512,512)
        Image.fromarray(a).save(OUT/'data'/f'H14_G_ROI_observed_{marker}_raw.tif')
        Image.fromarray(a).save(OUT/'data'/f'H14_G_ROI_observed_{marker}_raw.png')
    a=crops['CD19']
    record={'status':'PASS','source':str(source),'source_article':manifest['article_url'],
        'source_bytes':source.stat().st_size,'author_md5_previously_verified':manifest['md5'],
        'source_pages':pages,'source_is_acquired_processed_scan':True,
        'roi_xyxy':list(roi),'parent_relative_xy':[384,320],'shape_yx':[512,512],
        'pixel_size_um':prior['roi']['pixel_size_um'],
        'orientation':'Native row/column order; origin upper; x right, y down; no rotation or reflection.',
        'exact_pixel_match_to_G_cached_DAPI':True,'exact_pixel_match_to_cached_CD19':True,
        'main_display':{'lower':0,'upper':255,'gamma':1,'cmap':'gray','local_contrast_rescaling':False},
        'raw_cd19':{'min':int(a.min()),'max':int(a.max()),'mean':float(a.mean()),
                    'median':float(np.median(a)),'fraction_le5':float(np.mean(a<=5)),
                    'p99_5':float(np.percentile(a,99.5))},
        'crop_sha256':sha(OUT/'data/H14_G_ROI_observed_CD19_raw.tif'),
        'roi_selection_unchanged':True,'n_centroid_cells':251,'n_annotated_B':43,
        'display_pixels_synthetic':False,'no_prediction_or_label_used_to_modify_CD19_pixels':True}
    (OUT/'data/ROI_CORRESPONDENCE.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record['raw_cd19'],indent=2))

if __name__=='__main__':main()
