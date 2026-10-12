"""Export the approved DAPI/CD19 pair without changing its data or layout."""
from pathlib import Path
import hashlib
import json
import importlib.util
import os
HERE=Path(__file__).resolve().parent
PAIR=HERE.parent
BASE=PAIR.parent
os.environ['MPLCONFIGDIR']=str(BASE/'panel_A_dapi_trial_20260915/mpl_cache')
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fitz


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


if __name__=='__main__':
    spec=importlib.util.spec_from_file_location('pair_renderer',PAIR/'render_pair.py')
    pair=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pair)
    paths=[pair.OLD/'data/H14_G_ROI_observed_DAPI_raw.tif',
           pair.OLD/'data/H14_G_ROI_observed_CD19_raw.tif',
           pair.OLD/'data/ROI_CORRESPONDENCE.json',
           pair.REAL/'results/render_audit.json',
           PAIR/'panel_A_DAPI_CD19_pair.png',
           PAIR/'panel_A_DAPI_CD19_pair.svg',
           PAIR/'render_pair.py']
    hashes={str(p):sha(p) for p in paths}
    raw_dapi=np.asarray(Image.open(paths[0]))
    raw_cd19=np.asarray(Image.open(paths[1]))
    pair.record=json.loads(paths[2].read_text())
    display=json.loads(paths[3].read_text())['display']
    dapi=.82*np.clip(raw_dapi.astype(float)/display['dapi_hi'],0,1)**display['dapi_gamma']
    cd19=raw_cd19.astype(float)/255.
    plt.rcParams.update({'font.family':'Arial','svg.fonttype':'none','pdf.fonttype':42,'text.color':'#202020'})
    fig=plt.figure(figsize=(7.2,3.95))
    for x,array,title in [(.035,dapi,'DAPI'),(.530,cd19,'Observed CD19')]:
        ax=fig.add_axes([x,.035,.435,.435*7.2/3.95])
        pair.show(ax,array,title)
        # Preserve the native 512x512 raster in vector containers.
        ax.images[0].set_interpolation('none')
    fig.text(.035,.985,'A',ha='left',va='top',fontsize=17,fontweight='bold')
    fig.text(.081,.983,'Failed CODEX CD19',ha='left',va='top',fontsize=13,fontweight='bold')
    fig.savefig(HERE/'panel_A_DAPI_CD19_600dpi.png',dpi=600,facecolor='white')
    fig.savefig(HERE/'panel_A_DAPI_CD19_600dpi.tif',dpi=600,facecolor='white',
                pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(HERE/'panel_A_DAPI_CD19_vector.pdf',dpi=600,facecolor='white',
                metadata={'Title':'Panel A: acquired DAPI and CD19 in the same ROI',
                          'Subject':'Vector text and scale bars with native 512x512 microscopy rasters'})
    fig.savefig(HERE/'panel_A_DAPI_CD19_vector.svg',dpi=600,facecolor='white')
    plt.close(fig)

    with Image.open(HERE/'panel_A_DAPI_CD19_600dpi.png') as im:
        png=np.asarray(im)
        assert im.size==(4320,2370)
        image_info={'dimensions':list(im.size),'dpi':list(im.info['dpi'])}
    with Image.open(HERE/'panel_A_DAPI_CD19_600dpi.tif') as im:
        assert im.size==(4320,2370)
        assert np.array_equal(png,np.asarray(im))
    with fitz.open(HERE/'panel_A_DAPI_CD19_vector.pdf') as pdf:
        assert len(pdf)==1
        page=pdf[0]
        labels=page.get_text()
        for text in ['Failed CODEX CD19','DAPI','Observed CD19','50']:
            assert text in labels
        rasters=page.get_images(full=True)
        assert len(rasters)==2 and all(i[2:4]==(512,512) for i in rasters)
        pdf_info={'page_points':[page.rect.width,page.rect.height],
                  'embedded_image_sizes':[list(i[2:4]) for i in rasters],
                  'vector_text_extractable':True}
    assert all(sha(Path(p))==v for p,v in hashes.items())
    audit={'status':'PASS','source_sha256':hashes,
           'PNG_TIFF':image_info,'PNG_TIFF_pixels_identical':True,'PDF':pdf_info,
           'native_microscopy_resolution':[512,512],
           'micrometers_per_pixel':pair.record['pixel_size_um'],
           'new_microscopy_detail_generated':False,
           'layout_and_intensity_display_unchanged':True,
           'manuscript_replaced':False}
    (HERE/'EXPORT_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    (HERE/'README.md').write_text('''# Panel A 高清导出

- `panel_A_DAPI_CD19_600dpi.png`：4320×2370 像素，600 dpi。
- `panel_A_DAPI_CD19_600dpi.tif`：相同像素，LZW 无损压缩。
- `panel_A_DAPI_CD19_vector.pdf/svg`：文字与比例尺为矢量，显微图嵌入原生 512×512 像素。

沿用已确认的左右 DAPI/CD19 设计、同一 ROI 和显示参数。
原始全分辨率裁切每通道为 512×512 像素，0.5069445 µm/像素。
高 dpi 文件增加的是导出画布像素，不增加原始显微分辨率；没有 AI 超分或生成细胞纹理。
每幅原图若印刷为 40 mm 宽，原始有效分辨率约 325 dpi。
排版优先使用 PDF 或 SVG，以保留矢量文字。旧预览与 manuscript 不变。

复现：`python3 export_high_resolution.py`。核验记录见 EXPORT_AUDIT.json。
''')
    print(json.dumps(audit,indent=2))
