"""Same-width visual comparison; original artwork is read-only."""
from pathlib import Path
import os
import json
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'.cache/matplotlib')
os.environ['MPLBACKEND']='Agg'
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
import numpy as np

def main():
    for name in ['Arial.ttf','Arial Bold.ttf']:
        font_manager.fontManager.addfont('/System/Library/Fonts/Supplemental/'+name)
    plt.rcParams.update({'font.family':'Arial'})
    old=Image.open(HERE.parent/'Combined ABCDE/main_figure_ABCDE.png')
    new=Image.open(HERE/'main_figure_1_polished.png')
    w,h=372,201
    fig=plt.figure(figsize=(w/25.4,h/25.4),facecolor='white')
    for x,im,label in [(3,old,'Previous'),(189,new,'Polished')]:
        fig.text(x/w,1-5/h,label,fontsize=10,weight='bold',va='center')
        image_h=180*im.height/im.width
        ax=fig.add_axes([x/w,1-(10+image_h)/h,180/w,image_h/h])
        ax.imshow(im);ax.axis('off')
    fig.savefig(HERE/'before_after_same_width.png',dpi=210)
    plt.close(fig)
    # Record the Panel A revision against its own archived predecessor.
    previous=HERE/'before_panel_a_expansion/main_figure_1_polished.png'
    lower_previous=HERE/'before_lower_row_polish/main_figure_1_polished.png'
    if previous.exists():
        a=np.array(Image.open(previous))
        a_revision=Image.open(lower_previous) if lower_previous.exists() else new
        b=np.array(a_revision)
        assert a.shape==b.shape
        bx=round(57/180*b.shape[1]); by=round(124/188*b.shape[0])
        mask=np.ones(b.shape[:2],dtype=bool);mask[:by,:bx]=False
        changed=int(np.any(a!=b,axis=2)[mask].sum())
        assert changed==0,changed
        (HERE/'qa/panel_a_only_changes.json').write_text(json.dumps({
            'revision':'Panel A expansion, before the later lower-row revision',
            'image_dimensions':list(a.shape),'changed_pixels_outside_panel_a':changed,
            'panels_B_to_E_pixel_identical':True},indent=2)+'\n')
        Image.fromarray(b).crop((0,0,bx,by)).save(HERE/'panel_a_expanded.png')
        fig=plt.figure(figsize=(118/25.4,132/25.4),facecolor='white')
        for x,im,label in [(0,Image.fromarray(a),'Previous A'),(61,new,'Expanded A')]:
            fig.text((x+2)/118,1-3/132,label,fontsize=8,weight='bold',va='center')
            ax=fig.add_axes([x/118,1-(7+124)/132,57/118,124/132])
            ax.imshow(im.crop((0,0,bx,by)));ax.axis('off')
        fig.savefig(HERE/'panel_a_before_after.png',dpi=300)
        plt.close(fig)
    if lower_previous.exists():
        a=np.array(Image.open(lower_previous));b=np.array(new)
        assert a.shape==b.shape
        boundary=round(124/188*b.shape[0])
        changed=int(np.any(a[:boundary]!=b[:boundary],axis=2).sum())
        assert changed==0,changed
        (HERE/'qa/lower_row_only_changes.json').write_text(json.dumps({
            'revision':'C/D enlarged, C dataset legend removed, E labels above bars',
            'image_dimensions':list(b.shape),'changed_pixels_above_lower_row':changed,
            'panels_A_B_pixel_identical':True},indent=2)+'\n')
        new.crop((0,boundary,new.width,new.height)).save(HERE/'lower_row_polished.png')

if __name__=='__main__':main()
