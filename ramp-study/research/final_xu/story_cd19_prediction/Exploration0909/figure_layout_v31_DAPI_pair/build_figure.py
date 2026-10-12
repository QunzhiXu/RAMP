"""Reassemble Figure 3 with paired acquired channels and a tighter first row."""
from pathlib import Path
import ast
import json
import hashlib
import shutil
import sys
import copy

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(BASE/'scripts'))
import assemble_cd19_layout_v30_polish as frozen
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Bbox
from pypdf import PdfReader,PdfWriter

OLD=BASE/'figure_layout_v30_polish'
SCRIPT=BASE/'scripts/assemble_cd19_layout_v30_polish.py'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def baseline_figure():
    """Run the unchanged v30 artist construction without its export/backup code."""
    module=ast.parse(SCRIPT.read_text())
    main=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    begin=next(i for i,n in enumerate(main.body) if 'plt.rcParams.update' in ast.unparse(n).splitlines()[0])
    end=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and
             any(isinstance(t,ast.Name) and t.id=='name' for t in n.targets) and
             isinstance(n.value,ast.Constant) and n.value.value=='cd19_story_three_row_v30_polish')
    builder=copy.deepcopy(main)
    builder.name='build_frozen_figure'
    builder.body=builder.body[begin:end]+ast.parse('return fig, groups, icon').body
    code=ast.fix_missing_locations(ast.Module(body=[builder],type_ignores=[]))
    scope=dict(frozen.__dict__)
    scope['OUT']=HERE
    scope['icon']=np.asarray(Image.open(OLD/'assets/ramp_sign_no_pipes.png').convert('RGB'))
    exec(compile(code,str(SCRIPT),'exec'),scope)
    return scope['build_frozen_figure']()


def compact_workflow(ax,icon):
    """Keep box text and icon readable while shortening the connections."""
    ax.clear()
    ax.set(xlim=(0,5.3),ylim=(0,2.07),aspect='auto')
    ax.axis('off')
    boxes,box_texts={},{}
    style=frozen.previous
    def box(name,x,y,w,h,title,body=None,color=None):
        p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.035',
                         facecolor='white',edgecolor=style.FRAME,linewidth=style.FRAME_LW)
        ax.add_patch(p);boxes[name]=p
        if body is None:
            ts=[ax.text(x+w/2,y+h/2,title,ha='center',va='center',fontsize=10.5,
                        fontweight='bold',color=color or style.INK,linespacing=1.2)]
        else:
            ts=[ax.text(x+w/2,y+h-.15,title,ha='center',va='center',
                        fontsize=10.6 if name=='completed' else 11.2,
                        fontweight='bold',color=color or style.INK),
                ax.text(x+w/2,y+(.20 if '\n' in body else .14),body,ha='center',va='center',
                        fontsize=10.5,linespacing=1.12,color=style.INK)]
        box_texts[name]=ts
    def arrow(start,end,via=None):
        args=dict(arrowstyle='->',mutation_scale=7.,linewidth=.78,color='#73787B',
                  shrinkA=0,shrinkB=0,zorder=3,capstyle='round',joinstyle='round')
        if via:
            points=np.asarray([start,*via,end],float)
            vertices,codes=[points[0]],[MplPath.MOVETO]
            for i in range(1,len(points)-1):
                p,prev,nxt=points[i],points[i-1],points[i+1]
                incoming,outgoing=p-prev,nxt-p
                radius=min(.09,np.linalg.norm(incoming)/3,np.linalg.norm(outgoing)/3)
                vertices.extend([p-radius*incoming/np.linalg.norm(incoming),p,
                                 p+radius*outgoing/np.linalg.norm(outgoing)])
                codes.extend([MplPath.LINETO,MplPath.CURVE3,MplPath.CURVE3])
            vertices.append(points[-1]);codes.append(MplPath.LINETO)
            ax.add_patch(FancyArrowPatch(path=MplPath(vertices,codes),**args))
        else:
            ax.add_patch(FancyArrowPatch(start,end,**args))
    box('codex',.08,1.24,1.40,.59,'CODEX','14 shared markers\nfailed CD19 withheld','#3E70B2')
    box('reference',.08,.13,1.40,.59,'References','CyTOF + AbSeq\nmeasured CD19','#31916E')
    box('completed',2.55,1.30,1.60,.47,'Completed CODEX','+ predicted CD19')
    ax.text(2.00,1.785,'Masked-marker\nNN',ha='center',va='center',fontsize=9.7,
            linespacing=1.12,color=style.INK)
    arrow((1.535,1.535),(2.495,1.535))
    arrow((.78,.775),(2.00,1.48),via=[(.78,.99),(2.00,.99)])
    ix,iy,iw=2.21,.05,1.84
    ih=iw*icon.shape[0]/icon.shape[1]
    im=ax.imshow(icon,extent=(ix,ix+iw,iy,iy+ih),aspect='auto',interpolation='lanczos',zorder=2)
    arrow((3.35,1.245),(3.35,iy+ih+.04))
    arrow((1.535,.63),(ix-.055,.63))
    box('output',4.36,.36,.85,.54,'Aligned\ndatasets')
    arrow((ix+iw+.05,.63),(4.305,.63))
    return boxes,box_texts,im


if __name__=='__main__':
    for p in [HERE/'data',HERE/'qa']:
        p.mkdir(parents=True,exist_ok=True)
    protected=[SCRIPT,*[p for p in OLD.rglob('*') if p.is_file()],
               BASE/'real_spatial0913/results/render_audit.json']
    hashes={str(p):sha(p) for p in protected}
    for name in ['H14_G_ROI_observed_CD19_raw.tif','H14_G_ROI_observed_DAPI_raw.tif','ROI_CORRESPONDENCE.json']:
        shutil.copy2(OLD/'data'/name,HERE/'data'/name)
    fig,groups,icon=baseline_figure()
    W,H=fig.get_size_inches()
    original_bounds={letter:[list(ax.get_position().bounds) for ax in axes] for letter,axes in groups.items()}
    original_payload={letter:frozen.previous.payload(axes) for letter,axes in groups.items()}
    original_titles=[t.get_text() for t in groups['C'][0].texts]

    raw_cd19=np.asarray(Image.open(HERE/'data/H14_G_ROI_observed_CD19_raw.tif'))
    raw_dapi=np.asarray(Image.open(HERE/'data/H14_G_ROI_observed_DAPI_raw.tif'))
    record=json.loads((HERE/'data/ROI_CORRESPONDENCE.json').read_text())
    display=json.loads((BASE/'real_spatial0913/results/render_audit.json').read_text())['display']
    dapi=.82*np.clip(raw_dapi.astype(float)/display['dapi_hi'],0,1)**display['dapi_gamma']
    old_a=groups['A'][0]
    pos=old_a.get_position().frozen()
    side=pos.height*H
    fig.delaxes(old_a)
    groups['A']=[]
    for x,array,title in [(.20,dapi,'DAPI'),(2.03,raw_cd19.astype(float)/255.,'Observed CD19')]:
        ax=fig.add_axes([x/W,pos.y0,side/W,side/H])
        ax.imshow(array,cmap='gray',vmin=0,vmax=1,origin='upper',interpolation='none')
        ax.axis('off')
        ax.text(.96,.96,title,transform=ax.transAxes,ha='right',va='top',
                fontsize=10.5,color='#E9E9E9')
        length=50/record['pixel_size_um']
        xx,yy=512-length-24,512-28
        ax.plot([xx,xx+length],[yy,yy],color='white',lw=2.4,solid_capstyle='butt')
        ax.text(xx+length/2,yy-10,'50 µm',color='white',fontsize=9,ha='center',va='bottom')
        groups['A'].append(ax)

    # Keep typography and y scales; use narrower x dimensions for the two B plots.
    for ax,left,width in zip(groups['B'],[4.42,6.93],[2.08,1.95]):
        p=ax.get_position()
        ax.set_position([left/W,p.y0,width/W,p.height])
    b_note=next(t for t in groups['B'][1].texts if t.get_text()=='Observed CD19\nB-versus-T AUROC')
    b_note.set_text('Observed CD19\nB-versus-T\nAUROC')
    b_note.set_position((.68,.44))
    for text in groups['B'][1].texts:
        if text.get_text().startswith('median '):
            value=text.get_text().split(' ',1)[1]
            text.set_text('median\n'+value)
            text.set_verticalalignment('top')
            if value=='0.53':
                text.set_y(.82)
    for letter,left in [('B',4.00),('C',9.03)]:
        for text in fig.texts:
            if text.get_text()==letter:
                text.set_x((left+.02)/W)
            elif text.get_text()==frozen.previous.TITLES[letter]:
                text.set_x((left+.30)/W)
    cax=groups['C'][0]
    p=cax.get_position()
    cax.set_position([9.03/W,p.y0,5.3/W,p.height])
    boxes,texts,cimage=compact_workflow(cax,icon)

    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    clearances={}
    for name,items in texts.items():
        box=boxes[name].get_window_extent(renderer)
        for i,t in enumerate(items):
            bb=t.get_window_extent(renderer)
            margin=min(bb.x0-box.x0,box.x1-bb.x1,bb.y0-box.y0,box.y1-bb.y1)/fig.dpi*72
            clearances[f'{name}_{i}']=margin
            assert margin>1,(name,margin)
    for t in cax.texts:
        tb=t.get_window_extent(renderer)
        for name,box in boxes.items():
            if t not in texts[name]:
                assert not tb.overlaps(box.get_window_extent(renderer)),(t.get_text(),name)
    image_box=cimage.get_window_extent(renderer)
    assert abs(image_box.width/image_box.height-icon.shape[1]/icon.shape[0])<1e-9
    for letter in ['D','E','F','G','H']:
        assert original_payload[letter]==frozen.previous.payload(groups[letter]),letter
        assert original_bounds[letter]==[list(ax.get_position().bounds) for ax in groups[letter]],letter
    assert original_payload['B']==frozen.previous.payload(groups['B'])
    assert sorted(original_titles)==sorted(t.get_text() for t in cax.texts)
    assert np.array_equal(groups['A'][1].images[0].get_array(),raw_cd19.astype(float)/255.)
    for ax in groups['A']:
        assert abs(ax.get_position().width*W-side)<1e-9
        assert abs(ax.get_position().y1-groups['B'][0].get_position().y1)<1e-9
    b_box=groups['B'][1].get_window_extent(renderer)
    nb=b_note.get_window_extent(renderer)
    assert b_box.contains(nb.x0,nb.y0) and b_box.contains(nb.x1,nb.y1)
    for col in groups['B'][1].collections:
        pts=groups['B'][1].transData.transform(col.get_offsets())
        assert not np.any((pts[:,0]>=nb.x0)&(pts[:,0]<=nb.x1)&(pts[:,1]>=nb.y0)&(pts[:,1]<=nb.y1))
    notes=groups['B'][1].texts
    for i,text in enumerate(notes):
        bb=(text.get_bbox_patch().get_window_extent(renderer) if text.get_bbox_patch() is not None
            else text.get_window_extent(renderer))
        for other in notes[i+1:]:
            ob=(other.get_bbox_patch().get_window_extent(renderer) if other.get_bbox_patch() is not None
                else other.get_window_extent(renderer))
            assert not bb.overlaps(ob),(text.get_text(),other.get_text())
    left_box=groups['B'][0].get_window_extent(renderer)
    tick_left=min(t.get_window_extent(renderer).x0 for t in groups['B'][1].get_yticklabels() if t.get_visible())
    assert tick_left-left_box.x1>3/72*fig.dpi
    title_b=next(t for t in fig.texts if t.get_text()==frozen.previous.TITLES['B'])
    heading_c=next(t for t in fig.texts if t.get_text()=='C')
    assert title_b.get_window_extent(renderer).x1<heading_c.get_window_extent(renderer).x0
    assert (groups['B'][0].yaxis.label.get_window_extent(renderer).x0
            -groups['A'][1].get_window_extent(renderer).x1)>4/72*fig.dpi

    name='cd19_story_three_row_v31_DAPI_pair'
    fig.savefig(HERE/f'{name}.png',dpi=300,facecolor='white')
    fig.savefig(HERE/f'{name}.svg',dpi=300,facecolor='white')
    fig.savefig(HERE/'qa/working_size.pdf',dpi=600,facecolor='white')
    fig.savefig(HERE/'row1_preview.png',dpi=300,facecolor='white',
                bbox_inches=Bbox.from_extents(.10,H-2.34,14.42,H-.08))
    fig.savefig(HERE/'panel_A_preview.png',dpi=600,facecolor='white',
                bbox_inches=Bbox.from_extents(.10,H-2.15,3.83,H-.08))
    fig.savefig(HERE/'panel_B_preview.png',dpi=300,facecolor='white',
                bbox_inches=Bbox.from_extents(3.86,H-2.34,8.97,H-.08))
    fig.savefig(HERE/'panel_C_preview.png',dpi=300,facecolor='white',
                bbox_inches=Bbox.from_extents(9.01,H-2.34,14.42,H-.08))
    plt.close(fig)
    pdf=PdfReader(HERE/'qa/working_size.pdf').pages[0]
    width=180/25.4*72
    pdf.scale_to(width,width*H/W)
    writer=PdfWriter();writer.add_page(pdf)
    writer.add_metadata({'/Title':'Figure 3: paired acquired DAPI and CD19 with compact first row'})
    with (HERE/f'{name}.pdf').open('wb') as f:
        writer.write(f)

    old_png=np.asarray(Image.open(OLD/'cd19_story_three_row_v30_polish.png').convert('RGB'))
    new_png=np.asarray(Image.open(HERE/f'{name}.png').convert('RGB'))
    assert old_png.shape==new_png.shape
    difference=np.any(old_png!=new_png,axis=-1)
    yy,xx=np.where(difference)
    assert yy.max()<720,{'changed_pixel_bbox':[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())]}
    assert all(sha(Path(p))==digest for p,digest in hashes.items())
    audit={'status':'PASS','protected_input_sha256':hashes,'baseline_source':str(SCRIPT),
           'canvas_inches':[W,H],'output_width_mm':180,
           'A_image_side_inches_before_and_after':[side,side],
           'A_two_views':['DAPI','Observed CD19'],'A_titles_inside_upper_right':True,
           'A_and_B_plot_tops_aligned':True,'B_plot_widths_inches':[2.08,1.95],
           'C_width_inches_before_after':[5.78,5.3],
           'C_icon_width_inches_before_after':[1.84,1.84],
           'C_box_text_clearances_working_points':clearances,
           'C_workflow_text_content_unchanged':True,
           'B_scientific_payload_unchanged':True,'D_through_H_pixelwise_unchanged':True,
           'changed_pixel_bbox':[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
           'old_files_unchanged':True,'no_refit_or_data_change':True,'manuscript_replaced':False}
    (HERE/'LAYOUT_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    (HERE/'README.md').write_text('''# Figure 3 v31：DAPI/CD19 双图与紧凑第一行

主图：cd19_story_three_row_v31_DAPI_pair.png / svg / pdf。
局部预览：row1_preview.png、panel_A_preview.png、panel_B_preview.png、panel_C_preview.png。

A 并排显示真实 DAPI 与实测 CD19，两图保持 v30 单图的尺寸、ROI、方向、50 µm 标尺。
通道名移到各图右上方的黑色空白处，图像上沿与 B 对齐，整图高度不增加。
DAPI 显示与原 G 灰色底图一致（上限 109、gamma 0.8、增益 0.82）；
CD19 是原 0–255 灰度显示，无 DAPI 混色、局部拉伸或细胞掩膜。
两个不同通道的亮度不作同单位比较；DAPI 提供形态参考，CD19 特异性由 B 定量支持。

B 收窄两个统计图，保留数值、字号和坐标范围；C 缩短连接线与框间距，
保留全部信息、原 RAMP 图标尺寸及比例。D–H 的内容、布局与像素完全不变。
原 v30、此前单独预览与 manuscript 均保留，此版尚未同步到 manuscript。

复现：python3 build_figure.py。构建器复用冻结 v30 的绘图部分及检查，不运行其旧输出写入。
源文件哈希、布局参数及像素检查见 LAYOUT_AUDIT.json。
''')
    print(json.dumps({k:v for k,v in audit.items() if k!='protected_input_sha256'},indent=2))
