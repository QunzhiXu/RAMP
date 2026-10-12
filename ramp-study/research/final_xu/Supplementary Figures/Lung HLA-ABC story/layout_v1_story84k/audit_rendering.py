"""Compare per-cell colors/order and rendering with the current story script."""
from pathlib import Path
import sys, ast, importlib.util, json
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('supplement_renderer',HERE/'build_figure.py')
new=importlib.util.module_from_spec(spec);spec.loader.exec_module(new)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def function_from_file(path,name,namespace):
    nodes=ast.parse(path.read_text()).body
    node=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]

def main():
    story=new.STORY/'Figure/Panel B'
    source=story/'scripts/draw_panel_B.py'
    ref={'np':np,'pd':pd}
    original=function_from_file(source,'scatter',ref)
    bounds=function_from_file(source,'bounds',ref)
    constants={}
    for node in ast.parse(source.read_text()).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            key=node.targets[0].id
            if key in ['PLATFORMS','PLATFORM_COLORS','TYPE_COLORS']:
                constants[key]=ast.literal_eval(node.value)
    main_node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    order_names={'order','types','bottom','upper_labels','middle','masks','type_order'}
    assignments=[n for n in main_node.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in order_names]
    original_md=pd.read_csv(story/'data/PLOT_DATA.csv.gz',low_memory=False)
    scope={'np':np,'md':original_md}
    exec(compile(ast.Module(body=assignments,type_ignores=[]),str(source),'exec'),scope)
    md=pd.read_csv(HERE/'display_metadata.csv.gz',low_memory=False)
    xy=np.load(HERE/'coordinates/RAMP.npy')
    style=json.loads((HERE/'STORY_STYLE.json').read_text())
    assert np.array_equal(md.cell_id,original_md.cell_key)
    assert np.array_equal(xy,np.load(story/'data/RAMP_UMAP.npy'))
    assert style['celltype_palette']==constants['TYPE_COLORS']
    assert style['dataset_palette']=={d:constants['PLATFORM_COLORS'][v] for d,v in constants['PLATFORMS'].items()}
    old_namespace={'np':np}
    old=function_from_file(HERE/'versions/before_story_render_match_20260922/build_figure.py','scatter',old_namespace)
    images={};checks={}
    for by in ['Dataset','Cell type']:
        expected=scope['order'] if by=='Dataset' else scope['type_order']
        assert np.array_equal(new.point_order(md,by),expected)
        for name in ['Story (scaled)','Previous supplement','Matched supplement']:
            fig=plt.figure(figsize=(39/25.4,27/25.4),dpi=300,facecolor='white')
            ax=fig.add_axes([0,0,1,1])
            if name=='Story (scaled)':
                values,palette=(original_md.platform,constants['PLATFORM_COLORS']) if by=='Dataset' else (original_md.display_cell_type,constants['TYPE_COLORS'])
                artist=original(ax,xy,values,palette,expected)
                # Resizing the original figure scales area quadratically.
                artist.set_sizes(artist.get_sizes()*(27/new.STORY_PLOT_HEIGHT_MM)**2)
                limits=bounds(xy)
                ax.set_xlim(limits[0]);ax.set_ylim(limits[1]);ax.set_aspect('equal',adjustable='box');ax.set_axis_off()
            elif name=='Previous supplement':
                old(ax,md,xy,by,style)
                assert np.array_equal(np.vstack([c.get_offsets() for c in ax.collections]),xy[expected])
            else:
                artist=new.scatter(ax,md,xy,by,style)
                assert np.array_equal(artist.get_offsets(),xy[expected])
            fig.canvas.draw()
            images[(by,name)]=np.asarray(fig.canvas.buffer_rgba()).copy()
            plt.close(fig)
        reference=images[(by,'Story (scaled)')]
        previous=images[(by,'Previous supplement')]
        matched=images[(by,'Matched supplement')]
        assert np.array_equal(reference,matched),by
        checks[by]=dict(exact_per_cell_order_matched=True,exact_palette_matched=True,
            corrected_pixels_identical_to_proportionally_scaled_story=True,
            previous_different_pixel_fraction=float(np.any(reference!=previous,axis=2).mean()),
            previous_rgb_mean_absolute_difference=float(np.abs(reference[:,:,:3].astype(float)-previous[:,:,:3]).mean()))
    plt.rcParams.update({'font.family':'Arial','font.size':8.5,'text.color':'#263238'})
    fig=plt.figure(figsize=(142/25.4,83/25.4),dpi=300,facecolor='white')
    names=['Story (scaled)','Previous supplement','Matched supplement']
    for row,by in enumerate(['Dataset','Cell type']):
        fig.text(4/142,1-(16+row*37)/83,by,weight='bold',fontsize=8)
        for col,name in enumerate(names):
            left=5+col*46;top=20+row*37
            if row==0:fig.text((left+19.5)/142,1-5/83,name,ha='center',va='top',weight='bold')
            ax=fig.add_axes([left/142,1-(top+25)/83,39/142,25/83])
            ax.imshow(images[(by,name)],interpolation='none');ax.set_axis_off()
    out=HERE/'output';out.mkdir(exist_ok=True)
    fig.savefig(out/'story_rendering_comparison.png',dpi=300,facecolor='white');plt.close(fig)
    record=dict(status='passed',source_story=str(source),comparison_checks=checks,
        original_alpha=.65,previous_dataset_alpha=.88,previous_celltype_alpha=.92,
        original_area_pt2=.60,story_plot_height_mm=new.STORY_PLOT_HEIGHT_MM,
        supplement_plot_height_mm=27,matched_area_pt2=new.SCATTER_AREA,
        previous_dataset_area_pt2=.25,previous_celltype_area_pt2=.29,
        dataset_relative_area_excess=float(.25/new.SCATTER_AREA-1),
        celltype_relative_area_excess=float(.29/new.SCATTER_AREA-1),
        original_cell_order_unchanged=True,coordinate_refit=False,
        explanation='Previous colors and per-cell draw order were exact, but alpha and marker area relative to plot size were not; legend ordering also differed.')
    (HERE/'RENDERING_AUDIT.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ['matched_area_pt2','dataset_relative_area_excess','celltype_relative_area_excess','comparison_checks']},indent=2))

if __name__=='__main__':main()
