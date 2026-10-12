"""Compare only the SOX10/RAMP left display limit. Densities and right limit are frozen."""
from pathlib import Path
import sys,importlib.util,json,hashlib
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('ridge_preview',HERE.parent/'build_preview.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np,pandas as pd,fitz

def main():
    locked=[h.ROOT.parent/'bioarxiv_Ver2/main.pdf',h.ROOT.parent/'bioarxiv_Ver2/main.tex',h.ROOT.parent/'bioarxiv_Ver2/figure/figure1_ramp_overview.pdf',h.HERE/'output/pdf/figure1_ridge_range_comparison.pdf']
    before={str(p):h.sha(p) for p in locked}
    meta,markers,matrices,sources=h.load();prepared,_,_=h.prepare(meta,markers,matrices)
    obj=prepared[('RAMP','SOX10')];original_lo,hi=obj['new_limits']
    candidates=[('Current preview',original_lo),('Left limit −1.5',-1.5),('Left limit −1.0',-1)]
    rows=[];summary=[]
    fig=plt.figure(figsize=(7.6,4.4),facecolor='white')
    fig.text(.055,.965,'RAMP | SOX10',fontsize=13,fontweight='bold',va='top')
    handles=[Line2D([],[],marker='o',ls='none',color=c,markersize=4,label=l) for c,l in zip(h.COLORS,['CODEX','IMC','MIBI'])]
    fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(.975,.977),ncol=3,frameon=False,fontsize=9,handletextpad=.15,columnspacing=1.2)
    for k,(title,lo) in enumerate(candidates):
        view=dict(obj);view['new_limits']=[lo,hi]
        ax=fig.add_axes([.055+k*.315,.105,.275,.72]);h.draw(ax,view,'new')
        ax.set_title(title,fontsize=10.5,fontweight='bold',pad=9)
        ax.tick_params(axis='x',labelsize=9)
        # Identical y limits and physical axes sizes permit direct visual comparison.
        subset=[]
        for c in obj['curves']:
            v=c['values'];entry=dict(variant=title,unit_key=c['unit'],dataset=c['dataset'],display_low=lo,display_high=hi,
              n_cells=len(v),n_left=int((v<lo).sum()),n_right=int((v>hi).sum()),
              left_boundary_height_fraction=float(np.interp(lo,obj['x'],c['density'])),
              right_boundary_height_fraction=float(np.interp(hi,obj['x'],c['density'])),
              full_curve_sha256=hashlib.sha256(c['density'].tobytes()).hexdigest())
            rows.append(entry);subset.append(entry)
        summary.append(dict(variant=title,display_low=lo,display_high=hi,n_cells=sum(x['n_cells'] for x in subset),
          n_left=sum(x['n_left'] for x in subset),n_right=sum(x['n_right'] for x in subset),
          max_left_boundary_height_fraction=max(x['left_boundary_height_fraction'] for x in subset),
          max_right_boundary_height_fraction=max(x['right_boundary_height_fraction'] for x in subset)))
    out=HERE/'output/pdf/ramp_sox10_left_margin_comparison.pdf';fig.savefig(out,dpi=600)
    fig.savefig(HERE/'output/ramp_sox10_left_margin_comparison.svg');plt.close(fig)
    doc=fitz.open(out);doc[0].get_pixmap(matrix=fitz.Matrix(2.4,2.4)).save(HERE/'output/ramp_sox10_left_margin_comparison.png')
    pd.DataFrame(rows).to_csv(HERE/'per_ridge_audit.csv',index=False)
    pd.DataFrame(summary).to_csv(HERE/'range_summary.csv',index=False)
    for p,oldsha in {**sources,**before}.items():assert h.sha(Path(p))==oldsha,p
    for b in doc[0].get_text('dict')['blocks']:
      for l in b.get('lines',[]):
       for s in l['spans']:
        x0,y0,x1,y1=s['bbox'];assert x0>=0 and y0>=0 and x1<=doc[0].rect.width and y1<=doc[0].rect.height,s['text']
    (HERE/'PROVENANCE.json').write_text(json.dumps(dict(status='preview_only_not_installed',right_limit_fixed=hi,
       same_cells=True,same_bandwidth=True,same_density_arrays=True,left_xlim_only_changed=True,
       density_endpoints_not_forced_to_zero=True,manuscript_unchanged=before,source_hashes=sources,summaries=summary),indent=2)+'\n')
    print(pd.DataFrame(summary).to_string(index=False))
if __name__=='__main__':main()
