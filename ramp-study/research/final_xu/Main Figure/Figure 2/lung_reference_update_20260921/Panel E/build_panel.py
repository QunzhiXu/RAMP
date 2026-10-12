from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *

def draw(fig,box):
    title(fig,box,'E','Fixed-gate transfer')
    data=csv(out('E')/'density_coordinates.csv')
    stats=csv(out('E')/'gate_statistics.csv')
    manifest=json.loads(track(out('E')/'data_manifest.json').read_text())
    assert len(data)==14400 and len(stats)==12
    assert all(sha(p)==h for p,h in manifest['input_sha256'].items())
    colors={'Negative':'#788D9C','Positive':'#C58A34'}
    for x,label in [(99,'Negative'),(121,'Positive')]:
        mm_text(fig,box,x-2.2,3.1,'●',fontsize=6,color=colors[label],ha='right',va='center')
        mm_text(fig,box,x,3.1,label,fontsize=6.4,ha='left',va='center')
    rows=[('Raw','Triana AbSeq',12),('Raw','BM CODEX',22),
          ('RAMP','Triana AbSeq',35),('RAMP','BM CODEX',45)]
    for col,marker in enumerate(['CD3','CD4','CD8']):
        left=8.5+44.5*col
        mm_text(fig,box,left+17.25,9,marker,fontsize=8,weight='bold',ha='center')
        for row,(method,dataset,top) in enumerate(rows):
            m=data[data.marker.eq(marker)&data.method.eq(method)]
            xmin,xmax=m.asinh_value.min(),m.asinh_value.max()
            ymax=m.balanced_density.max()*1.42
            stat=stats[stats.marker.eq(marker)&stats.method.eq(method)&stats.dataset.eq(dataset)].iloc[0]
            ax=mm_axes(fig,box,(left,top,34.5,7))
            ax.axvspan(stat.threshold_asinh,xmax,color='#008866',alpha=.045,zorder=0)
            for label,color in colors.items():
                d=m[m.dataset.eq(dataset)&m.reference_class.eq(label)].sort_values('asinh_value')
                ax.fill_between(d.asinh_value,d.balanced_density,color=color,alpha=.34,lw=0)
                ax.plot(d.asinh_value,d.balanced_density,color=color,lw=.8)
            ax.axvline(stat.threshold_asinh,color=INK,ls=(0,(3,2)),lw=.7)
            ax.set_xlim(xmin,xmax);ax.set_ylim(0,ymax);ax.set_yticks([])
            ax.spines[['top','left','right']].set_visible(False);ax.spines['bottom'].set_color('#7A858C')
            ticks=np.arange(np.ceil(xmin/2)*2,xmax+.001,2) if method=='Raw' else np.arange(np.ceil(xmin),xmax+.001,1)
            ax.set_xticks(ticks)
            ax.tick_params(axis='x',length=1.8,pad=1,labelsize=6.3,labelbottom=row in [1,3])
            ax.text(.02,.98,'AbSeq' if dataset=='Triana AbSeq' else 'CODEX',transform=ax.transAxes,
                ha='left',va='top',fontsize=6.3,weight='bold')
            ax.text(.98,.98,f'BA = {stat.balanced_accuracy:.3f}',transform=ax.transAxes,
                ha='right',va='top',fontsize=6.3)
            if row==3:ax.set_xlabel(f'asinh({marker})',fontsize=6.5,labelpad=1.5)
    for y,label in [(20.5,'Raw'),(43.5,'RAMP')]:
        mm_text(fig,box,3,y,label,rotation=90,ha='center',va='center',fontsize=7.4,
            weight='bold',color='#008866' if label=='RAMP' else INK)
    return {'layout':'Four rows x three columns; one marker per column: CD3, CD4, CD8.',
        'row_order':['Raw AbSeq','Raw CODEX','RAMP AbSeq','RAMP CODEX'],
        'statistics':stats.to_dict('records'),'data_manifest':str(out('E')/'data_manifest.json'),
        'source_units':3,'target_units':12,
        'display':'Source and target share x/y ranges within each marker and method. All curves use arcsinh with cofactor 1.',
        'legend':'Colors indicate predefined annotation-reference positive/negative populations; not gate assignments.',
        'selection':'Existing illustrative bone-marrow AbSeq-to-CODEX direction expanded to all three existing CD3/CD4/CD8 tasks, without donor selection or gate refitting.'}

if __name__=='__main__':run_panel('E',draw)
