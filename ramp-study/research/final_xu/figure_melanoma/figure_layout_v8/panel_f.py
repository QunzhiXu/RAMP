"""Two horizontal patient-level box plots, each with its own numeric scale."""
def draw_panel_f(context,fig,rect):
    np=context['np'];axis=context['axis'];text=context['text'];title=context['title']
    ink=context['INK'];data=context['TWO_METRICS']
    x,y,w,h=rect
    title(fig,rect,'F','State enrichment')
    text(fig,x+.02,y+.30,'14 patients',fontsize=9.1,color=ink)
    offsets=np.random.default_rng(20260914).uniform(-.115,.115,14)
    specs=[
        ('AUROC','AUROC','#3F78B5',.66,.89,(0,1),[0,.5,1],['0','0.5','1.0'],.5),
        ('Enrichment','Target enrichment (top 20%)','#43A295',1.83,2.06,(0,3.5),[0,1,2,3],['0×','1×','2×','3×'],1.0),
    ]
    summaries={};axes=[];labels=[]
    for key,label,color,ty,ay,limits,ticks,ticknames,reference in specs:
        labels.append(text(fig,x+.16,y+ty,label,fontsize=9.4,fontweight='bold'))
        ax=axis(fig,x+.16,y+ay,w-.28,.55)
        values=data.loc[data.metric.eq(key)].sort_values('patient').value.to_numpy(float)
        assert len(values)==14 and np.isfinite(values).all()
        ax.axvline(reference,color='#88949C',lw=.9,ls=(0,(3,3)),zorder=0)
        ax.boxplot([values],positions=[0],widths=.40,orientation='horizontal',
            patch_artist=True,showfliers=False,manage_ticks=False,
            boxprops=dict(facecolor=color+'30',edgecolor=color,linewidth=1.1),
            medianprops=dict(color=ink,linewidth=1.4),
            whiskerprops=dict(color=color,linewidth=1),capprops=dict(color=color,linewidth=1))
        ax.scatter(values,offsets,s=22,c=color,edgecolor='white',lw=.45,zorder=4)
        ax.set(xlim=limits,ylim=(-.34,.34),xticks=ticks,yticks=[])
        ax.set_xticklabels(ticknames,fontsize=8.6)
        ax.tick_params(axis='x',length=2,pad=3)
        ax.spines[['top','right','left']].set_visible(False)
        ax.spines['bottom'].set_linewidth(.8)
        summaries[key]=dict(n=14,median=float(np.median(values)),mean=float(values.mean()),x_limits=list(limits),reference=reference)
        axes.append(ax)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    fw,fh=fig.get_size_inches()
    local=(x/fw,1-(y+h)/fh,(x+w)/fw,1-y/fh)
    local_texts=[t for t in fig.texts if x/fw <= t.get_position()[0] <= (x+w)/fw and 1-(y+h)/fh <= t.get_position()[1] <= 1-y/fh]
    for artist in local_texts+[t for ax in axes for t in ax.get_xticklabels()]:
        box=artist.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        assert box.x0>=local[0]-.002 and box.x1<=local[2]+.002 and box.y0>=local[1] and box.y1<=local[3],(artist.get_text(),box)
    context['DRAW_AUDIT']['F']={
        'n_patients':14,'n_points':28,'metrics':summaries,
        'title':'State enrichment','box_orientation':'horizontal',
        'arrangement':'two vertically stacked axes with independent x scales',
        'boxplot':'median, quartiles, whiskers to 1.5 IQR; every patient shown',
        'enrichment_definition':'precision_top20 divided by each patient target prevalence',
        'call_rule':'top 20% of candidate CD8 in whole held-out patient',
        'same_evaluable_patients_as_v7':True,'numeric_values_or_threshold_changed':False,
        'all_point_and_tick_labels_inside_panel':True}
