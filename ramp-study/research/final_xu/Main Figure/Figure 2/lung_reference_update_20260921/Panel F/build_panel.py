from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
def draw(fig,box):
    title(fig,box,'F','Cross-platform gating')
    source=ROOT/'Main Figure/fastmnn_reconstructed_audit_20260914/value_evaluation/summary'
    audit=json.loads(track(source/'manifest.json').read_text())
    assert audit['status']=='PASS' and audit['inputs_unchanged']
    assert all(sha(p)==h for p,h in audit['input_sha256'].items())
    d=csv(source/'gate_task_means.csv')
    directions=csv(source/'gate_direction_means.csv')
    keys=['benchmark','marker','method']
    counts=directions.groupby(keys).size().rename('n_directions').reset_index()
    d=d.merge(counts,on=keys,validate='one_to_one')
    means=csv(source/'gate_overall_equal_case.csv').set_index('method').balanced_accuracy
    assert len(d)==66 and d.n_directions.eq(6).all()
    old=csv(FIG2/'seven_benchmarks_main23_20260914/Panel F/plotted_values.csv')
    retained=d[~d.method.eq('fastMNN reconstructed')].sort_values(keys).reset_index(drop=True)
    old=old.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(retained[old.columns],old,check_exact=False,rtol=0,atol=1e-14)
    # Retain the exact saved five-method values; aggregation/parser roundoff
    # in the extension can otherwise differ by one floating-point ULP.
    d=pd.concat([old,d[d.method.eq('fastMNN reconstructed')]],ignore_index=True).sort_values(keys).reset_index(drop=True)
    old_means=csv(FIG2/'seven_benchmarks_main23_20260914/Panel F/equal_benchmark_means.csv').set_index('method').balanced_accuracy
    means=pd.concat([old_means,means.loc[['fastMNN reconstructed']]])
    recomputed=d.groupby(['benchmark','method']).balanced_accuracy.mean().groupby('method').mean()
    np.testing.assert_allclose(recomputed.sort_index(),means.sort_index(),rtol=0,atol=1e-14)
    save_data('F',d);save_data('F',means.reset_index(),'equal_benchmark_means.csv')
    ax=mm_axes(fig,box,(10,12,34,34))
    boxpoints(ax,d,VALUE_METHODS,'balanced_accuracy',means)
    ax.set_ylim(.45,1.005);ax.set_yticks([.5,.6,.7,.8,.9,1])
    ax.axhline(.5,ls='--',lw=.55,color='#96A0A6',zorder=1)
    ax.set_ylabel('Balanced accuracy',labelpad=3)
    mm_text(fig,box,27,9,'CD3 · CD4 · CD8',ha='center',fontsize=6.6)
    mm_text(fig,box,24,60,'11 marker tasks · 4 benchmarks',ha='center',fontsize=6.2)
    return {'metric':'Fixed-source-gate cross-platform balanced accuracy','points_per_method':11,
        'each_point':'A benchmark/marker task, averaged equally over six transfer directions, each direction averaged over eligible target units.',
        'black_diamond':'Four benchmarks weighted equally; not the unweighted mean of 11 tasks.',
        'box':'Median, IQR, 1.5 IQR whiskers of all 11 task values; all values overlaid.',
        'methods':VALUE_METHODS,'fastmnn':'Standard PCA-fastMNN reconstructed marker values; same source-only gate protocol and eligibility. A-D still use its corrected-PC integration representation.',
        'embedding_only_methods':'MaxFuse has no corresponding marker values and is not plotted.',
        'p_values':'None; tasks sharing donors are not treated as independent patient replicates.'}
if __name__=='__main__':run_panel('F',draw)
