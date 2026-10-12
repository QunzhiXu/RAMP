"""Prepare densities for the existing three frozen bone-marrow gate tasks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from scipy.stats import gaussian_kde

SOURCE='Triana AbSeq'
TARGET='BM CODEX'
MARKERS=['CD3','CD4','CD8']
T_LABELS={'CD4 T','CD8 T','CD8− T','Other T / CD8−','Other T / NKT'}
NEG_CD8={'CD4 T','CD8− T','Other T / CD8−'}
AMBIGUOUS={'Non-joint','Mix','Unclassified / technical','Unannotated','nan'}

def balanced_weights(positive,units):
    weights=np.zeros(len(positive))
    for cls in [False,True]:
        groups=np.unique(units[positive==cls])
        for unit in groups:
            mask=(positive==cls)&(units==unit)
            weights[mask]=.5/(len(groups)*mask.sum())
    assert np.isclose(weights.sum(),1)
    return weights

def main():
    base=ROOT/'benchmark_results/bone_marrow/ramp/alignment'
    meta=csv(base/'input/cell_metadata.csv.gz',low_memory=False)
    markers=track(base/'input/markers.txt').read_text().splitlines()
    matrices={'Raw':np.load(track(base/'input/raw.npy'),mmap_mode='r'),
        'RAMP':np.load(track(base/'data/ramp_aligned.npy'),mmap_mode='r')}
    assert all(m.shape==(len(meta),len(markers)) for m in matrices.values())
    col='celltype_joint_evaluation' if 'celltype_joint_evaluation' in meta else 'celltype'
    labels=meta[col].fillna('Unannotated').astype(str)
    units=meta.unit_key.astype(str).to_numpy()
    gates=csv(FIG2/'value_evaluation_20260914/results/bone_marrow/source_gates.csv')
    eligibility=csv(FIG2/'value_evaluation_20260914/results/bone_marrow/gate_eligibility.csv')
    directions=csv(FIG2/'value_evaluation_20260914/summary/gate_direction_means.csv')
    olddensity=csv(PRIOR/'Panel E/density_coordinates.csv')
    oldstats=csv(PRIOR/'Panel E/gate_statistics.csv')
    density=[];statistics=[];display=[]
    for marker in MARKERS:
        if marker=='CD3':
            included=~labels.isin(AMBIGUOUS).to_numpy();positive=labels.isin(T_LABELS).to_numpy()
        elif marker=='CD4':
            included=labels.isin(['CD4 T','CD8 T']).to_numpy();positive=labels.eq('CD4 T').to_numpy()
        else:
            included=labels.isin({'CD8 T'}|NEG_CD8).to_numpy();positive=labels.eq('CD8 T').to_numpy()
        masks={}
        for dataset in [SOURCE,TARGET]:
            e=eligibility[eligibility.marker.eq(marker)&eligibility.dataset.eq(dataset)&eligibility.primary_eligible]
            masks[dataset]=np.flatnonzero(included&meta.dataset.eq(dataset).to_numpy()&np.isin(units,e.unit_key.astype(str)))
        assert len(np.unique(units[masks[SOURCE]]))==3
        assert len(np.unique(units[masks[TARGET]]))==12
        for method in ['Raw','RAMP']:
            g=gates[gates.marker.eq(marker)&gates.source_dataset.eq(SOURCE)&gates.method.eq(method)].iloc[0]
            threshold=float(g.threshold);gate_x=float(np.arcsinh(threshold))
            values=np.asarray(matrices[method][:,markers.index(marker)],dtype=float)
            all_x=np.arcsinh(values[np.concatenate(list(masks.values()))])
            lo,hi=float(all_x.min()),float(all_x.max());pad=.04*(hi-lo)
            xx=np.linspace(0 if method=='Raw' else lo-pad,hi+pad,600)
            for dataset in [SOURCE,TARGET]:
                ids=masks[dataset];y=positive[ids];w=balanced_weights(y,units[ids])
                v=values[ids];x=np.arcsinh(v)
                ba=float(w[(v>=threshold)==y].sum())
                expected=float(g.source_balanced_accuracy) if dataset==SOURCE else float(directions[
                    directions.benchmark.eq('bone_marrow')&directions.marker.eq(marker)&directions.method.eq(method)&
                    directions.source_dataset.eq(SOURCE)&directions.target_dataset.eq(TARGET)].iloc[0].balanced_accuracy)
                assert np.isclose(ba,expected,atol=1e-12),(marker,method,dataset,ba,expected)
                stat=dict(benchmark='bone_marrow',marker=marker,method=method,dataset=dataset,
                    threshold_native=threshold,threshold_asinh=gate_x,balanced_accuracy=ba,
                    n_cells=len(ids),n_units=len(np.unique(units[ids])),n_positive=int(y.sum()),n_negative=int((~y).sum()))
                if marker=='CD3':
                    previous=oldstats[oldstats.method.eq(method)&oldstats.dataset.eq(dataset)].iloc[0]
                    assert np.isclose(previous.balanced_accuracy,ba,atol=1e-12)
                statistics.append(stat)
                for cls in [False,True]:
                    if marker=='CD3':
                        oldclass='T cells' if cls else 'Other cells'
                        old=olddensity[olddensity.method.eq(method)&olddensity.dataset.eq(dataset)&olddensity.reference_class.eq(oldclass)]
                        grid=old.asinh_cd3.to_numpy();z=old.balanced_density.to_numpy()
                    else:
                        grid=xx
                        z=.5*gaussian_kde(x[y==cls],weights=w[y==cls],bw_method=.22)(grid)
                    assert len(grid)==600 and np.isfinite(z).all()
                    density.extend(dict(marker=marker,method=method,dataset=dataset,
                        reference_class='Positive' if cls else 'Negative',asinh_value=float(a),balanced_density=float(b))
                        for a,b in zip(grid,z))
                display.append(pd.DataFrame({'cell_id':meta.cell_id.iloc[ids].to_numpy(),'marker':marker,
                    'method':method,'dataset':dataset,'unit_key':units[ids],
                    'reference_positive':y,'marker_native':v,'marker_asinh':x,'weight':w,
                    'gate_positive':v>=threshold}))
    save_data('E',pd.DataFrame(density),'density_coordinates.csv')
    save_data('E',pd.DataFrame(statistics),'gate_statistics.csv')
    save_data('E',pd.concat(display,ignore_index=True),'display_cells.csv.gz')
    assert all(sha(p)==h for p,h in READS.items())
    (out('E')/'data_manifest.json').write_text(json.dumps({'input_sha256':READS,'inputs_unchanged':True,
        'benchmark':'bone_marrow','markers':MARKERS,'source':SOURCE,'target':TARGET,
        'source_units':3,'target_units':12,'frozen_ba_verified':True,'cd3_curves_reused_exactly':True,
        'kde':'Gaussian KDE of asinh(value), cofactor 1, factor 0.22; class/donor balanced; full observed range.',
        'eligibility':'Exactly the frozen gate task definitions; all eligible units, no new donor or ROI selection.',
        'phenotypes':{'CD3':'T vs definite non-T cells','CD4':'CD4 T vs CD8 T cells',
            'CD8':'CD8 T vs annotated CD4 T / CD8-negative T cells'},
        'threshold_fits':'None; previously calibrated source gates reused unchanged.'},indent=2)+'\n')
    print(pd.DataFrame(statistics)[['marker','method','dataset','balanced_accuracy','n_cells']].to_string(index=False))

if __name__=='__main__':main()
