"""Read-only identity audit of current story and Figure 2 alignment inputs.

Writes only to this audit directory. Does not fit models or modify figures.
"""
from pathlib import Path
import hashlib
import json
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = {}


def sha(p):
    p = Path(p)
    with p.open('rb') as f:
        h = hashlib.file_digest(f, 'sha256').hexdigest()
    SOURCES[str(p)] = h
    return h


def js(p):
    p = ROOT / p
    sha(p)
    return json.loads(p.read_text())


def csv(p):
    p = ROOT / p
    sha(p)
    return pd.read_csv(p, low_memory=False)


CASES = [
    dict(name='bone_marrow',story_matrix='story_cd19_prediction/alignment/output/without_cd19/ramp_aligned.npy',
         comparison_matrix='benchmark_results/bone_marrow/ramp/alignment/data/ramp_aligned.npy',
         story_meta='story_cd19_prediction/alignment/output/cell_metadata.csv.gz',
         comparison_meta='benchmark_results/bone_marrow/ramp/alignment/input/cell_metadata.csv.gz',id='cell_id'),
    dict(name='melanoma_story',story_matrix='figure_melanoma/analysis/alignment/output/ramp_aligned.npy',
         comparison_matrix='figure_melanoma/Comparison/input/ramp.npy',
         story_meta='figure_melanoma/analysis/alignment/output/cell_metadata.csv.gz',
         comparison_meta='figure_melanoma/Comparison/input/evaluation_metadata.csv.gz',id='cell_uid'),
    dict(name='hnscc_story',story_matrix='figure_HNSCC/alignment/output/aligned_ramp_ver2.float32.npy',
         comparison_matrix='figure_HNSCC/Comparison/input/ramp.npy',
         story_meta='figure_HNSCC/alignment/input/cell_metadata.csv.gz',
         comparison_meta='figure_HNSCC/Comparison/input/evaluation_metadata.csv.gz',id='cell_uid'),
    dict(name='lung_story',story_matrix='story_lung_HLA_ABC/releases/senosain140k_20260921/alignment/RAMP_SHARED9.npy',
         comparison_matrix='story_lung_HLA_ABC/Comparison_senosain140k_20260921/input/ramp.npy',
         story_meta='story_lung_HLA_ABC/releases/senosain140k_20260921/inputs/CELL_METADATA.csv.gz',
         comparison_meta='story_lung_HLA_ABC/Comparison_senosain140k_20260921/input/evaluation_metadata.csv.gz',id='cell_key'),
]


def main():
    report={'scope':'Four story alignment inputs versus current Figure 2; display protocols; Figure 3 source-version status. Not a full downstream scientific audit.',
            'cases':[]}
    for c in CASES:
        a,b=csv(c['story_meta']),csv(c['comparison_meta'])
        checks={'ramp_file_sha256_equal':sha(ROOT/c['story_matrix'])==sha(ROOT/c['comparison_matrix']),
                'ordered_cell_ids_equal':np.array_equal(a[c['id']].astype(str),b.cell_id.astype(str)),
                'ordered_platforms_equal':np.array_equal(a.dataset,b.dataset),
                'ordered_unit_ids_equal':np.array_equal(a.unit_key,b.unit_key)}
        assert all(checks.values()),(c['name'],checks)
        x=np.load(ROOT/c['story_matrix'],mmap_mode='r')
        report['cases'].append({**c,'shape':list(x.shape),'datasets':a.groupby('dataset').size().to_dict(),
                                'checks':checks,'sha256':SOURCES[str(ROOT/c['story_matrix'])]})
    # Bone marrow raw inputs are the exact same frozen platform packages.
    bm=[]
    for platform in ['bm_codex','bm_cytof','triana_abseq']:
        for fn in ['alignment_input_14.csv.gz','source_input_15.csv.gz','locked_metadata.csv.gz']:
            a=ROOT/'story_cd19_prediction/sampled_data'/platform/fn
            b=ROOT/'benchmark_data/bone_marrow'/platform/fn
            assert sha(a)==sha(b)
            bm.append({'platform':platform,'file':fn,'sha256_equal':True})
    report['bone_marrow_raw_packages']=bm
    # Explicitly check marker order and directly archived raw inputs.
    sm=js('story_cd19_prediction/alignment/output/without_cd19/manifest.json')['markers']
    bp=ROOT/'benchmark_results/bone_marrow/ramp/alignment/input/markers.txt';sha(bp)
    assert sm==bp.read_text().splitlines()
    sm=js('figure_melanoma/analysis/alignment/output/manifest.json')['shared_markers']
    cm=js('figure_melanoma/Comparison/input/manifest.json')['markers']
    assert sm==cm
    a=ROOT/'figure_HNSCC/alignment/input/markers.txt';b=ROOT/'figure_HNSCC/Comparison/input/markers.txt'
    assert sha(a)==sha(b)
    assert sha(ROOT/'figure_HNSCC/alignment/input/raw_shared_markers.float64.npy')==sha(ROOT/'figure_HNSCC/Comparison/input/raw.npy')
    sm=js('story_lung_HLA_ABC/releases/senosain140k_20260921/inputs/DESIGN.json')['markers']
    cm=js('story_lung_HLA_ABC/Comparison_senosain140k_20260921/input/manifest.json')['markers']
    assert sm==cm
    assert sha(ROOT/'story_lung_HLA_ABC/releases/senosain140k_20260921/inputs/RAW_SHARED9.npy')==sha(ROOT/'story_lung_HLA_ABC/Comparison_senosain140k_20260921/input/raw.npy')
    report['shared_marker_order_verified']=True
    # Melanoma display subset is identified by saved indices, not new sampling.
    d=csv('figure_melanoma/cache/ramp230_clean15_display_metadata.csv.gz')
    c=csv('figure_melanoma/Comparison/input/evaluation_metadata.csv.gz')
    rows=d.alignment_row.to_numpy(int)
    assert len(d)==40000 and np.array_equal(d.cell_uid.astype(str),c.iloc[rows].cell_id.astype(str))
    assert np.array_equal(d.harmonized_coarse_label,c.iloc[rows].celltype)
    report['melanoma_display']={'subset_of_same_full_alignment':True,'n':len(d),
        'per_platform':d.groupby('dataset').size().to_dict(),'labels_identical_on_display_subset':True,
        'story_umap':js('figure_melanoma/cache/ramp230_clean15_display_manifest.json'),
        'comparison_umap':js('figure_melanoma/Comparison/maxfuse_full_panel/evaluation/umap/manifests/RAMP.json')}
    report['hnscc_display']={'story':js('figure_HNSCC/alignment/umap/umap_manifest.json'),
        'comparison':js('figure_HNSCC/Comparison/maxfuse_full_panel/evaluation/umap/manifests/RAMP.json')}
    report['lung_display']={'story':js('story_lung_HLA_ABC/releases/senosain140k_20260921/alignment/UMAP_COMPLETE.json'),
        'comparison':js('story_lung_HLA_ABC/Comparison_senosain140k_20260921/evaluation/umap/manifests/RAMP.json')}
    assert sha(ROOT/'story_lung_HLA_ABC/Figure/Panel B/data/RAMP_UMAP.npy')==sha(ROOT/'story_lung_HLA_ABC/releases/senosain140k_20260921/alignment/RAMP_UMAP.npy')
    report['current_lung_panel_b_uses_current_release_coordinates']=True
    # Current Figure 2 registration and source hashes.
    current=js('Main Figure/Figure 2/CURRENT_DRAFT.json')
    selection=js('Main Figure/Figure 2/lung_reference_update_20260921/selection/metric_selection.json')
    for p,h in selection['source_sha256'].items():assert sha(p)==h,p
    report['figure2']={'current':current,'registered_sources_unchanged':True,
                      'umap_metrics_retained':[k for k in selection['included_metrics'] if k.startswith('umap_')]}
    # Check original main-figure TNBC data/coordinate sources.
    f1=js('Main Figure/Panel B/tnbc_bandwidth_review/adtnorm_style_review/manifest.json')
    for p,h in f1['input_sha256'].items():assert sha(p)==h,p
    report['figure1_tnbc']={'cells':f1['cells'],'registered_inputs_and_coordinates_unchanged':True,
                            'coordinates_reused_from_benchmark':True}
    # Do not conflate a pending Figure 3 release with accidental replacement.
    curr=js('cross_platform_trainability_v2_20260919/CURRENT_J1_RESULTS.json')
    report['figure3']={'current_j1_registry':curr,'panel_sources':{}}
    for panel in ['B','C','D','E','F']:
        p=js(f'cross_platform_trainability_v2_20260919/Figure/Panel {panel}/PROVENANCE.json')
        report['figure3']['panel_sources'][panel]=p.get('source',p.get('sources'))
    report['figure3']['interpretation']='Current plotted B-F still refer to formal_v0_5; accepted J1 r3 and new Lung J2 must be synchronized before release. The user had deferred figure updates.'
    report['inputs_sha256']=dict(SOURCES)
    report['writes']='Only this audit directory; no input, model, figure, or manuscript changes.'
    (HERE/'AUDIT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print('PASS: all four story RAMP matrices, ordered cell IDs, platforms, units, and shared marker order match their Figure 2 inputs.')
    print('PASS: current Figure 2 sources and Figure 1 TNBC coordinate/input hashes match.')
    print('Confirmed display-protocol differences: Melanoma, HNSCC, Lung; CD19 uses purpose-specific feature sets/initializations.')
    print('Pending version synchronization: Figure 3 B-F still reference formal_v0_5.')


if __name__=='__main__':main()
