"""Read-only, selected plotting-data bridges; no plotting or scientific fitting.

The paper/figure identities are pinned by inspect_assets. This is not a general
validator for arbitrary reports, nor a claim that every plotted pixel was rebuilt.
"""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

from inspect_assets import HERE, ROOT, STUDY, PAPER, rec, sha, read, write, prior, source_index

INSPECTION = HERE/'runs/inspection_20261011'
TOLERANCE = 1e-14


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keyed_check(actual, expected, keys, values, rows, tolerance=TOLERANCE):
    """Exact identity/coverage; absolute (not relative) tolerance on finite values."""
    require(type(rows) is int and rows > 0, 'Invalid expected row count')
    for frame in (actual, expected):
        require(set(keys+values) <= set(frame.columns), 'Missing columns')
        require(len(frame) == rows, 'Row count differs')
        require(not frame[keys].isna().any().any(), 'Missing identity')
        require(not frame.duplicated(keys).any(), 'Duplicate identity')
    a = actual.set_index(keys)[values].sort_index()
    e = expected.set_index(keys)[values].sort_index()
    require(a.index.equals(e.index), 'Identity set differs')
    maximum = 0.0
    for column in values:
        if pd.api.types.is_numeric_dtype(a[column]) and pd.api.types.is_numeric_dtype(e[column]):
            av, ev = a[column].to_numpy(float), e[column].to_numpy(float)
            require(np.isfinite(av).all() and np.isfinite(ev).all(), 'Non-finite value')
            delta = float(np.max(np.abs(av-ev)))
            require(delta <= tolerance, 'Numeric value differs: '+column)
            maximum = max(maximum, delta)
        else:
            require(a[column].equals(e[column]), 'Text value differs: '+column)
    return {'rows': rows, 'keys': keys, 'value_columns': values,
            'max_abs_difference': maximum, 'absolute_tolerance': tolerance}


def relative_profile(frame):
    d = frame.copy()
    d['quality'] = np.where(d.direction.eq('higher'), d['mean'], -d['mean'])
    require(set(d.direction) <= {'higher', 'lower'}, 'Unknown metric direction')
    groups = d.groupby(['benchmark', 'metric']).quality
    best, worst = groups.transform('max'), groups.transform('min')
    d['relative_performance'] = 1-np.divide(best-d.quality, best-worst,
        out=np.zeros(len(d)), where=(best-worst).ne(0))
    return d.groupby(['benchmark','family','method']).relative_performance.mean().groupby(
        ['family','method']).mean().reset_index()


class Evidence:
    def __init__(self):
        self.sources = {}
        self.checks = {}

    def track(self, p):
        p = Path(p)
        if not p.is_absolute():
            p = ROOT/p
        self.sources[str(p.relative_to(ROOT))] = rec(p)
        return p

    def csv(self, p):
        return pd.read_csv(self.track(p), float_precision='round_trip', low_memory=False)

    def json(self, p):
        return read(self.track(p))

    def check(self, name, actual, expected, keys, values, rows, tolerance=TOLERANCE):
        require(name not in self.checks, 'Duplicate check')
        self.checks[name] = keyed_check(actual, expected, keys, values, rows, tolerance)

    def unchanged(self):
        for name, record in self.sources.items():
            require(rec(ROOT/name) == record, 'Consumed source changed: '+name)


def numerical_bridges(e):
    fig2 = ROOT/'final_xu/Main Figure/Figure 2/lung_reference_update_20260921'
    rank = e.csv(STUDY/'benchmarks/integration_evaluation/validation/replay_numerical_20261011/main23_ranks.csv')
    require(len(rank) == 966 and rank.metric.nunique() == 23, 'Main23 shape differs')
    for panel, metric in [('A','louvain_celltype_ari'), ('B','umap_celltype_asw_macro')]:
        e.check('Figure2'+panel, e.csv(fig2/f'Panel {panel}/plotted_values.csv'),
                rank[rank.metric.eq(metric)], ['benchmark','method','metric'],
                ['family','direction','mean','rank'],42)
    e.check('Figure2C', e.csv(fig2/'Panel C/plotted_values.csv'), relative_profile(rank),
            ['family','method'],['relative_performance'],24)
    bycase = rank.groupby(['benchmark','family','method'])['rank'].mean().groupby(
        ['benchmark','method']).mean().rename('mean_rank').reset_index()
    e.check('Figure2D_points', e.csv(fig2/'Panel D/plotted_values.csv'), bycase,
            ['benchmark','method'],['mean_rank'],42)
    e.check('Figure2D_diamonds', e.csv(fig2/'Panel D/equal_benchmark_means.csv'),
            bycase.groupby('method').mean_rank.mean().reset_index(), ['method'],['mean_rank'],6)
    values = STUDY/'benchmarks/value_evaluation/runs/value_execution_20261011/runtime/tree/Main Figure/fastmnn_reconstructed_audit_20260914/value_evaluation/summary'
    gate = e.csv(values/'gate_task_means.csv')
    e.check('Figure2F_points',e.csv(fig2/'Panel F/plotted_values.csv'),gate,
            ['benchmark','marker','method'],['balanced_accuracy','f1','sensitivity','specificity','precision','auc'],66)
    e.check('Figure2F_diamonds',e.csv(fig2/'Panel F/equal_benchmark_means.csv'),
            gate.groupby(['benchmark','method']).balanced_accuracy.mean().groupby('method').mean().reset_index(),
            ['method'],['balanced_accuracy'],6)
    pieces = []
    for n, count in [(8,378),(9,588)]:
        d = e.csv(f'final_xu/Supplementary Figures/Figure 2 complete benchmark/preview_v1/S{n}_plotted_values.csv')
        e.check(f'S{n}',d,rank[rank.metric.isin(d.metric.unique())].rename(columns={'mean':'score'}),
                ['benchmark','method','metric'],['score'],count)
        pieces.append(d[['benchmark','method','metric']])
    require(len(pd.concat(pieces).drop_duplicates()) == 966, 'S8/S9 do not partition main23')

    heads = e.csv(STUDY/'benchmarks/trainability/head_retraining_full/validation/combined_aggregation_20261011/primary_means.csv')
    require(len(heads) == 1140, 'Head summary differs')
    # The accepted summary also contains 150 ALL rows; these are already
    # aggregated, not an eighth/seventh benchmark to average into figure points.
    require(int(heads.task.eq('ALL').sum()) == 150, 'Aggregate row count differs')
    heads = heads[heads.task.ne('ALL')]
    f3 = ROOT/'final_xu/figure3_readability_preview_20260924/data'
    groupkeys = ['task_type','task','metric','method']
    averaged = heads.groupby(groupkeys).score.mean().reset_index()
    for panel, kind, metrics, count in [('b','J1',['balanced_accuracy','macro_f1'],140),
                                         ('c','J2',['effective_spearman','mae'],120)]:
        d = averaged[averaged.task_type.eq(kind)&averaged.metric.isin(metrics)]
        e.check('Figure3'+panel.upper(),e.csv(f3/f'panel_{panel}_per_benchmark_scores.csv'),d,
                groupkeys,['score'],count)
    overall = averaged.groupby(['task_type','metric','method']).score.mean().reset_index()
    overall['panel'] = overall.task_type.map({'J1':'B','J2':'C'})
    plotted = e.csv(f3/'overall_means.csv')
    ref = overall.merge(plotted[['panel','metric','method']],on=['panel','metric','method'],validate='one_to_one')
    e.check('Figure3_overall',plotted,ref.rename(columns={'score':'mean'}),
            ['panel','metric','method'],['mean'],40)
    tech = {'bone_marrow','tnbc','melanoma','lung_cancer'}
    for n, count, kind, metrics, limits in [(10,240,'J1',['balanced_accuracy','macro_f1'],(.2,1)),
            (11,180,'J1',['balanced_accuracy','macro_f1'],(.2,1)),
            (12,180,'J2',['effective_spearman'],(.05,.55)), (13,180,'J2',['mae'],(.18,.46))]:
        plotted = e.csv(f'final_xu/Supplementary Figures/Figure 3 model benchmark breakdown/final_20260922/S{n}_plotted_values.csv')
        ref = heads[heads.task_type.eq(kind)&heads.metric.isin(metrics)]
        if n in (10,11):
            ref = ref[ref.task.isin(tech) if n == 10 else ~ref.task.isin(tech)]
        e.check(f'S{n}',plotted,ref,['task','family','metric','method'],['score'],count)
        require(plotted.score.between(*limits).all(), 'Plotted point outside fixed axes')
        e.checks[f'S{n}']['all_points_within_axes'] = list(limits)


def hnscc_bridge(e):
    base = ROOT/'final_xu/figure_HNSCC'
    polish = base/'big_figure/polish_20260913'
    final = polish/'CEF_D_colorbar_aligned_20260914'
    layout = e.json(final/'LAYOUT_AUDIT.json')
    for panel, item in layout['placements'].items():
        require(sha(e.track(item['source'])) == item['sha256'], 'Panel source changed: '+panel)
    current_c = polish/'horizontal_E_spatial_left/panel_C/panel_C_alignment_compact.pdf'
    require(Path(layout['placements']['C']['source']) == current_c, 'Unexpected Panel C route')
    horizontal = e.json(polish/'horizontal_E_spatial_left/LAYOUT_AUDIT.json')
    coords = base/'alignment/umap/full_94488_umap_coordinates.csv.gz'
    require(sha(e.track(coords)) == horizontal['inputs_sha256'][str(coords)], 'UMAP source binding differs')
    for p in [polish/'horizontal_E_spatial_left/build_preview.py',
              base/'big_figure/combined_ABCDE_two_rows/build_combined_two_rows.py']:
        e.track(p)  # Static build_c trace reviewed, never imported/executed.
    run = STUDY/'applications/hnscc/runs/rebuild_20261011/tree/figure_HNSCC'
    recovery = STUDY/'applications/hnscc/runs/umap_0511_diagnostic_20261011/tree/figure_HNSCC/alignment/umap'
    newcoords = e.track(recovery/'full_94488_umap_coordinates.csv.gz')
    with gzip.open(coords,'rb') as f, gzip.open(newcoords,'rb') as g:
        require(f.read() == g.read(), 'Decompressed HNSCC coordinates differ')
    d = e.csv(coords)
    require(len(d) == 94488 and len(d.columns) == 11, 'HNSCC coordinate dimensions differ')
    e.checks['Figure7C'] = {'rows':94488,'columns':11,'decompressed_bytes_equal':True,
                           'route':'full alignment UMAP, not Rahim transfer-program UMAP',
                           'positive_reproduction_version':'umap-learn 0.5.11',
                           'historical_environment_proven':False}
    e.track(STUDY/'applications/hnscc/validation/umap_0511_recovery_20261011/report.json')
    e.check('Figure7D', e.csv(final/'panel_D/plotted_donor_values.csv'),e.csv(run/'display_values/panel_D.csv'),
            ['marker_raw','donor'],['association'],18,0)
    donor = e.csv(run/'molecular30/donor_contrasts.csv')
    keys = ['donor','modality','marker','label']
    require(donor.fraction.eq(.3).all(), 'Unexpected molecular selection')
    melted = donor.melt(id_vars=keys,value_vars=['low','high'],var_name='group',value_name='value')
    e.check('Figure7E', e.csv(polish/'CEF_compact/panel_E/plotted_donor_values.csv'),melted,
            keys+['group'],['value'],36)
    old = e.csv(polish/'larger_LAG3_spatial/panel_F/plotted_cells.csv')
    e.check('Figure7F',old,e.csv(run/'fixed_roi/large_roi_cells.csv'),['cell_uid'],
            [c for c in old.columns if c != 'cell_uid'],343,0)
    require(set(old.patient) == {29770} and set(old.unique_region) == {'521_S1reg012'},'ROI differs')


def source_indexes(e):
    reports = {}
    for name in ['source_census','curated_sources']:
        items = e.json(STUDY/f'inventory_20261010/evidence/{name}.json')['files']
        checked = 0
        for item in items:
            if item.get('sha256'):
                require(rec(ROOT/item['path']) == {'bytes':item['bytes'],'sha256':item['sha256']},
                        'Historical index changed: '+item['path'])
                checked += 1
        reports[name] = {'records':len(items),'hashed_files_checked':checked}
    return reports


def run(output):
    out = Path(output).absolute()
    require(out.is_relative_to(HERE/'runs') and not out.is_symlink() and
            not any(p.is_symlink() for p in out.parents), 'Invalid output root')
    require(not out.exists(), 'Output already exists')
    require(shutil.disk_usage(HERE).free > 20_000_000_000,'Insufficient free space')
    source_index()
    before = prior()
    require(before == read(INSPECTION/'PRIOR.json'), 'Prior study files changed since inspection')
    out.mkdir()
    try:
        write(out/'REQUEST.json',{'controller':rec(Path(__file__)), 'asset_inspector':rec(HERE/'inspect_assets.py'),
            'inspection_complete':rec(INSPECTION/'COMPLETE.json'), 'python':sys.version,
            'pandas':pd.__version__,'numpy':np.__version__, 'no_fitting':True,
            'scope':'selected plotting-data bridges and source protection, not complete figure regeneration'})
        e = Evidence()
        numerical_bridges(e)
        hnscc_bridge(e)
        indexes = source_indexes(e)
        e.unchanged()
        require(prior() == before,'Prior study files changed')
        source_index()
        report = {'status':'selected_numeric_bridges_passed_not_full_figure_acceptance',
                  'checks':e.checks,'sources':e.sources,'original_indexes':indexes,
                  'prior_small_study_files_unchanged':len(before),
                  'scope_excludes':['raw microscopy extraction','plotting script reexecution',
                      'every scatter/ridge pixel reconstruction','clinical or model refitting','publication approval']}
        write(out/'report.json',report)
        write(out/'COMPLETE.json',{'status':report['status'],'report':rec(out/'report.json'),
                                 'request':rec(out/'REQUEST.json')})
        return {'checks':len(e.checks),'sources':len(e.sources),'status':report['status']}
    except BaseException as exc:
        write(out/'FAILED.json',{'error':repr(exc)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.out),indent=2))
