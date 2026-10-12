"""Aggregate the prespecified frozen-matrix evaluation; no outcome-based selection."""
from evaluate_values import HERE, CASES, METHODS, sha, dump, save
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = HERE/'summary'
OUT.mkdir(exist_ok=True)
LABELS = {'bone_marrow': 'Bone marrow', 'tnbc': 'TNBC', 'lung_cancer': 'Lung cancer',
          'melanoma': 'Melanoma', 'Equal-case mean': 'Equal-case mean'}
METRICS = ['balanced_accuracy', 'f1', 'sensitivity', 'specificity', 'precision', 'auc']


def read_all(name):
    return pd.concat([pd.read_csv(HERE/'results'/c/name).assign(benchmark=c) for c in CASES], ignore_index=True)


def mean_by(frame, keys, columns):
    return frame.groupby(keys, observed=True, dropna=False)[columns].mean().reset_index()


def assert_matched(frame, keys):
    assert not frame.duplicated(keys+['method']).any(), 'Duplicate method/key'
    counts = frame.groupby(keys, observed=True, dropna=False).method.nunique()
    assert (counts == len(METHODS)).all(), counts[counts != len(METHODS)]
    assert set(frame.method) == set(METHODS)


complete = [json.loads((HERE/'results'/c/'COMPLETE.json').read_text()) for c in CASES]
assert all(x['status'] == 'PASS' and x['inputs_unchanged'] for x in complete)
assert all(x['protocol_sha256'] == sha(HERE/'PROTOCOL.md') for x in complete)
assert all(sha(p) == h for x in complete for p, h in x['input_hashes'].items())

# Gate portability: units -> directions -> tasks -> cases.
eligibility = read_all('gate_eligibility.csv')
sources = read_all('source_gates.csv')
scores = read_all('gate_target_unit_scores.csv')
primary = scores[scores.primary_eligible].copy()
key = ['benchmark', 'marker', 'source_dataset', 'target_dataset', 'target_unit']
assert_matched(primary, key)
assert primary[METRICS].notna().all().all()
assert np.allclose(primary.balanced_accuracy, (primary.sensitivity+primary.specificity)/2)
assert ((primary.tp+primary.fn) == primary.n_positive).all()
assert ((primary.tn+primary.fp) == primary.n_negative).all()
assert ((primary[METRICS] >= 0) & (primary[METRICS] <= 1)).all().all()
direction_keys = ['benchmark', 'marker', 'method', 'source_dataset', 'target_dataset']
directions = mean_by(primary, direction_keys, METRICS)
directions = directions.merge(primary.groupby(direction_keys).size().rename('n_target_units').reset_index(), on=direction_keys)
tasks = mean_by(directions, ['benchmark', 'marker', 'method'], METRICS)
tasks = tasks.merge(directions.groupby(['benchmark', 'marker', 'method']).size().rename('n_directions').reset_index())
case_gates = mean_by(tasks, ['benchmark', 'method'], METRICS)
overall_gates = mean_by(case_gates, ['method'], METRICS)
common_case = mean_by(tasks[tasks.marker.isin(['CD3', 'CD8'])], ['benchmark', 'method'], METRICS)
common_overall = mean_by(common_case, ['method'], METRICS)
for frame, filename in [(directions, 'gate_direction_means.csv'), (tasks, 'gate_task_means.csv'),
    (case_gates, 'gate_case_means.csv'), (overall_gates, 'gate_overall_equal_case.csv'),
    (common_case, 'gate_common_CD3_CD8_case_means.csv'), (common_overall, 'gate_common_CD3_CD8_overall.csv')]:
    save(frame, OUT/filename)
coverage = eligibility.groupby(['benchmark', 'marker', 'dataset']).agg(
    total_units=('unit_key', 'size'), eligible_units=('primary_eligible', 'sum'),
    total_annotated_positive=('n_positive', 'sum'), total_annotated_negative=('n_negative', 'sum')).reset_index()
save(coverage, OUT/'gate_coverage.csv')
save(sources[sources.status.ne('OK')], OUT/'insufficient_source_gates.csv')

# Paired changes from Raw at each identical target unit and source/target direction.
raw = primary[primary.method.eq('Raw')][key+METRICS].rename(columns={m:'raw_'+m for m in METRICS})
paired = primary.merge(raw, on=key, validate='many_to_one')
for m in METRICS:
    paired['change_'+m] = paired[m]-paired['raw_'+m]
save(paired, OUT/'gate_paired_vs_raw.csv.gz')

# Marker discrimination preservation, evaluated within sample only.
aucs = read_all('within_unit_auc_preservation.csv')
aucs = aucs[aucs.primary_eligible].copy()
assert_matched(aucs, ['benchmark', 'marker', 'dataset', 'unit_key'])
aucs['absolute_auc_change'] = aucs.auc_change.abs()
aucs['auc_loss'] = -aucs.auc_change.clip(upper=0)
aucs['auc_gain'] = aucs.auc_change.clip(lower=0)
aucs['loss_over_001'] = aucs.auc_change < -.01
aucs['gain_over_001'] = aucs.auc_change > .01
acols = ['auc', 'raw_auc', 'auc_change', 'absolute_auc_change', 'auc_loss', 'auc_gain', 'loss_over_001', 'gain_over_001']
ap = mean_by(aucs, ['benchmark', 'marker', 'dataset', 'method'], acols)
at = mean_by(ap, ['benchmark', 'marker', 'method'], acols)
ac = mean_by(at, ['benchmark', 'method'], acols)
ao = mean_by(ac, ['method'], acols)
save(ac, OUT/'auc_preservation_case_means.csv')
save(ao, OUT/'auc_preservation_overall.csv')
save(aucs.groupby('method').agg(n_units_tasks=('auc_change', 'size'),
    largest_absolute_auc_change=('absolute_auc_change', 'max'), largest_auc_loss=('auc_change', 'min'),
    n_losses_over_001=('loss_over_001', 'sum'), n_gains_over_001=('gain_over_001', 'sum')).reset_index(), OUT/'auc_preservation_counts.csv')

# Distribution alignment: all valid, matched method/marker comparisons, with the
# predeclared hierarchy of unit pairs -> platform pairs -> types -> markers -> cases.
scales = read_all('marker_scales.csv')
valid = scales.groupby(['benchmark', 'marker']).valid_scale.all().rename('all_methods_valid').reset_index()
save(scales, OUT/'marker_scales_all.csv')
save(valid[~valid.all_methods_valid], OUT/'degenerate_markers_excluded_from_all_methods.csv')
emd = read_all('emd_unit_pairs.csv.gz').merge(valid, on=['benchmark', 'marker'])
emd = emd[emd.all_methods_valid].copy()
ekey = ['benchmark', 'marker', 'celltype', 'platform_left', 'platform_right', 'unit_left', 'unit_right']
assert_matched(emd, ekey)
assert emd.emd_normalized.notna().all() and (emd.emd_normalized >= 0).all()
ppkeys = ['benchmark', 'method', 'marker', 'celltype', 'platform_left', 'platform_right']
ecols = ['emd_normalized', 'emd_native']
ep = mean_by(emd, ppkeys, ecols)
ep = ep.merge(emd.groupby(ppkeys).size().rename('n_unit_pairs').reset_index(), on=ppkeys)
et = mean_by(ep, ['benchmark', 'method', 'marker', 'celltype'], ecols)
em = mean_by(et, ['benchmark', 'method', 'marker'], ecols)
ec = mean_by(em, ['benchmark', 'method'], ecols)
eo = mean_by(ec, ['method'], ['emd_normalized'])
for f, name in [(ep,'emd_platform_pair_means.csv'),(et,'emd_celltype_means.csv'),
                (em,'emd_marker_means.csv'),(ec,'emd_case_means.csv'),(eo,'emd_overall_equal_case.csv')]:
    save(f, OUT/name)

# Within-type dispersion diagnostic; no claim of ground-truth variability.
disp = read_all('within_type_dispersion.csv.gz').merge(valid, on=['benchmark', 'marker'])
disp = disp[disp.all_methods_valid].copy()
dp = mean_by(disp, ['benchmark', 'method', 'marker', 'celltype', 'dataset'], ['mad_normalized'])
dt = mean_by(dp, ['benchmark', 'method', 'marker', 'celltype'], ['mad_normalized'])
dm = mean_by(dt, ['benchmark', 'method', 'marker'], ['mad_normalized'])
dc = mean_by(dm, ['benchmark', 'method'], ['mad_normalized'])
draw = dc[dc.method.eq('Raw')][['benchmark', 'mad_normalized']].rename(columns={'mad_normalized':'raw_mad_normalized'})
dc = dc.merge(draw, on='benchmark')
dc['relative_to_raw'] = dc.mad_normalized/dc.raw_mad_normalized
save(dc, OUT/'dispersion_case_diagnostic.csv')

summary = overall_gates.merge(eo, on='method').merge(ao[['method', 'auc_change', 'auc_loss', 'absolute_auc_change']], on='method')
for col in ['balanced_accuracy','f1']:
    summary[col+'_rank'] = summary[col].rank(ascending=False, method='min')
summary['emd_rank'] = summary.emd_normalized.rank(method='min')
summary = summary.set_index('method').loc[METHODS].reset_index()
save(summary, OUT/'RESULTS_OVERVIEW.csv')
print(summary.to_string(index=False), flush=True)
print('\nPer-case fixed-gate balanced accuracy:\n'+case_gates.pivot(index='benchmark', columns='method', values='balanced_accuracy')[METHODS].to_string(), flush=True)
print('\nPer-case normalized EMD:\n'+ec.pivot(index='benchmark', columns='method', values='emd_normalized')[METHODS].to_string(), flush=True)
print('\nGate coverage:\n'+coverage.to_string(index=False), flush=True)

# Review figures display every method/case, not just favorable tasks.
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'axes.titlesize':11,
    'axes.titleweight':'bold', 'svg.fonttype':'none', 'pdf.fonttype':42})

def heatmap(ax, values, title, cmap, vmin=None, vmax=None, fmt='.3f'):
    arr = values.to_numpy()
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(values.columns)), values.columns, rotation=25, ha='right', fontsize=9)
    ax.set_yticks(range(len(values.index)), [LABELS.get(i, i) for i in values.index], fontsize=9)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            color = im.cmap(im.norm(arr[i,j]))
            lum = .2126*color[0]+.7152*color[1]+.0722*color[2]
            ax.text(j, i, format(arr[i,j], fmt), ha='center', va='center', fontsize=10,
                color='white' if lum < .48 else '#15232b', fontweight='bold' if j == 4 else 'normal')
    ax.set_title(title, loc='left', pad=9)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5, len(values.columns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(values.index), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(Rectangle((3.5,-.5),1,len(values.index),fill=False,edgecolor='#173e36',linewidth=1.2,clip_on=False))
    return im


def matrix_with_mean(frame, column):
    v = frame.pivot(index='benchmark', columns='method', values=column).reindex(index=CASES, columns=METHODS)
    v.loc['Equal-case mean'] = v.mean()
    return v


fig, axes = plt.subplots(2,2,figsize=(12.2,7.2),layout='constrained')
heatmap(axes[0,0],matrix_with_mean(case_gates,'balanced_accuracy'),'A  Fixed-gate balanced accuracy  ↑','YlGnBu', .4, 1)
heatmap(axes[0,1],matrix_with_mean(case_gates,'f1'),'B  Fixed-gate F1  ↑','YlGnBu', 0, 1)
heatmap(axes[1,0],matrix_with_mean(ec,'emd_normalized'),'C  Same-cell-type normalized EMD  ↓','YlGnBu_r')
heatmap(axes[1,1],matrix_with_mean(ac,'auc_change'),'D  Within-sample AUROC change vs Raw','RdBu', -.20,.20,fmt='+.3f')
fig.suptitle('Saved marker-value evaluation · all four benchmarks',fontsize=15,fontweight='bold')
fig.savefig(OUT/'value_evaluation_overview.png',dpi=220)
fig.savefig(OUT/'value_evaluation_overview.svg')
plt.close(fig)

task_frames = []
for c in CASES:
    for m in ['CD3','CD8','CD4']:
        t = tasks[(tasks.benchmark == c) & (tasks.marker == m)].copy()
        if len(t):
            t['task'] = LABELS[c]+' · '+m
            task_frames.append(t)
tf = pd.concat(task_frames)
order = tf.task.drop_duplicates().tolist()
fig, axes = plt.subplots(1,2,figsize=(12.2,7.1),layout='constrained')
for ax, col, title in zip(axes,['balanced_accuracy','f1'],['Fixed-gate balanced accuracy  ↑','Fixed-gate F1  ↑']):
    mat = tf.pivot(index='task',columns='method',values=col).reindex(index=order,columns=METHODS)
    heatmap(ax,mat,title,'YlGnBu',0,1)
fig.suptitle('All predeclared marker tasks · six source → target directions per task',fontsize=13,fontweight='bold')
fig.savefig(OUT/'gate_all_marker_tasks.png',dpi=220)
fig.savefig(OUT/'gate_all_marker_tasks.svg')
plt.close(fig)

dump(OUT/'VERIFICATION.json',dict(status='PASS', n_benchmarks=len(complete),
    n_cells=sum(x['n_cells'] for x in complete), n_units=sum(x['n_units'] for x in complete),
    n_expression_methods=len(METHODS), primary_gate_unit_direction_tasks_per_method=int(len(primary)/len(METHODS)),
    primary_unique_unit_tasks_per_method=int(len(aucs)/len(METHODS)),
    source_task_gates_per_method=int(len(sources)/len(METHODS)),
    failed_source_task_gates_per_method=int((sources.status != 'OK').sum()/len(METHODS)),
    emd_unit_type_marker_pairs_per_method=int(len(emd)/len(METHODS)),
    degenerate_marker_cases=int((~valid.all_methods_valid).sum()),
    gate_task_direction_counts=tasks[tasks.method.eq('Raw')][['benchmark','marker','n_directions']].to_dict('records'),
    source_hashes_rechecked_unchanged=True, matched_method_rows=True,
    protocol_sha256=sha(HERE/'PROTOCOL.md'),
    scripts_sha256={p.name:sha(p) for p in [HERE/'evaluate_values.py',HERE/'summarize_values.py']}))
