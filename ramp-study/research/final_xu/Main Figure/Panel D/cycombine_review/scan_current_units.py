"""Check canonical marker contrasts in every current frozen sample unit."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from figure_style import ROOT, CASES, track, READS, sha
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata, mannwhitneyu

rules = pd.read_csv(track(ROOT/'benchmark_results/exact_value_artifacts/config/canonical_expectations.csv'))
rows, inventory, skips = [], [], []

for case in CASES:
    base = ROOT/'benchmark_results'/case
    meta = pd.read_csv(track(base/'ramp/alignment/input/cell_metadata.csv.gz'), low_memory=False)
    markers = track(base/'ramp/alignment/input/markers.txt').read_text().splitlines()
    raw = np.load(track(base/'ramp/alignment/input/raw.npy'), mmap_mode='r')
    corrected = pd.read_csv(track(base/'cycombine/alignment/data/corrected_expression.csv.gz'), usecols=['cell_id', *markers])
    assert corrected.cell_id.is_unique and meta.cell_id.is_unique
    assert set(corrected.cell_id) == set(meta.cell_id)
    output = corrected.set_index('cell_id').loc[meta.cell_id, markers].to_numpy()
    assert raw.shape == output.shape and np.isfinite(output).all()
    for unit, idx in meta.groupby('unit_key', sort=False).indices.items():
        local = meta.iloc[idx]
        dataset = local.dataset.iloc[0]
        assert local.dataset.nunique() == 1
        inventory.append({'case': case, 'dataset': dataset, 'unit_key': unit, 'n_cells': len(idx)})
        rank_cache = {}
        for rule in rules[rules.cancer.eq(case)].itertuples():
            if rule.dataset_allowlist != '*' and dataset not in rule.dataset_allowlist.split(';'): continue
            if rule.marker not in markers: continue
            j = markers.index(rule.marker)
            if j not in rank_cache:
                rank_cache[j] = [rankdata(v[idx,j], method='average')/(len(idx)+1) for v in [raw, output]]
            before, after = rank_cache[j]
            pos = local.celltype.isin(rule.positive_labels.split(';')).to_numpy()
            for neg_label in rule.negative_labels.split(';'):
                neg = local.celltype.eq(neg_label).to_numpy()
                key = {'case': case, 'dataset': dataset, 'unit_key': unit, 'marker': rule.marker,
                       'rule_id': rule.rule_id, 'target': rule.positive_labels, 'non_target': neg_label,
                       'n_target': int(pos.sum()), 'n_non_target': int(neg.sum())}
                if pos.sum() < 50 or neg.sum() < 50:
                    skips.append(key)
                    continue
                row = dict(key)
                for name, values in [('raw', before), ('corrected', after)]:
                    row[name+'_auc'] = float(mannwhitneyu(values[pos], values[neg]).statistic/(pos.sum()*neg.sum()))
                    for label, mask in [('target', pos), ('non_target', neg)]:
                        a = values[mask]
                        row[name+'_'+label+'_mean_rank'] = float(a.mean())
                        row[name+'_'+label+'_median_rank'] = float(np.median(a))
                row['auc_loss'] = row['raw_auc']-row['corrected_auc']
                a,b,c,d = [row[k] for k in ['raw_target_mean_rank','raw_non_target_mean_rank','corrected_target_mean_rank','corrected_non_target_mean_rank']]
                row['non_target_rank_shift'] = d-b
                row['target_rank_shift'] = c-a
                if row['raw_auc'] < .7: classification = 'raw specificity insufficient'
                elif a-b >= .1 and c-d <= -.05 and (a-b)-(c-d) >= .2: classification = 'mean-order reversal'
                elif b <= .4 and d >= .6 and d-b >= .2: classification = 'clear non-target low-to-high'
                elif a >= .6 and c <= .4 and c-a <= -.2: classification = 'clear target high-to-low'
                elif b <= .45 and d >= .55 and d-b >= .15: classification = 'strong non-target intrusion'
                else: classification = 'below strong-event thresholds'
                row['classification'] = classification
                row['strong_event'] = classification not in ['raw specificity insufficient','below strong-event thresholds']
                rows.append(row)
    print(case, 'complete', flush=True)

data = pd.DataFrame(rows)
data.to_csv(HERE/'all_unit_canonical_comparisons.csv.gz', index=False)
hits = data[data.strong_event].sort_values('auc_loss', ascending=False)
hits.to_csv(HERE/'strong_unit_candidates.csv', index=False)
data[data.raw_auc.ge(.8)].sort_values('auc_loss',ascending=False).to_csv(HERE/'ranked_specificity_losses.csv', index=False)
pd.DataFrame(inventory).to_csv(HERE/'unit_coverage.csv', index=False)
pd.DataFrame(skips).to_csv(HERE/'low_count_skips.csv.gz', index=False)
pooled = pd.read_csv(track(ROOT/'benchmark_results/exact_value_artifacts/summary/canonical_all_comparisons.csv.gz'))
pooled = pooled[pooled.method.eq('cyCombine')].copy()
pooled['auc_loss'] = pooled.raw_target_vs_nontarget_auc-pooled.corrected_target_vs_nontarget_auc
pooled.to_csv(HERE/'pooled_canonical_comparisons.csv', index=False)
assert all(sha(p) == digest for p, digest in READS.items())
(HERE/'manifest.json').write_text(json.dumps({'input_sha256': READS, 'inputs_unchanged': True,
    'n_units': len(inventory), 'eligible_comparisons': len(data), 'strong_unit_candidates': len(hits),
    'scope': 'All sample units of the current four frozen benchmarks; original canonical lineage-marker rules; both contrasted populations >=50 cells.',
    'display_scale': 'Average within-unit rank divided by n+1, ranking all cells in each sample.',
    'classification': 'Same four strong-event definitions as the existing pooled canonical audit, now evaluated within each sample.',
    'caution': 'Sample-level exploratory candidates; distinct from the earlier pooled dataset hit counts. No new fitting or changes to marker outputs.'},indent=2)+'\n')
print('Units:', len(inventory), 'contrasts:', len(data), 'strong candidates:', len(hits))
print(hits[['case','dataset','unit_key','marker','target','non_target','n_target','n_non_target','raw_auc','corrected_auc','classification']].head(18).to_string(index=False))
