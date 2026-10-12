"""Document differences without changing tolerances, completed runs or old reports."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from common import HERE, STORY, EXP, read, write, record, sha, reserve
from prepare import verify as verify_lock


def main():
    lock = verify_lock(HERE / 'frozen_20261011', source=True)
    source = Path(lock['source_root'])
    run = HERE / 'runs/rebuild_20261011_r2'
    tree = run / 'tree'
    compared = HERE / 'validation/comparison_20261011/report.json'
    report = read(compared)
    expected = {f'{EXP}/CD4_NN_UMAP_validation/PLOT_AUDIT.json',
                f'{STORY}/alignment/output/manifest.json',
                f'{STORY}/figure/panel_B/data/all_marker_qc_scores.csv',
                f'{STORY}/figure/panel_B/data/codex_top3_marker_donor_scores.csv'}
    if set(report['differing_files']) != expected:
        raise ValueError('Unrecognized difference set')
    qc = {}
    for name in ('all_marker_qc_scores.csv', 'codex_top3_marker_donor_scores.csv'):
        rel = f'{STORY}/figure/panel_B/data/{name}'
        new = pd.read_csv(tree / rel, float_precision='round_trip')
        old = pd.read_csv(source / rel, float_precision='round_trip')
        different = [c for c in new if not new[c].equals(old[c])]
        if set(different) != {'raw_shape_anomaly', 'shape_anomaly'}:
            raise ValueError('QC difference scope changed')
        diff = np.abs(new[different].to_numpy() - old[different].to_numpy())
        if not np.isfinite(diff).all():
            raise ValueError('Nonfinite QC difference')
        qc[name] = {'different_columns': different, 'max_abs_difference': float(diff.max()),
                    'changed_rows': int((diff > 0).any(axis=1).sum()),
                    'percentiles_alarm_ranks_all_other_columns_exact': True,
                    'diagnosis': 'Rounding-scale intermediate difference; historical operation-level cause not established'}
    rel = f'{STORY}/sampled_data/relabel/relabeled_metadata.csv.gz'
    old_manifest = read(source / STORY / 'alignment/output/manifest.json')
    historical_hash = old_manifest['source_hashes'][str(source / rel)]
    current_hash = sha(source / rel)
    producer = read(source / STORY / 'sampled_data/relabel/manifest.json')
    if producer['relabeled_metadata_output']['sha256'] != current_hash:
        raise ValueError('Current relabel producer hash does not match')
    frozen_md = pd.read_csv(source / STORY / 'alignment/output/cell_metadata.csv.gz')
    current_md = pd.read_csv(source / rel)
    keys = ['dataset', 'cell_id']
    ordered = current_md.set_index(keys).loc[pd.MultiIndex.from_frame(frozen_md[keys])].reset_index()
    if not ordered[frozen_md.columns].astype(str).equals(frozen_md.astype(str)):
        raise ValueError('Consumed relabel fields differ from original alignment metadata')
    if sha(tree / STORY / 'alignment/output/cell_metadata.csv.gz') != sha(source / STORY / 'alignment/output/cell_metadata.csv.gz'):
        raise ValueError('Saved metadata is not byte-identical')
    old_plot = read(source / EXP / 'CD4_NN_UMAP_validation/PLOT_AUDIT.json')
    new_plot = read(tree / EXP / 'CD4_NN_UMAP_validation/PLOT_AUDIT.json')
    if new_plot.pop('embedding_label') != 'CODEX-only UMAP' or new_plot != old_plot:
        raise ValueError('Unexpected CD4 plotting audit difference')
    result = {'status': 'differences_documented_not_erased', 'comparison_report': record(compared),
              'qc': qc, 'CD4_plot_record': 'Only embedding_label=CODEX-only UMAP added by the currently frozen historical script',
              'relabel': {'historical_alignment_manifest_sha256_field': historical_hash,
                          'current_input_sha256': current_hash, 'current_producer_manifest_matches': True,
                          'consumed_columns': list(frozen_md.columns), 'all_88031_rows_match_original_saved_metadata': True,
                          'new_saved_metadata_bytes_identical': True,
                          'old_full_file_with_historical_hash_recovered': False,
                          'reason_for_old_hash_disagreement': 'Not established; do not assume only gzip metadata changed'},
              'scripts_or_scientific_parameters_changed': False, 'old_reports_rewritten': False,
              'code': record(Path(__file__))}
    out = reserve(HERE / 'validation/differences_20261011', [source, run])
    write(out / 'report.json', result)
    write(out / 'COMPLETE.json', {'status': 'complete', 'report_sha256': sha(out / 'report.json')})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
