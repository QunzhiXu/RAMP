"""Install the fully checked update into bioRxiv_Ver1, preserving exact backups."""
from pathlib import Path
import hashlib
import json
import shutil

B = Path(__file__).resolve().parent
ROOT = B.parents[2]
LIVE = ROOT.parent / 'bioRxiv_Ver1'
STAGE = B / 'manuscript_stage'
BACKUP = LIVE / '_backups/figure2_lung_reference_20260921'
FILES = [
    'template.tex', 'template.pdf', 'README.md', 'FIGURE2_CAPTION.md',
    'figure/figure2_benchmark.pdf', 'supplementary_data/README.md',
    'supplementary_data/all31_metric_scores_and_ranks.csv',
    'supplementary_data/comparison_preprocessing.json',
    'supplementary_data/rank_sensitivity.csv',
    *[f'supplementary_data/{s}_{w}_by_task.csv'
      for s in ['Main23', 'Full31'] for w in ['category_equal', 'metric_equal']],
]


def sha(p):
    with p.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    verified = json.loads((B / 'RELEASE_VERIFICATION.json').read_text())
    assert verified['status'] == 'PASS'
    assert verified.get('visual_review_passed') is True
    assert not BACKUP.exists(), 'Backup exists; inspect prior installation before retrying.'
    original = json.loads((B / 'LIVE_BEFORE.json').read_text())
    for relative, digest in original.items():
        assert sha(LIVE / relative) == digest, f'Live file changed during staging: {relative}'
    for relative in FILES:
        assert (STAGE / relative).is_file(), relative
    for relative, digest in verified['output_sha256'].items():
        assert sha(B / relative) == digest, f'QA output changed: {relative}'
    backup_hashes = {}
    for relative in FILES:
        live = LIVE / relative
        if live.exists():
            dest = BACKUP / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(live, dest)
            backup_hashes[relative] = sha(dest)
    (BACKUP / 'MANIFEST.json').write_text(json.dumps(backup_hashes, indent=2) + '\n')
    output_hashes = {}
    for relative in FILES:
        target = LIVE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(STAGE / relative, target)
        assert sha(target) == sha(STAGE / relative)
        output_hashes[relative] = sha(target)
    assert all(sha(LIVE / p) == h for p, h in original.items() if p not in FILES)
    result = {'status': 'PASS', 'target': str(LIVE), 'backup': str(BACKUP),
              'installed_sha256': output_hashes, 'all_other_live_files_unchanged': True}
    (B / 'INSTALL_VERIFY.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
