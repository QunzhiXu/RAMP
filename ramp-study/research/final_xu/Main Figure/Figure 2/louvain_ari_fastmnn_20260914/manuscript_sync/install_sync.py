"""Back up and install the already compiled, reviewed Figure 2 manuscript update."""
from pathlib import Path
import datetime, hashlib, json, shutil

HERE = Path(__file__).resolve().parent
DEST = Path('/Users/qxu6/Documents/New project/manuscript')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    original = json.loads((HERE/'original_manuscript_hashes.json').read_text())
    assert all(sha(p) == h for p, h in original.items()), 'Source manuscript changed during staging'
    qa = json.loads((HERE/'QA.json').read_text())
    for stem in ['manuscript', 'manuscript_figure1_polished']:
        assert qa[stem]['figure2_page'] == 4
        assert qa[stem]['caption_on_same_page']
        assert qa[stem]['no_overfull_or_undefined_reference']
    pairs = []
    for stem in ['manuscript', 'manuscript_figure1_polished']:
        for ext in ['tex', 'pdf']:
            pairs.append((HERE/'staged'/f'{stem}.{ext}', DEST/f'{stem}.{ext}'))
    for ext in ['png', 'svg', 'pdf']:
        pairs.append((HERE/'staged/figure'/f'figure2_benchmark.{ext}', DEST/'figure'/f'figure2_benchmark.{ext}'))
    pairs += [(HERE/'manuscript_figure1_polished_figure2_page4.png', DEST/'figure2_latest_page.png'),
              (HERE/'FIGURE2_UPDATE.md', DEST/'FIGURE2_UPDATE.md')]
    assert all(source.is_file() for source, _ in pairs)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = DEST/'backups'/f'figure2_before_{timestamp}'
    assert not backup.exists()
    backup.mkdir(parents=True)
    records = []
    # Complete every backup before replacing any existing manuscript file.
    for source, target in pairs:
        old_hash = sha(target) if target.exists() else None
        if target.exists():
            saved = backup/target.relative_to(DEST)
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
            assert sha(saved) == old_hash
        records.append({'source': str(source), 'file': str(target),
                        'old_sha256': old_hash, 'new_sha256': sha(source)})
    manifest = {'status': 'backed_up', 'timestamp': timestamp, 'backup': str(backup),
                'files': records, 'figure_page': 4, 'caption_page': 4, 'pages': 18,
                'qa': qa, 'source_figure_version': str(HERE.parent)}
    (backup/'SYNC_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for source, target in pairs:
        shutil.copy2(source, target)
        assert sha(source) == sha(target)
    targets = {str(target) for _, target in pairs}
    untouched = {p: h for p, h in original.items() if p not in targets}
    assert all(sha(p) == h for p, h in untouched.items())
    manifest.update(status='PASS', untouched_original_files=len(untouched))
    (backup/'SYNC_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (HERE/'INSTALL_VERIFY.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'status':'PASS', 'backup':str(backup), 'installed':len(pairs),
                      'untouched_original_files':len(untouched), 'figure2_page':4}, indent=2))


if __name__ == '__main__':
    main()
