"""Copy the verified standalone Overleaf bundle beside the active manuscript."""
from pathlib import Path
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
LIVE = HERE.parent.parent.parent / 'bioRxiv_Ver1'
checks = json.loads((HERE / 'EXPORT_CHECKS.json').read_text())
source = HERE / 'Supplementary_Overleaf'
archive = HERE / 'Supplementary_Overleaf.zip'
destination = LIVE / source.name
zip_destination = LIVE / archive.name

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

assert checks['status'] == 'ready_to_install'
assert sha(archive) == checks['zip_sha256']
assert not destination.exists() and not zip_destination.exists(), 'Do not overwrite an existing export.'
originals = json.loads((HERE / 'SOURCE_COPY_MANIFEST.json').read_text())['files']
for item in originals:
    assert sha(item['source']) == item['sha256']
shutil.copytree(source, destination)
shutil.copy2(archive, zip_destination)
for p in source.rglob('*'):
    if p.is_file():
        assert sha(p) == sha(destination / p.relative_to(source))
assert sha(zip_destination) == checks['zip_sha256']
for item in originals:
    assert sha(item['source']) == item['sha256']
receipt = dict(status='installed_and_verified', directory=str(destination),
               archive=str(zip_destination), archive_bytes=zip_destination.stat().st_size,
               files=checks['files'], zip_sha256=checks['zip_sha256'],
               original_manuscript_files_unchanged=True)
(HERE / 'INSTALL_VERIFY.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
