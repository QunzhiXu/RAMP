"""Check pagination, unchanged pages, and render the revised Lung page for review."""
from pathlib import Path
import hashlib
import json
import re
import fitz

HERE = Path(__file__).resolve().parent/'manuscript_sync'
STAGE = HERE/'staging'
ROOT = Path('/Users/qxu6/Documents/New project/manuscript')


def main():
    qa = dict(status='PASS', page_count={}, unaffected_pages_identical={},
              compile_checks={}, TeX_and_caption_unchanged=True)
    previews = []
    tex_hashes = json.loads((HERE/'tex_hashes.json').read_text())
    for path, expected in tex_hashes.items():
        assert hashlib.sha256((STAGE/Path(path).name).read_bytes()).hexdigest() == expected
    for name in ['manuscript', 'manuscript_figure1_polished']:
        old = fitz.open(ROOT/f'{name}.pdf')
        new = fitz.open(STAGE/f'{name}.pdf')
        assert len(new) == len(old) == 18
        assert 'Measured HLA-A' in new[9].get_text()
        assert 'Figure 5' in new[9].get_text()
        qa['page_count'][name] = len(new)
        changed = []
        for i in range(len(new)):
            if i != 9:
                a = old[i].get_pixmap(dpi=72, alpha=False)
                b = new[i].get_pixmap(dpi=72, alpha=False)
                if a.samples != b.samples:
                    changed.append(i+1)
        assert not changed, changed
        qa['unaffected_pages_identical'][name] = True
        log = (STAGE/f'{name}.log').read_text(errors='replace')
        issues = re.findall(r'^.*(?:Overfull|undefined|Fatal error|^!).*$', log, re.MULTILINE)
        assert not issues, issues
        qa['compile_checks'][name] = issues
        pix = new[9].get_pixmap(dpi=150, alpha=False)
        pix.save(HERE/f'{name}_lung_page10.png')
        previews.append(pix.samples)
        assert old[9].get_text().split() == new[9].get_text().split()
    qa['Lung_pages_identical_between_manuscript_variants'] = previews[0] == previews[1]
    assert qa['Lung_pages_identical_between_manuscript_variants']
    (HERE/'COMPILE_QA.json').write_text(json.dumps(qa, indent=2)+'\n')
    print(json.dumps(qa, indent=2))


if __name__ == '__main__':
    main()
