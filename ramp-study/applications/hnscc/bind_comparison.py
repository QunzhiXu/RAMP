"""Bind historical all-cell identities for a CD8-only reference-array comparison."""
from pathlib import Path
from common import HERE, read, write, sha, record
from prepare import verify

if __name__=='__main__':
    lock=verify(HERE/'frozen_20261011'); source=Path(lock['source_root'])
    rels=['story_test_hnscc/final_return_exploration_20260910/target_all_metadata.csv.gz',
          'story_test_hnscc/final_return_exploration_20260910/01_prepare.py']
    out=HERE/'comparison_references_20261011'; out.mkdir()
    value={'base_lock_sha256':sha(HERE/'frozen_20261011/LOCK.json'),
           'role':'comparison-only row identities and historical producer; never mirrored into execution',
           'files':{r:record(source/r) for r in rels}}
    write(out/'LOCK.json',value)
    write(out/'COMPLETE.json',{'status':'complete','lock_sha256':sha(out/'LOCK.json')})
    print(sha(out/'LOCK.json'))
