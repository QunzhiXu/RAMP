"""Read-only numerical comparisons; never used by fitting or parameter selection."""
import argparse
from pathlib import Path
import numpy as np
import inventory as inv
import safe_io as io
import study


def compare_files(result, spec, source):
    result,source=Path(result),Path(source);rows={}
    for name,ref in spec['references'].items():
        old=source/ref['source'];new=result/name;io.check(old,ref)
        byte_equal=io.record(new)=={k:ref[k] for k in ('bytes','sha256')}
        row={'byte_identical':byte_equal,'reference':{k:ref[k] for k in ('bytes','sha256')},'current':io.record(new)}
        if name.endswith('.npy'):
            a=np.load(old,allow_pickle=False);b=np.load(new,allow_pickle=False)
            if a.shape!=b.shape or a.dtype!=b.dtype or not np.array_equal(a,b):raise ValueError('Historical matrix differs: '+name)
            row.update(shape=list(a.shape),dtype=str(a.dtype),maximum_absolute_difference=float(np.max(np.abs(a-b))))
        elif name.endswith('.npz'):
            with np.load(old,allow_pickle=False) as a,np.load(new,allow_pickle=False) as b:
                if a.files!=b.files:raise ValueError('Profile/map names or order differ')
                for n in a.files:
                    if a[n].dtype!=b[n].dtype or a[n].shape!=b[n].shape or not np.array_equal(a[n],b[n]):raise ValueError('Profile/map array differs: '+n)
                row['exact_arrays']=len(a.files)
        if not byte_equal:raise ValueError('Historical scientific output is not byte-identical: '+name)
        rows[name]=row
    return {'case':spec['case'],'files':rows,'byte_identical_files':sum(r['byte_identical'] for r in rows.values())}


def compare(run,bundle,source):
    s=study.verify_bundle(bundle);study.verify_run(run,bundle)
    return compare_files(Path(run)/'result',s,source)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True)
    a=p.parse_args();r=compare(a.run,a.bundle,a.source_root)
    print({'case':r['case'],'byte_identical_files':r['byte_identical_files']})


if __name__=='__main__':main()
