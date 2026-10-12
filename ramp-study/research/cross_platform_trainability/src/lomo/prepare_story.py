"""Stage one leave-one-marker-out run for an application (story) task.

The three applications were not run through the benchmark's per-method
runners; they use their own driver, Comparison/scripts/run_r.R, which reads a
marker file and a matrix whose columns must be exactly the three identity
columns followed by those markers.  So the held-out marker is removed from
both.  The driver locates the R libraries as <root>/algorithm/<name>, which is
why the staged copy sits two levels below a directory holding the mirrored
algorithm tree.
"""
import os, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import resolve

PROJECT = Path(__file__).resolve().parents[2]
WORK = PROJECT / 'lomo_work'
FINAL_XU = resolve.SRC
STORY = resolve.STORY
ENV = dict(os.environ, LC_COLLATE='en_US.UTF-8', LANG='en_US.UTF-8',
           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           VECLIB_MAXIMUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
IDENTITY = ['cell_id', 'dataset', 'unit_key']


def stage(task, drop, workdir):
    src = FINAL_XU / STORY[task] / 'Comparison'
    w = Path(workdir); shutil.rmtree(w, ignore_errors=True)
    here = w / 'stage' / 'Comparison'
    (here / 'input').mkdir(parents=True)
    (here / 'methods').mkdir(); (here / 'logs').mkdir(); (here / 'tmp').mkdir()
    link = w / 'algorithm'
    link.symlink_to(WORK / 'algorithm', target_is_directory=True)
    shutil.copy(src / 'run_config.json', here / 'run_config.json')
    shutil.copytree(src / 'scripts', here / 'scripts')

    markers = [m for m in (src / 'input' / 'markers.txt').read_text().split() if m]
    if drop not in markers:
        raise ValueError(f'{drop} not among {task} markers')
    keep = [m for m in markers if m != drop]
    (here / 'input' / 'markers.txt').write_text('\n'.join(keep) + '\n')
    for name in ('analysis_ready.csv.gz', 'raw.csv.gz'):
        frame = pd.read_csv(src / 'input' / name, low_memory=False)
        frame[IDENTITY + keep].to_csv(here / 'input' / name, index=False)
    shutil.copy(src / 'input' / 'cell_ids.npy', here / 'input' / 'cell_ids.npy')
    return here, keep


def run(task, method, drop, workdir, timeout=7200):
    here, keep = stage(task, drop, workdir)
    cmd = ['Rscript', str(here / 'scripts' / 'run_r.R'), str(here), method]
    r = subprocess.run(cmd, capture_output=True, text=True, env=ENV,
                       cwd=str(here), timeout=timeout)
    return dict(out=here / 'methods' / method, markers=keep), r
