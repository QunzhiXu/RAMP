"""Archive guarded controllers before a new sequential run; no resume stitching."""
from pathlib import Path
import shutil
from common import HERE, codes, write, sha
from baseline import capture
from study import fit

if __name__=='__main__':
    prior=HERE/'validation/preflight_20261011'; capture(prior)
    folder=HERE/'validation/execution_20261011'; folder.mkdir()
    (folder/'controller').mkdir()
    for name in codes(): shutil.copyfile(HERE/name,folder/'controller'/name)
    write(folder/'REQUEST.json',{'code_records':codes(),'prior_sha256':sha(prior/'PRIOR.json')})
    try:
        result=fit(HERE/'frozen_20261011',HERE/'runs/rebuild_20261011')
        write(folder/'report.json',result)
        write(folder/'COMPLETE.json',{'report_sha256':sha(folder/'report.json')})
    except BaseException as e:
        write(folder/'FAILED.json',{'error':repr(e)}); raise
