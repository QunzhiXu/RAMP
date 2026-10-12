"""Verify frozen inputs, manifests and test evidence, then seal formal shards."""
import sys
sys.dont_write_bytecode=True
import json
from collections import Counter
from pathlib import Path
from trainability.core import ROOT,RELEASE_CHECKS,require,file_sha256,digest,write_json
from trainability.real_data import load_task


def main():
    folder=ROOT/'config/formal_v0_5';suite=json.loads((folder/'SUITE.json').read_text())
    require(suite['status']=='awaiting_release_checks','Suite already sealed or unexpected state')
    profile=json.loads(Path(suite['profile']).read_text())
    test_path=ROOT/'validation/checks_v0_5/REPORT.json';tests=json.loads(test_path.read_text())
    require(tests['status']=='passed' and tests['failures']==tests['errors']==0,'Tests failed')
    for name,h in tests['sha256'].items():require(file_sha256(ROOT/name)==h,'Tested file changed: '+name)
    chain_path=ROOT/'validation/formal_chain_pilot_v0_5/summary_final/REPORT.json';chain=json.loads(chain_path.read_text())
    require(chain['status']=='complete' and chain['passed']==8,'Real J1-J4 chain incomplete')
    source_path=ROOT/'validation/formal_preflight_v0_5/SOURCE_CHECK.json'
    source=json.loads(source_path.read_text());require(source['status']=='passed' and not source['changed'],'Original source changed')
    inputs=json.loads((ROOT/'validation/formal_preflight_v0_5/INPUTS.json').read_text())
    for name,h in inputs['original_source_hashes'].items():require(file_sha256(name)==h,'Original source changed since preflight')
    task_map={r['task']:load_task(r['task']) for r in inputs['pools']}
    for r in inputs['pools']:
        require(task_map[r['task']].pool.sha256==r['pool_sha256'],'Registered pool changed')
        require(file_sha256(r['manifest'])==r['manifest_sha256'],'Input manifest changed')
    code_hashes={str(p.relative_to(ROOT)):file_sha256(p) for p in (ROOT/'src').rglob('*') if p.suffix in {'.py','.R'}}
    code_sha=digest(code_hashes);job_counts=Counter();paired={}
    for item in suite['shards']:
        require(file_sha256(item['config'])==item['sha256'],'Frozen shard changed')
        cfg=json.loads(Path(item['config']).read_text())
        require(cfg['mode']=='formal' and cfg['status']=='frozen' and cfg['formal_run_enabled'],'Invalid formal state')
        require(cfg['profile_sha256']==suite['profile_sha256'],'Shard/profile mismatch')
        require(len({j['id'] for j in cfg['jobs']})==len(cfg['jobs'])==item['jobs'],'Duplicate/missing jobs')
        for job in cfg['jobs']:
            task=task_map[job['task']];job_counts[job['task_type']]+=1
            src=job.get('sources',[job.get('source')]);require(job['target'] not in src,'Source/target overlap')
            require(job['seed'] in profile['seeds'] and len(job['supervision_sha256'])==64,'Unfrozen supervision')
            if job['task_type']=='J2':require(all(c['j2_eligible'] for c in task.contracts.values()) and job['marker'] in task.markers,'Ineligible masked-marker job')
            if job['task_type']=='J4':require(job['method'] not in ['fastmnn','maxfuse'] and job['pretrain']==profile['j4'],'Invalid J4 method/budget')
            key=digest({k:v for k,v in job.items() if k not in ['id','method']})
            paired.setdefault((job['task_type'],key),set()).add(job['method'])
    for (kind,key),methods in paired.items():
        expected=set(profile['methods'])-({'fastmnn','maxfuse'} if kind=='J4' else set())
        require(methods==expected,'Methods do not share exact task/supervision/candidate support')
    preflight=json.loads((ROOT/'validation/formal_preflight_v0_5/REPORT.json').read_text())
    require(dict(job_counts)==preflight['job_counts'],'Job counts changed')
    # Training-critical sources used by the real-chain pilot must match release.
    for method in ['unaligned','ramp']:
        run=json.loads((ROOT/'validation/formal_chain_pilot_v0_5'/method/'RUN.json').read_text())
        for name in ['runner','j4','heads','scoring','inputs','adapters','real_data','sampling','validation']:
            key='src/trainability/'+name+'.py'
            require(run['code_hashes'][key]==code_hashes[key],'Training code changed since real-chain validation: '+key)
    checks=dict(status='passed',version='0.5.0',tests=tests['tests'],real_chain_jobs=8,
        job_counts=dict(job_counts),total_jobs=sum(job_counts.values()),shards=len(suite['shards']),
        exact_method_pairing=True,source_files_checked=source['original_sources_checked'],
        original_sources_changed=0,code_sha256=code_sha,code_hashes=code_hashes,
        exclusions='validation/formal_preflight_v0_5/EXCLUSIONS.json',
        source_budget='validation/formal_budget_v0_5/REPORT.json',
        test_evidence_sha256=file_sha256(test_path),chain_evidence_sha256=file_sha256(chain_path))
    release_report=ROOT/'validation/formal_preflight_v0_5/RELEASE_REPORT.json';write_json(release_report,checks)
    paths={k:release_report for k in RELEASE_CHECKS}
    for k in ['mask_isolation','source_only_selection','aggregation_checks']:paths[k]=test_path
    paths['canonical_melanoma']=ROOT/'validation/canonical_melanoma_v0_3/REPORT.json'
    paths['runtime_pilot']=chain_path
    for item in suite['shards']:
        cfg=json.loads(Path(item['config']).read_text())
        evidence={k:dict(status='passed',evidence_path=str(p),sha256=file_sha256(p)) for k,p in paths.items()}
        evidence['configuration_frozen'].update(configuration_sha256=item['sha256'],code_sha256=code_sha)
        write_json(cfg['release_evidence'],evidence)
    suite.update(status='released',release_report=str(release_report),release_report_sha256=file_sha256(release_report),code_sha256=code_sha)
    write_json(folder/'SUITE.json',suite,replace=True)
    print('RELEASED',sum(job_counts.values()),'jobs across',len(suite['shards']),'shards')


if __name__=='__main__':main()
