"""Read-only source accounting, without fitting or accessing research data."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from release_check import relative, unique_object, verify


def check(root=ROOT):
    root=Path(root);integrity=verify(root)
    c=json.loads((root/'ramp-study/SOURCE_COVERAGE.json').read_text(),object_pairs_hook=unique_object)
    if c.get('publish_allowed') is not False:raise ValueError('Publication not approved')
    seen=set();code=0;configs=0;aliases=0
    for r in c['files']:
        source=relative(r['source']).as_posix();dest=relative(r['destination']).as_posix()
        if source in seen:raise ValueError('Duplicate coverage source')
        seen.add(source)
        data=(root/dest).read_bytes()
        if len(data)!=r['bytes'] or hashlib.sha256(data).hexdigest()!=r['sha256']:
            raise ValueError('Uncovered source: '+source)
        if r['kind']=='code':code+=1
        elif r['kind']=='configuration_or_environment':configs+=1
        else:raise ValueError('Unknown source role')
        if r['disposition']=='byte_identical_runtime_copy':aliases+=1
        if r['publication'] not in {'author_and_dependency_terms_pending','maxfuse_distribution_review'}:
            raise ValueError('Publication review bypass')
    counts={'indexed_source_records':len(seen),'code_records':code,'config_records':configs,
            'byte_identical_runtime_aliases':aliases}
    if any(c['summary'][k]!=v for k,v in counts.items()):raise ValueError('Coverage counts differ')
    for consumer,source in c['dependencies']:
        if consumer not in seen or source not in seen:raise ValueError('Uncovered discovered source dependency')
    routes=json.loads((root/'ramp-study/ENTRYPOINTS.json').read_text())
    expected={f'Figure {i}' for i in range(1,8)}|{f'S{i}' for i in range(1,14)}
    if {r['figure'] for r in routes['figures']}!=expected or len(routes['figures'])!=20:
        raise ValueError('Figure route set differs')
    for r in routes['figures']+routes['analyses']:
        for name in r['sources']:
            relative(name)
            if name not in seen:raise ValueError('Entrypoint absent from coverage: '+name)
    return {'status':'source_and_configuration_coverage_verified','source_integrity':integrity,
            **counts,'figures':20,'analysis_routes':len(routes['analyses']),
            'scientific_execution':False,'publish_allowed':False}


if __name__=='__main__':print(json.dumps(check(),indent=2))
