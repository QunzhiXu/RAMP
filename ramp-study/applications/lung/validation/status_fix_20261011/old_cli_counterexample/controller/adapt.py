"""Extract unchanged reference-input statements; run the original R analysis."""
import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from common import EXP, read, write, sha, stage_root
from prepare import R_EXEC, ALIGN_SHA

def load(path):
    spec=importlib.util.spec_from_file_location('lung_frozen_entry',path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def prefix(source):
    tree=ast.parse(source)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    boundary=[]
    for i,n in enumerate(main.body):
        if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='rt.status' and n.value.args and isinstance(n.value.args[0],ast.Constant) and n.value.args[0].value=='fitting canonical RAMP 2.3.0':
            boundary.append(i)
    if len(boundary)!=1: raise ValueError('Reference prefix boundary differs')
    main.body=main.body[:boundary[0]]
    # Verify the same inputs again after rebuilding; return the provenance map.
    main.body += ast.parse('rt.unchanged(h)\nreturn h').body
    return ast.fix_missing_locations(ast.Module(body=[main],type_ignores=[]))

def reference_inputs(run):
    import numpy as np
    base=Path(run)/'tree'/EXP; script=base/'scripts/align.py'
    if (base/'alignment/COMPLETE.json').exists(): raise ValueError('Unexpected alignment completion')
    sys.path.insert(0,str(script.parent)); mod=load(script)
    selected=prefix(script.read_text()); exec(compile(selected,str(script),'exec'),mod.__dict__)
    inputs=mod.main()
    lock=read(Path(run)/'LOCK.json')
    if sha(base/'inputs/RAW_SHARED9.npy')!=lock['files'][EXP+'/inputs/RAW_SHARED9.npy']['sha256']:
        raise ValueError('Rebuilt raw reference is not the accepted alignment cohort')
    for name,digest in ALIGN_SHA.items():
        if sha(base/'alignment'/name)!=digest: raise ValueError('Alignment changed')
    write(base/'alignment/COMPLETE.json',{'status':'complete','mode':'reuse_accepted_alignment_without_refit',
        'accepted_manifest':lock['accepted_manifest'],'array_sha256':ALIGN_SHA,
        'rebuilt_raw_sha256':sha(base/'inputs/RAW_SHARED9.npy')})
    write(stage_root(run)/'reference_inputs/PREFIX.json',{'source_sha256':sha(script),
        'selected_ast':ast.dump(selected,include_attributes=False),'input_hashes':inputs,
        'scientific_input_statements_unchanged':True,'ramp_fit_called':False})

def r_probe(base, env):
    code='stopifnot(getRversion()=="4.3.2"); for(p in c("survival","data.table")) cat(p,as.character(packageVersion(p)),"\\n"); cat(as.character(getRversion()),"\\n"); cat(Sys.getlocale(),"\\n")'
    p=subprocess.run([R_EXEC,'--vanilla','-e',code],cwd=base,env=env,text=True,capture_output=True)
    if p.returncode or not p.stdout.startswith('survival 3.5.7 \ndata.table 1.17.8 \n4.3.2 \n'):
        raise ValueError('R runtime differs: '+p.stdout+p.stderr)
    return {'stdout':p.stdout,'stderr':p.stderr,'executable_sha256':sha(Path(R_EXEC).resolve())}

def survival(run):
    base=Path(run)/'tree'/EXP; folder=stage_root(run)/'survival'
    env=os.environ.copy(); env.update(TMPDIR='tmp',TMP='tmp',TEMP='tmp')
    before=r_probe(base,env); write(folder/'R_environment_before.json',before)
    cmd=[R_EXEC,'--vanilla',str(base/'scripts/survival.R'),str(base)]
    write(folder/'R_COMMAND.json',{'argv':cmd,'cwd':str(base),'TMPDIR':'tmp'})
    with (folder/'R.stdout').open('x') as out,(folder/'R.stderr').open('x') as err:
        result=subprocess.run(cmd,cwd=base,env=env,stdout=out,stderr=err)
    if result.returncode: raise RuntimeError('Frozen survival script failed')
    after=r_probe(base,env); write(folder/'R_environment_after.json',after)
    if before!=after: raise ValueError('R runtime changed')
