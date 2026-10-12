"""Small route adapters; original numerical functions and constants remain frozen."""
import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from common import EXP, read, write, sha, stage_root, record
from prepare import R_EXEC

def load(path):
    spec=importlib.util.spec_from_file_location('hnscc_frozen_'+Path(path).stem,path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def r_probe(base,env):
    code='stopifnot(getRversion()=="4.3.2"); for(p in c("survival","data.table")) cat(p,as.character(packageVersion(p)),"\\n"); cat(as.character(getRversion()),"\\n"); cat(Sys.getlocale(),"\\n")'
    p=subprocess.run([R_EXEC,'--vanilla','-e',code],cwd=base,env=env,text=True,capture_output=True)
    if p.returncode or not p.stdout.startswith('survival 3.5.7 \ndata.table 1.17.8 \n4.3.2 \n'):
        raise ValueError('R runtime differs: '+p.stdout+p.stderr)
    return {'stdout':p.stdout,'stderr':p.stderr,'executable_sha256':sha(Path(R_EXEC).resolve())}

def survival(run):
    base=Path(run)/'tree'/EXP; folder=stage_root(run)/'survival'
    env=os.environ.copy(); env.update(TMPDIR='tmp',TMP='tmp',TEMP='tmp')
    before=r_probe(base,env); write(folder/'R_environment_before.json',before)
    cmd=[R_EXEC,'--vanilla',str(base/'survival_motivation/02_run_survival_and_plot.R')]
    write(folder/'R_COMMAND.json',{'argv':cmd,'cwd':str(base),'TMPDIR':'tmp'})
    with (folder/'R.stdout').open('x') as out,(folder/'R.stderr').open('x') as err:
        result=subprocess.run(cmd,cwd=base,env=env,stdout=out,stderr=err)
    if result.returncode: raise RuntimeError('Frozen survival script failed')
    after=r_probe(base,env); write(folder/'R_environment_after.json',after)
    if before!=after: raise ValueError('R runtime changed')

def linked_formal_path(path,base,lock):
    """Relocation-only counterpart of the historical in-tree path guard."""
    path=Path(path); resolved=path.resolve(strict=True)
    if not path.absolute().is_relative_to(base.absolute()): raise ValueError('Not a formal runtime path')
    if any(p.lower() in {'test','test0827'} for p in path.relative_to(base).parts):
        raise ValueError('Forbidden exploratory runtime dependency')
    if path.is_symlink():
        rel=EXP+'/'+str(path.relative_to(base))
        if rel not in lock['input_paths'] or resolved!=Path(lock['input_paths'][rel]).resolve():
            raise ValueError('Unrecognized formal input link')
        expected={k:lock['files'][rel][k] for k in ('bytes','sha256')}
        if record(resolved)!=expected: raise ValueError('Formal input link changed')
    elif not resolved.is_relative_to(base.resolve()): raise ValueError('Formal path escaped')

def return_state(run):
    base=Path(run)/'tree'/EXP; mod=load(base/'spatial_return_survival/01_return_fixed_program.py')
    lock=read(Path(run)/'LOCK.json')
    mod.assert_formal_path=lambda p: linked_formal_path(p,base,lock)
    mod.main()
    write(stage_root(run)/'return_state/ADAPTATION.json',{
        'only_override':'assert_formal_path permits exact locked read-only input links',
        'numerical_functions_unchanged':True,'source_sha256':sha(base/'spatial_return_survival/01_return_fixed_program.py')})

def fixed_fraction_function(source):
    """Unchanged comparison prefix, omitting comparison to the unselected 20% arm."""
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='compare')
    cuts=[i for i,n in enumerate(node.body) if isinstance(n,ast.Assign)
          and any(isinstance(t,ast.Name) and t.id=='baseline' for t in n.targets)]
    if len(cuts)!=1: raise ValueError('Fraction prefix boundary differs')
    node.body=node.body[:cuts[0]]+ast.parse('return pd.DataFrame(counts), donor, summary, assignment').body
    return ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))

def molecular30(run):
    import numpy as np
    import pandas as pd
    from scipy.stats import rankdata
    base=Path(run)/'tree'/EXP
    path=base/'big_figure/panel_E/group_fraction_scan_20260910/scan_group_fractions.py'
    mod=load(path)
    mod.INPUTS={'adt':base/'sampled_data/rahim_citeseq/complete/GSE212797_adata_ADT.h5ad',
                'rna':base/'sampled_data/rahim_citeseq/complete/GSE212797_adata.h5ad'}
    mod.FRACTIONS=[.30]
    selected=fixed_fraction_function(path.read_text())
    exec(compile(selected,str(path),'exec'),mod.__dict__)
    d=pd.read_csv(base/'biomarker_discovery/screen/rahim_tumor_cd8_biomarker_programs.csv.gz',dtype={'donor':str})
    if len(d)!=4788 or not d.cell_uid.is_unique or d.donor.nunique()!=3: raise ValueError('Molecular cohort differs')
    d['state_rank']=d.groupby('donor')[mod.SCORE].transform(lambda v:(rankdata(v)-.5)/len(v))
    values=mod.load_features(d)
    counts,donor,summary,assignment=mod.compare(d,values)
    out=base/'molecular30'; out.mkdir()
    for name,frame in [('group_counts',counts),('donor_contrasts',donor),('equal_donor_contrasts',summary)]:
        frame.to_csv(out/(name+'.csv'),index=False)
    assignment.to_csv(out/'cell_group_assignments.csv.gz',index=False)
    write(stage_root(run)/'molecular30/ADAPTATION.json',{'source_sha256':sha(path),
        'selected_ast':ast.dump(selected,include_attributes=False),'fractions':[.30],
        'source_program':'newly computed formal screen; no old scores or contrast tables read',
        'omitted':'20pct-relative effect, baseline check, fraction search and plots'})
    # Current panel D's 18 points, independently routed from the new 107-marker screen.
    markers=['TIGIT--VSTM3','CD39','CD224','CD134--OX40','TCR-A--TCR-B','CD83-1']
    effects=pd.read_csv(base/'biomarker_discovery/screen/marker_donor_effects_long.csv',dtype={'donor':str})
    effects=effects[effects.variant.eq('paper_clr')&effects.marker_raw.isin(markers)]
    if len(effects)!=18: raise ValueError('Display marker grid differs')
    out=base/'display_values'; out.mkdir(); effects.to_csv(out/'panel_D.csv',index=False)

def phenotype(run):
    import numpy as np
    import pandas as pd
    base=Path(run)/'tree'/EXP
    helper=load(base/'biomarker_discovery/01_project_spatial_state.py')
    raw,aligned,metadata,markers=helper.read_formal_alignment()
    selected=helper.read_selected_patients()
    curves,audit=helper.build_haist_curves(raw,aligned,metadata,markers)
    cells,full,_=helper.load_full_haist_pt_cd8(selected,curves,markers,audit)
    # Rebuild the old auxiliary full-cell representation, rather than supply its cached values.
    names=['LAG3','IL10','GranzymeB','Ki67']; blocks=[]; ids=set(cells.UID)
    for frame in pd.read_csv(helper.HAIST_CELL_TABLE,usecols=['UID']+names,dtype={'UID':str},chunksize=150000):
        blocks.append(frame[frame.UID.isin(ids)])
    measured=pd.concat(blocks).set_index('UID').loc[cells.UID]
    spatial=pd.read_csv(helper.SPATIAL_RESPONSE,dtype={'UID':str,'Pat_ID':str,'unique_region':str}).set_index('UID').loc[cells.UID]
    md=spatial[['Pat_ID','unique_region','x','y']].reset_index().rename(columns={'UID':'cell_uid','Pat_ID':'patient'})
    md[names]=measured[names].to_numpy()
    out=base/'rebuilt_phenotype'; out.mkdir()
    md.to_csv(out/'target_all_metadata.csv.gz',index=False)
    np.save(out/'target_all_aligned.npy',full)
    folder=base/'spatial_score_review_20260913'; sys.path.insert(0,str(folder))
    mod=load(folder/'check_codex_phenotype.py'); mod.OLD=out; mod.main()
    write(stage_root(run)/'phenotype/ADAPTATION.json',{'overridden_global':'OLD -> freshly rebuilt full PT-CD8 mapping and provider measurements',
        'cells':len(md),'patients':int(md.patient.nunique()),'old_cached_arrays_read':False})

def fixed_roi(run):
    import pandas as pd
    from scipy.stats import spearmanr
    base=Path(run)/'tree'/EXP
    d=pd.read_csv(base/'spatial_score_review_20260913/codex_measured_protein_cells.csv.gz',dtype={'patient':str,'cell_uid':str})
    q=d[d.unique_region.eq('521_S1reg012')]
    if len(q)!=343 or set(q.patient)!={'29770'}: raise ValueError('Fixed ROI identity differs')
    out=base/'fixed_roi'; out.mkdir(); q.to_csv(out/'large_roi_cells.csv',index=False)
    write(out/'SUMMARY.json',{'roi':'521_S1reg012','patient':'29770','cells':343,
        'rho':float(spearmanr(q.predicted_score,q.LAG3).statistic),
        'selection':'fixed published post-hoc illustrative ROI; no scan or reselection'})
