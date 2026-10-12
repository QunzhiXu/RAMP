from pathlib import Path
import json,hashlib,shutil,os,re
import pandas as pd
from ridge_common import PALETTE,STYLE,sha
HERE=Path(__file__).resolve().parent;SUP=HERE.parent
# Dataset order follows the existing UMAP legend; the ridge legend and
# top-to-bottom stacking use this explicit order instead of DataFrame order.
SPEC={
'Bone marrow':[('BM CODEX','CODEX','CODEX'),('BM CyTOF','CyTOF','CyTOF'),('Triana AbSeq','AbSeq','AbSeq')],
'TNBC':[('Tietscher IMC','IMC1','IMC'),('Meyer IMC','IMC2','IMC2'),('Shiao CODEX','CODEX','CODEX')],
'Melanoma three-platform':[('MEL_Hickey2023_CODEX','CODEX','CODEX'),('MEL_Hoch2021_IMC','IMC','IMC'),('MEL_Ptacek2021_MIBI','MIBI','MIBI')],
'Lung cancer benchmark':[('Senosain CyTOF','CyTOF','CyTOF'),('Cords IMC','IMC','IMC'),('Monkman CODEX','CODEX','CODEX')],
'Lung HLA-ABC story':[('Senosain CyTOF','CyTOF','CyTOF'),('Zheng mIF','mIF','mIF'),('Cords IMC','IMC','IMC')],
'Melanoma story':[('Moldoveanu IMC','IMC','IMC'),('Engelbrecht CITE-seq','CITE-seq','CITE-seq')],
'HNSCC story':[('GSE212797_ADT','CITE-seq','CITE-seq'),('Haist_CODEX','CODEX','CODEX')]}

def main():
    plan=[]
    for name,spec in SPEC.items():
        pointer=SUP/name/'CURRENT_PREVIEW.json';previous=json.loads(pointer.read_text())
        old=Path(previous.get('folder',pointer.parent/previous['version']));new=old.parent/'ridge_uniform_20260922'
        new.mkdir(exist_ok=False)
        # Copy small frozen config inputs; link data and coordinates read-only.
        for p in old.iterdir():
            if p.is_file() and (p.suffix=='.json' or p.name=='extract_fastmnn.R'):
                if p.name not in ['MANIFEST.json','QA.json','CURRENT_PREVIEW.json','RENDERING_AUDIT.json']:
                    shutil.copy2(p,new/p.name)
            elif p.name in ['coordinates','display_metadata.csv.gz','display_rows.npy','fastmnn_reconstructed.csv.gz']:
                (new/p.name).symlink_to(os.path.relpath(p,new),target_is_directory=p.is_dir())
        a=pd.read_csv(old/'ridge_display_audit.csv',dtype={'unit_key':str})
        units=a[['dataset','unit_key','n_cells']].drop_duplicates();assert not units.duplicated(['dataset','unit_key']).any()
        platforms=[];selection=[]
        for d,label,tech in spec:
            rows=units[units.dataset.eq(d)].sort_values('unit_key');ids=rows.unit_key.tolist()
            keep=len(ids)
            if name=='Lung HLA-ABC story':keep=10
            elif name=='HNSCC story' and d=='Haist_CODEX':keep=27
            rank=sorted(ids,key=lambda u:hashlib.sha256(f'20260922|{d}|{u}'.encode()).hexdigest())
            chosen=sorted(rank[:keep]);assert len(chosen)==keep
            platforms.append(dict(dataset=d,name=label,technology=tech,color=PALETTE[tech],available_units=len(ids),selected_unit_ids=chosen))
            for row in rows.itertuples():
                selection.append(dict(dataset=d,unit_key=row.unit_key,n_cells=int(row.n_cells),selected=row.unit_key in chosen,
                    selection_rank=rank.index(row.unit_key)+1,selection_hash=hashlib.sha256(f'20260922|{d}|{row.unit_key}'.encode()).hexdigest()))
        sf=pd.DataFrame(selection);sf.to_csv(new/'ridge_unit_selection.csv',index=False)
        cfg=dict(case=name,original_folder=str(old),palette_mode='fixed_technology_across_all_seven_figures',platforms=platforms,
            full_units=len(sf),ridge_units=int(sf.selected.sum()),full_cells=int(sf.n_cells.sum()),ridge_cells=int(sf.loc[sf.selected,'n_cells'].sum()),
            selection='All units retained' if sf.selected.all() else 'Display-only subset: deterministic SHA256 ranking of 20260922|dataset|unit ID; smallest hashes selected within platform; no expression, method or performance used',
            selection_seed=20260922,style=STYLE)
        (new/'RIDGE_CONFIG.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False)+'\n')
        source=(old/'build_figure.py').read_text()
        # New builders share one renderer and validate exactly the selected rows.
        target='from scipy.ndimage import gaussian_filter1d\n'
        assert target in source
        source=source.replace(target,target+"sys.path.insert(0,str(HERE.parents[1]/'ridge_style_update_20260922'))\nimport ridge_common\nRIDGE_CONFIG=json.loads((HERE/'RIDGE_CONFIG.json').read_text())\n")
        start=source.index('def ridge(');end=source.index('\ndef main(',start)
        source=source[:start]+'''def ridge(ax,v,meta,method,marker,style):
    return ridge_common.draw_ridge(ax,v,meta,method,marker,RIDGE_CONFIG,AUDIT)

'''+source[end:]
        source=source.replace('def main():\n','def main():\n    track(HERE/\'RIDGE_CONFIG.json\');track(Path(ridge_common.__file__))\n')
        # Put platform legends directly over their own blocks, not in a shared header.
        source=source.replace("loc='upper right',bbox_to_anchor=(201/WIDTH,1-17/HEIGHT)","loc='upper left',bbox_to_anchor=(9/WIDTH,1-34/HEIGHT)")
        source=source.replace("prop={'size':8,'weight':'bold'}","prop={'size':7.4,'weight':'bold'}")
        anchor="    text(fig,28.5,37.5,'Dataset'"
        assert anchor in source
        source=source.replace(anchor,"    ridge_common.add_legend(fig,RIDGE_CONFIG)\n"+anchor)
        source=source.replace(",37.5,",",40.,")
        source=source.replace('top=45+i*','top=46.5+i*')
        source=source.replace('1-267/HEIGHT','1-270/HEIGHT')
        source=re.sub(r'    assert len\(audit\)==6\*3\*\d+\n    assert audit\.groupby\(\[\x27method\x27,\x27marker\x27\]\)\.n_cells\.sum\(\)\.eq\(\d+\)\.all\(\)',
                      '    ridge_common.validate_audit(audit,RIDGE_CONFIG)',source)
        assert 'assert len(audit)==6*3*' not in source
        key="    (HERE/'MANIFEST.json').write_text"
        assert key in source;source=source.replace(key,"    report.update(ridge_common.manifest_extra(RIDGE_CONFIG))\n"+key)
        # Rendering changes do not replace the existing UMAP coordinates.
        (new/'build_figure.py').write_text(source);compile(source,str(new/'build_figure.py'),'exec')
        stem=Path(previous.get('png',str(old/'output/melanoma_story_supplement.png'))).name
        item=dict(name=name,old=str(old),new=str(new),pointer=str(pointer),previous_pointer=previous,png=stem,
            old_hashes={str(p):sha(p) for p in [old/'build_figure.py',old/'MANIFEST.json',old/'ridge_display_audit.csv',*list((old/'output').glob('*.png')),*list((old/'output/pdf').glob('*.pdf'))]})
        plan.append(item)
    (HERE/'REVISION_PLAN.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n')
    (HERE/'PALETTE.json').write_text(json.dumps(PALETTE,indent=2)+'\n')
    print([(p['name'],json.loads((Path(p['new'])/'RIDGE_CONFIG.json').read_text())['ridge_units']) for p in plan])
if __name__=='__main__':main()
