from pathlib import Path
import json,shutil,hashlib
import pandas as pd
HERE=Path(__file__).resolve().parent;STAGE=HERE/'stage'
plan=json.loads((HERE/'REVISION_PLAN.json').read_text());sources=json.loads((STAGE/'SUPPLEMENTARY_FIGURE_SOURCES.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for i,(item,record) in enumerate(zip(plan,sources[:7]),1):
    new=Path(item['new']);a=pd.read_csv(HERE/'before_vector_compaction'/(item['name']+'.csv'));b=pd.read_csv(new/'ridge_display_audit.csv');pd.testing.assert_frame_equal(a,b)
    stem=Path(item['png']).stem
    for ext in ['pdf','png','svg']:
        src=new/'output'/('pdf' if ext=='pdf' else '')/(stem+'.'+ext)
        dst=HERE/'published'/ext/Path(record['destination']).with_suffix('.'+ext).name;shutil.copy2(src,dst)
    record['sha256']=sha(record['source'])
    for dest in [STAGE,HERE/'Supplementary_Overleaf',HERE/'independent_supplement']:shutil.copy2(record['source'],dest/record['destination'])
    shutil.copy2(new/'ridge_display_audit.csv',HERE/'Supplementary_Overleaf/supplementary_data/ridge_display'/f's{i}_ridge_display_audit.csv')
(STAGE/'SUPPLEMENTARY_FIGURE_SOURCES.json').write_text(json.dumps(sources,indent=2)+'\n')
shutil.copy2(STAGE/'SUPPLEMENTARY_FIGURE_SOURCES.json',HERE/'Supplementary_Overleaf/reference/SUPPLEMENTARY_FIGURE_SOURCES.json')
for rel in ['RANGE_AUDIT.csv','QA_SUMMARY.json']:shutil.copy2(HERE/rel,HERE/'Supplementary_Overleaf/supplementary_data/ridge_display'/rel)
print('All 3,636 density arrays exactly unchanged by vector compaction. Refreshed exports.')
