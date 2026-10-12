from pathlib import Path
import json,shutil
HERE=Path(__file__).resolve().parent
plan=json.loads((HERE/'REVISION_PLAN.json').read_text());backup=HERE/'before_vector_compaction';backup.mkdir()
for item in plan:
    new=Path(item['new']);shutil.copy2(new/'ridge_display_audit.csv',backup/(item['name']+'.csv'))
p=HERE/'ridge_common.py';shutil.copy2(p,backup/p.name);s=p.read_text()
s=s.replace("x=centers;y=level+.70*density", """# Remove only redundant vertices within exactly flat runs for PDF/SVG
        # export. Density calculation and all non-flat curve vertices are kept.
        changed=np.diff(density)!=0
        keep=np.r_[True,changed] | np.r_[changed,True]
        x=centers[keep];y=level+.70*density[keep]""")
p.write_text(s)
print('Compacted only collinear flat-run export vertices; preserved full density arrays.')
