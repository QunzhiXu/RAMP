"""Join original annotations to the exact frozen cell IDs; no relabeling."""
from pathlib import Path
import csv
import gzip
import json
from collections import Counter

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FX=ROOT.parent
SPECS={
 'bone_marrow': [('benchmark_results/bone_marrow/ramp/alignment/input/cell_metadata.csv.gz','cell_id','provider_label_l1')],
 'tnbc': [('benchmark_results/tnbc/ramp/alignment/input/cell_metadata.csv.gz','cell_id','provider_label_l2')],
 'melanoma': [('benchmark_data/melanoma/input/locked_metadata.csv.gz','cell_id','provider_celltype_original')],
 'lung_cancer': [('benchmark_data/lung_cancer/input/locked_metadata.csv.gz','cell_id','annotation_label_original')],
 'lung_story': [('story_lung_HLA_ABC/data_umap/data/CELL_METADATA.csv.gz','cell_key','provider_cell_type')],
 'melanoma_story': [
   ('figure_melanoma/inputs/cohort_package/engelbrecht_citeseq/locked_metadata.csv.gz','source_cell_id','Broad_cell.type.1'),
   ('figure_melanoma/inputs/cohort_package/moldoveanu_imc/locked_metadata.csv.gz','cell_uid','Cluster.v2')],
 'hnscc_story': [('figure_HNSCC/alignment/input/cell_metadata.csv.gz','cell_uid','provider_cell_type')],
}


def main():
    out=[];checks={}
    for task,sources in SPECS.items():
        inp=ROOT/'data/real_inputs_v0_3_r2'/task
        classes=json.loads((inp/'manifest.json').read_text())['classes']
        with gzip.open(inp/'metadata.csv.gz','rt') as f:md={r['cell_id']:r for r in csv.DictReader(f)}
        counts=Counter();seen=set()
        for rel,key,col in sources:
            with gzip.open(FX/rel,'rt') as f:
                for r in csv.DictReader(f):
                    cell=r[key]
                    if cell not in md:continue
                    assert cell not in seen,(task,cell)
                    seen.add(cell);frozen=md[cell]
                    if frozen['celltype'] in classes:
                        counts[frozen['dataset'],frozen['celltype'],r[col],rel,col]+=1
        assert seen==set(md),(task,len(seen),len(md))
        for (ds,cl,label,rel,col),n in sorted(counts.items()):
            out.append(dict(task=task,dataset=ds,celltype=cl,provider_label=label,cells=n,
                            source_file=rel,provider_column=col))
        checks[task]={'frozen_cells':len(md),'matched_cells':len(seen),'unique_exact_cell_id_join':True}
    with (HERE/'PROVIDER_LABEL_COMPOSITION.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    (HERE/'PROVIDER_CHECKS.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
