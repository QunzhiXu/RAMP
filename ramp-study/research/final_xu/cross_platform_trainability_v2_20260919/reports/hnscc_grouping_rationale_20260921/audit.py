"""Label/marker/count evidence for HNSCC grouping; no training or score-driven grouping."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from trainability.core import file_sha256
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

PAIR_CONTRASTS=[
    ('CD8','CD8 T',['CD8 T'],'CD4 T',['CD4 T']),
    ('CD4','CD4 T',['CD4 T'],'CD8 T',['CD8 T']),
    ('CD38','Plasma',['Plasma'],'B',['B']),
    ('CD20','B',['B'],'Plasma',['Plasma']),
    ('CD21','B',['B'],'Plasma',['Plasma']),
    ('CD56','NK / gdT',['NK / gdT'],'CD8 T',['CD8 T']),
    ('CD3','CD4/CD8 T',['CD4 T','CD8 T'],'NK / gdT',['NK / gdT']),
    ('CD31','Vascular',['Vascular'],'Stromal',['Stromal']),
    ('CD31','Vascular',['Vascular'],'Epithelial / tumor',['Epithelial / tumor']),
    ('CD45','Immune',['B','Plasma','CD4 T','CD8 T','Myeloid','NK / gdT'],
     'Non-immune',['Epithelial / tumor','Stromal','Vascular']),
]

def main():
    manifest_path=ROOT/'data/real_inputs_v0_3_r2/hnscc_story/manifest.json'
    man=json.loads(manifest_path.read_text());markers=man['markers'];classes=man['classes']
    paths={'original_full':ROOT/'data/real_inputs_v0_3_r2/hnscc_story',
        'cap20':ROOT/'exploration_donor_cap20_20260921/inputs/hnscc_story'}
    inputs={str(manifest_path):file_sha256(manifest_path)};rows=[];counts=[]
    for pool,folder in paths.items():
        md=pd.read_csv(folder/'metadata.csv.gz',dtype=str,keep_default_na=False)
        raw=np.load(folder/'native.npy')
        for p in [folder/'metadata.csv.gz',folder/'native.npy']:inputs[str(p)]=file_sha256(p)
        for ds,g in md.groupby('dataset'):
            for cl in classes:
                cell=g[g.celltype.eq(cl)]
                counts.append(dict(pool=pool,dataset=ds,celltype=cl,cells=len(cell),patients=cell.patient_id.nunique()))
        for marker,pos_name,pos_classes,neg_name,neg_classes in PAIR_CONTRASTS:
            col=markers.index(marker)
            for (ds,unit),group in md.groupby(['dataset','unit_key']):
                g=group[group.celltype.isin(pos_classes+neg_classes)]
                positive=g.celltype.isin(pos_classes).to_numpy();np_=int(positive.sum());nn=len(g)-np_
                row=dict(pool=pool,dataset=ds,unit_key=unit,patient=group.patient_id.iloc[0],
                    marker=marker,positive=pos_name,negative=neg_name,n_positive=np_,n_negative=nn,
                    status='insufficient_support',auc=np.nan)
                if min(np_,nn)>=5:
                    rank=rankdata(raw[g.index,col],method='average')
                    row.update(status='ok',auc=float((rank[positive].sum()-np_*(np_+1)/2)/(np_*nn)))
                rows.append(row)
    all_rows=pd.DataFrame(rows);all_rows.to_csv(OUT/'MARKER_CONTRASTS_PER_DONOR.csv',index=False)
    keys=['pool','dataset','marker','positive','negative']
    summary=all_rows.groupby(keys).agg(mean_donor_auc=('auc','mean'),
        median_donor_auc=('auc','median'),evaluated_donors=('auc','count'),total_donors=('unit_key','count')).reset_index()
    summary.to_csv(OUT/'MARKER_CONTRASTS_SUMMARY.csv',index=False)
    pd.DataFrame(counts).to_csv(OUT/'JOINT_CLASS_COUNTS.csv',index=False)
    support_path=ROOT/'reports/sampling_feasibility_check_20260921/PER_CLASS_SOURCE_SPLIT.csv'
    support=pd.read_csv(support_path);support=support[support.task.eq('hnscc_story')]
    support.to_csv(OUT/'CURRENT_SUPERVISION_SUPPORT.csv',index=False)
    inputs[str(support_path)]=file_sha256(support_path)
    mappings={'original_9':{c:c for c in classes},
        'B_lineage_8':{c:('B lineage' if c in ['B','Plasma'] else c) for c in classes},
        'B_T_lineage_7':{c:('B lineage' if c in ['B','Plasma'] else 'CD4/CD8 T' if c in ['CD4 T','CD8 T'] else c) for c in classes},
        'nonimmune_coverage_check_only':{c:('Non-immune' if c in ['Epithelial / tumor','Stromal','Vascular'] else c) for c in classes}}
    sums=[]
    quantities=['fit_pool_cells','validation_pool_cells','actual_fit_cells','actual_validation_cells']
    for name,mapping in mappings.items():
        d=support.copy();d['group']=d.celltype.map(mapping)
        ds=d.groupby(['pool','source','seed','group'])[quantities].sum().reset_index()
        ds.insert(0,'mapping',name);sums.append(ds)
    pd.concat(sums).to_csv(OUT/'GROUPING_SUPPORT_COUNTS_ONLY.csv',index=False)
    for p,h in inputs.items():assert file_sha256(p)==h,p
    (OUT/'CHECKS.json').write_text(json.dumps(dict(no_alignment_or_training=True,no_labels_or_donors_changed=True,
        no_new_performance_rescoring=True,auc_direction_prespecified=True,
        minimum_per_class_per_unit_for_auc=5,
        caveat='Marker-label concordance is descriptive; provider labels may use these same markers. AUROC is not independent biological validation.',
        input_hashes=inputs),indent=2)+'\n')
    print(summary[summary.pool.eq('cap20')].to_string(index=False))

if __name__=='__main__':main()
