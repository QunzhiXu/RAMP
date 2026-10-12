"""Match frozen original Lung cells to current Senosain annotations; no relabeling."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
ANN=ROOT.parent/'raw_data/lung_cancer/Senosain_2025_CyTOF_and_omics/annotation'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    files=[ROOT/'data/real_inputs_v0_3_r2/lung_cancer/metadata.csv.gz',ROOT/'data/real_inputs_v0_3_r2/lung_cancer/native.npy',ANN/'rough_annotation/senosain_140k_final_cell_annotations.csv.gz',ANN/'senosain_140k_markers35.csv',ANN/'senosain_140k_raw35.npy']
    hashes={str(p):sha(p) for p in files}
    md=pd.read_csv(files[0]);ann=pd.read_csv(files[2],keep_default_na=False)
    sen=md[md.dataset.eq('Senosain CyTOF')].copy();sen['original_row']=sen.index
    j=sen.merge(ann,on='source_cell_id',how='left',validate='one_to_one',suffixes=('','_new'),indicator=True)
    assert len(j)==20000 and j._merge.eq('both').all()
    assert j.celltype.eq(j.final_cell_type).all()
    assert j.donor.astype(str).eq(j.patient_id_new.astype(str)).all()
    j[['cell_id','source_cell_id','donor','celltype','final_cell_type','parent_cluster','subcluster','qc_recommendation','annotation_confidence']].to_csv(OUT/'SENOSAIN_ANNOTATION_MATCH.csv.gz',index=False)
    j.groupby(['celltype','final_cell_type','parent_cluster','subcluster','qc_recommendation']).size().rename('cells').reset_index().to_csv(OUT/'SENOSAIN_ANNOTATION_COUNTS.csv',index=False)
    pd.crosstab(j.celltype,j.final_cell_type).to_csv(OUT/'SENOSAIN_LABEL_CROSSTAB.csv')
    old=np.load(files[1],mmap_mode='r');new=np.load(files[4],mmap_mode='r');markers=pd.read_csv(files[3]).marker.tolist()
    wanted=['CD3','CD4','CD8','CD31','HLA_DR','KI67','VIMENTIN','CYTOKERATIN']
    x=np.asarray(old[j.original_row.to_numpy()],dtype=float);y=np.asarray(new[j.sampled_row_zero_based.to_numpy()][:,[markers.index(m) for m in wanted]],dtype=float)
    same=np.allclose(x,y,rtol=1e-6,atol=1e-6)
    c02=int(j.parent_cluster.eq('C02').sum());my=int(j.celltype.eq('Myeloid').sum())
    result=dict(all_20000_cells_match=True,all_broad_labels_identical=True,all_donor_ids_match=True,senosain_8_marker_values_match_140k_raw35=bool(same),maximum_absolute_difference=float(np.max(np.abs(x-y))),
        c02_cells=c02,total_myeloid=my,c02_fraction_of_myeloid=c02/my,
        excluded_or_pending_qc_cells=int(j.qc_recommendation.ne('keep').sum()),all_excluded_or_pending_already_nonjoint=bool(j.loc[j.qc_recommendation.ne('keep'),'celltype'].eq('Mix').all()),
        current_primary_policy='If retraining, keep C02 as in accepted Lung story primary; no automatic relabeling or exclusion.',no_refit=True,no_input_changes=True,script_sha256=sha(__file__))
    assert all(sha(p)==h for p,h in hashes.items())
    (OUT/'ANNOTATION_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'ANNOTATION_INPUT_HASHES.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
