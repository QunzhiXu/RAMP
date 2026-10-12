from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent
REVIEW=HERE.parent
ROOT=REVIEW.parent.parent
INPUT=REVIEW/'codex_measured_protein_cells.csv.gz'
FILELIST=ROOT/'raw_data/hnscc/haist_2026_codex/selected_raw_images/file_list_study_component1.json'

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    h={str(p):sha(p) for p in [INPUT,FILELIST]}
    d=pd.read_csv(INPUT,dtype={'patient':str,'cell_uid':str})
    files={v['path']:v for v in json.loads(FILELIST.read_text())}
    rows=[]
    for roi,g in d.groupby('unique_region'):
        assert g.patient.nunique()==1
        n=len(g);valid=n>=3 and g.predicted_score.nunique()>1 and g.LAG3.nunique()>1
        rho=float(spearmanr(g.predicted_score,g.LAG3).statistic) if valid else np.nan
        path=f'Submission2/HNSCC/CODEX/raw_TIFF/{roi}.tif'
        rows.append(dict(roi=roi,patient=g.patient.iloc[0],n_cd8=n,rho=rho,estimable=valid,eligible_100=n>=100 and valid,eligible_200=n>=200 and valid,
                         official_image_listed=path in files,official_image_path=path,
                         official_image_size=files.get(path,{}).get('size'),
                         status='estimable' if valid else 'fewer_than_3_cells_or_constant_values'))
    r=pd.DataFrame(rows).sort_values(['rho','n_cd8','roi'],ascending=[False,False,True],na_position='last')
    assert len(r)==176 and r.patient.nunique()==74 and r.n_cd8.sum()==38299
    r['all_roi_rank']=np.arange(1,len(r)+1)
    r.to_csv(HERE/'all_176_roi_ranking.csv',index=False)
    top={}
    for minimum in [100,200]:
        b=r[r[f'eligible_{minimum}']].copy();b[f'rank_min{minimum}']=np.arange(1,len(b)+1)
        b.to_csv(HERE/f'eligible_min{minimum}_ranking.csv',index=False)
        top[f'min{minimum}']=b.iloc[0].to_dict()
        print(f'MINIMUM {minimum} CD8: {len(b)} ROIs')
        print(b[['roi','patient','n_cd8','rho','official_image_listed']].head(12).to_string(index=False))
    old=r[r.roi.eq('522_S2reg054')].iloc[0].to_dict()
    print('OLD',old)
    selected=top['min100'];selected['selection_rule']='highest continuous Spearman among complete ROIs with at least 100 CD8'
    (HERE/'selected_roi.json').write_text(json.dumps(selected,indent=2)+'\n')
    d[d.unique_region.eq(selected['roi'])].to_csv(HERE/'selected_roi_cells.csv',index=False)
    d[d.unique_region.eq(top['min200']['roi'])].to_csv(HERE/'large_roi_cells.csv',index=False)
    assert h=={str(p):sha(p) for p in [INPUT,FILELIST]}
    (HERE/'RANKING_AUDIT.json').write_text(json.dumps(dict(status='PASS',cells=38299,patients=74,rois=176,estimable=int(r.estimable.sum()),top=top,old_roi=old,
                input_hashes_unchanged=h,selection_uses_observed_association=True,full_cohort_unchanged=True),indent=2)+'\n')

if __name__=='__main__':main()
