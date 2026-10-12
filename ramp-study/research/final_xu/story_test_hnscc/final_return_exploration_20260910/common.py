from pathlib import Path
import hashlib
import importlib.util
import json
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FORMAL=ROOT/'figure_HNSCC'
HELPER=FORMAL/'biomarker_discovery/01_project_spatial_state.py'
spec=importlib.util.spec_from_file_location('formal_projection',HELPER)
helper=importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
ALIASES={'TIGIT--VSTM3':'TIGIT','CD39':'CD39','CD224':'CD224','CD134--OX40':'OX40','TCR-A--TCR-B':'TCRab','CD83-1':'CD83','CD137':'CD137','CD161':'CD161','CD335--NKp46':'NKp46','CD141--Thrombomodulin':'CD141','CD49b':'CD49b','CD152':'CD152'}
SOURCE_BROAD={'B cell':'B','CD4 T cell':'CD4','CD8 T cell':'CD8','Endothelial cell':'Endothelial','Fibroblast':'Fibroblast','Macrophage':'Myeloid','Mast cell':'Mast','NK/gd T cell':'NK_gdT','Plasma cell':'Plasma','Proliferating cell':'Proliferating','Salivary gland':'Other_source','cDC':'DC','pDC':'DC'}
TARGET_BROAD={'B':'B','CAF':'Fibroblast','CD4T':'CD4','CD8T':'CD8','DC':'DC','LEC':'Endothelial','NK':'NK_gdT','Treg':'CD4','VSMC_myofibroblast':'Fibroblast','gdT':'NK_gdT','granulocyte':'Unsupported','macrophages':'Myeloid','mast':'Mast','nerve':'Unsupported','plasma':'Plasma','stromal':'Fibroblast','tumor':'Unsupported','tumor_CA9':'Unsupported','vasculature':'Endothelial'}
READOUTS=['FoxP3','GranzymeB','Ki67','LAG3','IDO1','IL10','HIF1a','CD73','CD68','CD163','CD206','FAP','CD90','CXCL12','TenascinC','CollagenIV','Periostin']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()


def save_json(value,path):
    def clean(v):
        if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
        if isinstance(v,np.generic):return clean(v.item())
        if isinstance(v,float) and not np.isfinite(v):return None
        if isinstance(v,Path):return str(v)
        return v
    Path(path).write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')


def pct_matrix(v,groups):
    v=np.asarray(v,float)
    if v.ndim==1:v=v[:,None]
    out=np.empty_like(v)
    for group in pd.unique(groups):
        rows=np.flatnonzero(groups==group)
        out[rows]=(rankdata(v[rows],axis=0)-.5)/len(rows)
    return out


def bh(p):
    p=np.asarray(p,float);out=np.full(len(p),np.nan);ok=np.isfinite(p)
    order=np.argsort(p[ok]);pv=p[ok][order]
    adjusted=np.minimum.accumulate((pv*len(pv)/np.arange(1,len(pv)+1))[::-1])[::-1]
    a=np.empty(len(pv));a[order]=np.minimum(adjusted,1);out[ok]=a
    return out


def knn(tree,query,responses,k=30,chunk=30000):
    predictions=[];distances=[];indices=[];weights=[]
    for start in range(0,len(query),chunk):
        dd,ii=tree.query(query[start:start+chunk],k=min(k,len(responses)),workers=1)
        if dd.ndim==1:dd,ii=dd[:,None],ii[:,None]
        ww=np.exp(-dd/np.maximum(np.median(dd,axis=1,keepdims=True),1e-6))
        ww/=ww.sum(axis=1,keepdims=True)
        predictions.append(np.einsum('nk,nkf->nf',ww,responses[ii]))
        distances.append(np.median(dd,axis=1));indices.append(ii);weights.append(ww)
    return np.vstack(predictions),np.concatenate(distances),np.vstack(indices),np.vstack(weights)
