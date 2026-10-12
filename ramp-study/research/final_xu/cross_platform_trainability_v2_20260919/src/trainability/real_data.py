"""Read registered real inputs; fitting pools and answers have separate interfaces."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from .core import ROOT, require, file_sha256, digest
from .inputs import LabelFreeInput, FittingPool, build_assay_input


@dataclass
class RealTask:
    name: str
    native: np.ndarray
    metadata: pd.DataFrame
    markers: tuple
    contracts: dict
    receipt: dict
    pool: FittingPool

    def input(self, method, masked=None):
        if masked is not None:
            bad=[d for d,c in self.contracts.items() if not c['j2_eligible']]
            require(not bad, f'J2 upstream dependency unresolved in full U: {bad}')
        contracts={d:dict(c,review_status='verified_no_target_dependency',
            input_contract_version='real_inputs_v0.3',evidence_path=self.receipt['manifest_path'])
            for d,c in self.contracts.items()}
        if method in {'ramp','harmony_direct'}:
            contracts={d:dict(c,operator='identity') for d,c in contracts.items()}
        return build_assay_input(self.native,cell_ids=self.pool.cell_ids,
            datasets=self.pool.datasets,units=self.pool.units,markers=self.markers,
            canonical_marker=self.receipt['canonical_markers'],masked_marker=masked,contracts=contracts)

    def subset(self, indices, name):
        # Only infrastructure pilots may construct a smaller registered U.
        idx=np.asarray(indices,dtype=int)
        require(len(set(idx))==len(idx) and (idx>=0).all() and (idx<len(self.metadata)).all(), 'Bad pilot subset')
        md=self.metadata.iloc[idx].reset_index(drop=True)
        pool=FittingPool(f'PILOT:{self.name}:{name}',tuple(md.cell_id),tuple(md.dataset),tuple(md.unit_key))
        return RealTask(self.name,self.native[idx].copy(),md,self.markers,self.contracts,
            dict(self.receipt,pilot=True,parent_pool_sha256=self.pool.sha256),pool)


def load_task(name, root=None):
    root=Path(root) if root else ROOT/'data/real_inputs_v0_3_r2'
    folder=root/name
    receipt=json.loads((folder/'manifest.json').read_text())
    for f,h in receipt['outputs'].items():
        require(file_sha256(folder/f)==h,f'Registered input changed: {f}')
    md=pd.read_csv(folder/'metadata.csv.gz',dtype=str,keep_default_na=False)
    native=np.load(folder/'native.npy',allow_pickle=False)
    pool=FittingPool(receipt['pool_id'],tuple(md.cell_id),tuple(md.dataset),tuple(md.unit_key))
    require(pool.sha256==receipt['pool_sha256'],'Pool registration changed')
    require(native.shape==(len(md),len(receipt['markers'])),'Registered shape mismatch')
    receipt['manifest_path']=str(folder/'manifest.json')
    return RealTask(name,native,md,tuple(receipt['markers']),receipt['contracts'],receipt,pool)


def pilot_indices(task, per_dataset=1500, seed=20260919):
    # Donor-stratified, label-blind; neither expression nor test scores used.
    rng=np.random.default_rng(seed)
    picked=[]
    for ds,group in task.metadata.groupby('dataset',sort=False):
        units=list(group.groupby('unit_key',sort=True))
        quota=per_dataset//len(units)
        require(quota>=2,'Pilot too small to cover all units')
        chosen=[]
        for _,g in units:
            chosen.extend(rng.choice(g.index,min(quota,len(g)),replace=False).tolist())
        remaining=np.setdiff1d(group.index,chosen)
        extra=min(per_dataset-len(chosen),len(remaining))
        chosen.extend(rng.choice(remaining,extra,replace=False).tolist())
        picked.extend(chosen)
    return np.sort(picked)
