import unittest
import numpy as np
import pandas as pd
from trainability.core import ProtocolError
from trainability.inputs import FittingPool,prepare_representation
from trainability.real_data import RealTask
from trainability.runner import make_plan,make_j3_plans,run_supervised
from trainability.heads import Candidate
from trainability.j4 import PretrainConfig,run_j4
from _artifacts import temporary_artifacts


def toy():
    rng=np.random.default_rng(7);n=360
    md=pd.DataFrame(dict(cell_id=[f'{d}-{i}' for d in 'ABC' for i in range(n)],
        dataset=np.repeat(list('ABC'),n),unit_key=[f'{d}:{i//90}' for d in 'ABC' for i in range(n)],
        patient_id=[f'{d}:{i//90}' for d in 'ABC' for i in range(n)],
        celltype=['x' if i%2 else 'y' for d in 'ABC' for i in range(n)],identity_level='patient'))
    x=rng.uniform(0,5,(len(md),4));x[:,1]=x[:,0]+rng.uniform(0,.1,len(md))
    pool=FittingPool('toy',tuple(md.cell_id),tuple(md.dataset),tuple(md.unit_key))
    return RealTask('toy',x,md,tuple('WXYZ'),{d:dict(operator='log1p',j2_eligible=True) for d in 'ABC'},
        dict(manifest_path='synthetic_fixture_only',canonical_markers={m:m for m in 'WXYZ'},classes=['x','y'],pilot=True),pool)


class IntegrationTests(unittest.TestCase):
    def test_all_methods_receive_masked_input(self):
        t=toy()
        for m in ['ramp','harmony_direct','cycombine','cytonorm','fastmnn','maxfuse','unaligned']:
            a=t.input(m,'W').x.copy();t.native[:,0]=np.nan
            np.testing.assert_array_equal(a,t.input(m,'W').x)
            t.native[:,0]=1.

    def test_unresolved_upstream_dependency_blocks_j2(self):
        t=toy();t.contracts['C']['j2_eligible']=False
        with self.assertRaises(ProtocolError):t.input('ramp','W')
        self.assertEqual(t.input('ramp').x.shape,t.native.shape)

    def test_actual_j1_and_j2_freeze_then_score(self):
        t=toy()
        with temporary_artifacts() as root:
            for marker in [None,'W']:
                rep=prepare_representation(t.input('unaligned',marker),'unaligned',fitting_pool=t.pool)
                plan,mode=make_plan(t,'A','B',fit_budget=80,validation_budget=60,seed=9,
                    marker=marker,calibration_per_unit=10 if marker else 0)
                r=run_supervised(t,rep,plan,sources=['A'],target='B',marker=marker,
                    candidates=[Candidate('linear')],seed=9,out=root/str(marker),split_mode=mode)
                self.assertEqual(r['total_supervision'],140+(40 if marker else 0))
                self.assertFalse(r['prediction_receipt']['target_answers_included'])
                self.assertTrue((root/str(marker)/'predictions.npz').is_file())

    def test_wrong_mask_rejected(self):
        t=toy();rep=prepare_representation(t.input('unaligned'),'unaligned',fitting_pool=t.pool)
        plan,mode=make_plan(t,'A','B',fit_budget=20,validation_budget=40,seed=9,marker='W',calibration_per_unit=10)
        with temporary_artifacts() as root, self.assertRaises(ProtocolError):
            run_supervised(t,rep,plan,sources=['A'],target='B',marker='W',candidates=[Candidate('linear')],seed=9,out=root/'bad',split_mode=mode)

    def test_j3_plans_equal_budgets(self):
        t=toy();p=make_j3_plans(t,'A','B','C',fit_quotas={'x':5,'y':5},validation_quotas={'x':3,'y':3},seed=8)
        self.assertEqual({x.total for x in p.values()},{32})

    def test_patient_budget_changes_supervision_not_fitting_U(self):
        t=toy();before=t.pool.sha256
        p,mode=make_plan(t,'A','B',fit_budget=20,validation_budget=20,seed=9,labelled_patient_count=2)
        lookup=t.metadata.set_index('cell_id').patient_id.to_dict()
        self.assertEqual(len({lookup[i] for i in p.fit+p.validation}),2)
        self.assertEqual(t.pool.sha256,before)

    def test_j4_four_readouts_and_matched_updates(self):
        t=toy();rep=prepare_representation(t.input('unaligned'),'unaligned',fitting_pool=t.pool)
        c=PretrainConfig(width=4,corpus_per_half=20,pretrain_validation_per_source=10,
            probe_fit=20,probe_validation=20,pretrain_updates=2,supervised_updates=2,validation_every=1)
        with temporary_artifacts() as root:
            r=run_j4(t,rep,source_a='A',source_b='B',target='C',cfg=c,seed=5,out=root/'j4')
            self.assertEqual(len(r['scores']),7) # input baseline plus three readouts per corpus
            a=r['training']['A-only_random_supervised'];b=r['training']['A-only_finetuned']
            self.assertEqual(a['trainable_parameters'],b['trainable_parameters'])
            self.assertEqual(a['executed_updates'],b['executed_updates'])
            self.assertEqual(r['total_probe_supervision'],40)

    def test_pilot_cannot_change_full_pool(self):
        t=toy();s=t.subset(np.arange(20),'pilot')
        self.assertNotEqual(t.pool.sha256,s.pool.sha256)
        with self.assertRaises(ProtocolError):t.pool.validate(s.input('ramp'))

if __name__=='__main__':unittest.main()
