"""Protocol checks on synthetic data only; no real benchmark fits."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from trainability.core import ROOT, ProtocolError, Q_GRID, output_path, representation_cache_key, validate_release
from trainability.sampling import (nested_draws, Supervision, source_split, reserve_calibration,
    matched_j3, validate_j3_supervision, validate_pretraining_sets)
from trainability.inputs import build_assay_input, simple_representation, assert_target_invariance
from trainability.targets import SourceCalibration, constant_mae_baseline
from trainability.scoring import classification_score, regression_score, freeze_predictions, load_frozen_predictions
from trainability.aggregation import hierarchical_mean, paired_difference, resource_crossings, count_crossing_states
from trainability.heads import Candidate, fit_head
from trainability.pretraining import make_mask, make_encoder, masked_mse


class SamplingTests(unittest.TestCase):
    def test_label_blind_exact_nested_and_order_invariant(self):
        ids = [f'c{i}' for i in range(40)]
        a = nested_draws(ids, [2, 5, 20], 71)
        self.assertEqual(a, nested_draws(ids[::-1], [2, 5, 20], 71))
        self.assertEqual(a[2], a[20][:2])
        self.assertEqual(len(a[5]), 5)

    def test_budget_does_not_silently_truncate_or_replicate(self):
        for ids, ns in [(['a'], [2]), (['a', 'a'], [1]), (['a'], [0])]:
            with self.assertRaises(ProtocolError): nested_draws(ids, ns, 0)

    def test_supervision_union_and_target_exclusion(self):
        plan = Supervision(('a', 'b'), ('c',), ('d',))
        self.assertEqual(plan.total, 4)
        with self.assertRaises(ProtocolError): Supervision(('a',), ('a',))
        with self.assertRaises(ProtocolError): plan.assert_allowed('abcd', ['d'])

    def test_patient_disjoint_split(self):
        ids = [str(i) for i in range(20)]
        patients = {i: str(int(i)//5) for i in ids}
        split = source_split(ids, patients, .3, 9, patient_mapping_verified=True)
        self.assertFalse({patients[i] for i in split['fit_pool']} & {patients[i] for i in split['validation_pool']})
        self.assertEqual(split['mode'], 'patient_disjoint_validation')
        with self.assertRaises(ProtocolError): source_split(ids, patients, .3, 9, patient_mapping_verified=False)

    def test_single_patient_disclosure(self):
        split = source_split(['a','b','c'], dict.fromkeys('abc','p'), .3, 1, patient_mapping_verified=True)
        self.assertEqual(split['mode'], 'within_patient_cell_validation')

    def test_calibration_reserved_by_unit(self):
        ids = list('abcdefgh'); units = dict(zip(ids, ['u']*4+['v']*4))
        split = reserve_calibration(ids, units, 2, 3)
        self.assertEqual(len(split['calibration']), 4)
        self.assertFalse(set(split['calibration']) & set(split['remaining']))
        with self.assertRaises(ProtocolError): reserve_calibration(ids, units, 4, 3)

    def test_j3_equal_fit_and_validation_with_nested_mixture(self):
        a = [f'a{i}' for i in range(24)]; b = [f'b{i}' for i in range(24)]
        labels = {i: str(int(i[1:]) % 2) for i in a+b}
        fit = matched_j3(a[:16], b[:16], labels, {'0':3, '1':3}, 9)
        val = matched_j3(a[16:], b[16:], labels, {'0':1, '1':1}, 9)
        plans = {k:Supervision(fit[k], val[k]) for k in fit}
        validate_j3_supervision(plans, labels, a, b, ['target'])
        self.assertEqual({p.total for p in plans.values()}, {16})
        with self.assertRaises(ProtocolError): matched_j3(a[:3], b, labels, {'0':3, '1':3}, 9)
        plans['A+B'] = Supervision(fit['A+B'][:-1], val['A+B'])
        with self.assertRaises(ProtocolError): validate_j3_supervision(plans, labels, a, b, ['target'])

    def test_pretraining_partitions_cannot_overlap_or_include_target(self):
        probe = Supervision(('p',), ('v',))
        validate_pretraining_sets(['a'], ['b'], probe, list('abpv'), ['t'])
        for corpus in [['p'], ['t']]:
            with self.assertRaises(ProtocolError): validate_pretraining_sets(corpus, ['b'], probe, list('abpvt'), ['t'])


class InputTests(unittest.TestCase):
    def setUp(self):
        self.raw = np.array([[0.,4.,10.,2.],[2.,8.,20.,5.],[9.,1.,30.,8.],[4.,9.,40.,2.]])
        self.kw = dict(cell_ids=('a','b','c','d'), datasets=('S','S','T','T'),
            units=('S|u','S|u','T|u','T|u'), markers=('A','B','A_alias','C'),
            canonical_marker={'A':'A','B':'B','A_alias':'A','C':'C'}, masked_marker='A',
            contracts={d:dict(review_status='verified_no_target_dependency', evidence_path='synthetic fixture',
                input_contract_version='toy1',operator='clr_shared') for d in ['S','T']})

    def test_mask_all_aliases_before_clr_and_every_simple_method(self):
        for method in ['unaligned','clipped_zscore','minmax','percentile']:
            builder = lambda raw: simple_representation(build_assay_input(raw, **self.kw), method)
            assert_target_invariance(builder, self.raw, [0,2])
        data = build_assay_input(self.raw, **self.kw)
        self.assertEqual(data.markers, ('B','C'))
        self.assertFalse(data.x.flags.writeable)

    def test_detects_pre_mask_leak(self):
        leaky = lambda raw: np.log1p(raw[:,1:2]/np.exp(np.log1p(raw).mean(axis=1,keepdims=True)))
        with self.assertRaises(ProtocolError): assert_target_invariance(leaky, self.raw, [0,2])

    def test_unreviewed_dependency_and_pending_adapter_stop(self):
        self.kw['contracts']['S']['review_status'] = 'pending'
        with self.assertRaises(ProtocolError): build_assay_input(self.raw, **self.kw)
        self.kw['contracts']['S']['review_status'] = 'verified_no_target_dependency'
        with self.assertRaises(ProtocolError): simple_representation(build_assay_input(self.raw, **self.kw), 'ramp')

    def test_constant_columns_and_signed_provider(self):
        for contract in self.kw['contracts'].values(): contract['operator'] = 'identity'
        raw = np.full((4,4), -2.)
        data = build_assay_input(raw, **self.kw)
        np.testing.assert_array_equal(simple_representation(data,'unaligned'), -2*np.ones((4,2)))
        for method in ['clipped_zscore','minmax']: np.testing.assert_array_equal(simple_representation(data,method), np.zeros((4,2)))
        np.testing.assert_array_equal(simple_representation(data,'percentile'), .5*np.ones((4,2)))

    def test_cache_includes_mask_order_pool_and_version(self):
        kw = dict(input_sha256='raw', cell_ids=['a','b'], markers=['A','B'], masked_markers=[],
            input_contract={'version':1}, method='ramp',parameters={},seed=1,
            code_sha256='code1',environment_sha256='env')
        full = representation_cache_key(**kw)
        for changed in [dict(masked_markers=['A']),dict(cell_ids=['b','a']),dict(code_sha256='code2')]:
            self.assertNotEqual(full,representation_cache_key(**(kw|changed)))


class TargetAndScoreTests(unittest.TestCase):
    def test_source_midcdf_ties_and_no_borrowing(self):
        cal = SourceCalibration.fit(list('abcd'), [0,0,2,2], ['u']*4, list('abcd'))
        np.testing.assert_allclose(cal.transform(list('efgh'), [0,1,2,3], ['u']*4), [.25,.5,.75,1])
        self.assertEqual(constant_mae_baseline([.1,.2,.9]), .2)
        with self.assertRaises(ProtocolError): cal.transform(['e'], [1], ['v'])
        with self.assertRaises(ProtocolError): cal.transform(['a'], [1], ['u'])
        with self.assertRaises(ProtocolError): SourceCalibration.fit(list('abc'), [0,0,2], ['u']*3, list('abcd'))

    def test_patient_weighting_not_cell_weighting(self):
        result = classification_score(['A']+['B']*9, ['A']*10, ['small']+['large']*9, ['A','B'])
        self.assertAlmostEqual(result['accuracy'], .5)
        self.assertAlmostEqual(result['macro_f1'], 1/3)

    def test_regression_units_then_patients(self):
        y = np.concatenate([np.arange(30),np.arange(300),np.arange(30)])
        pred = np.concatenate([np.arange(30)/30,-np.arange(300)/300,np.arange(30)/30])
        result = regression_score(y,pred,['a1']*30+['a2']*300+['b1']*30,['a']*330+['b']*30)
        self.assertAlmostEqual(result['effective_spearman'], .5)

    def test_constant_prediction_is_explicit_failure_convention(self):
        result = regression_score(np.arange(30),np.ones(30)*.4,['u']*30,['p']*30)
        self.assertEqual(result['effective_spearman'],0)
        self.assertIsNone(result['units'][0]['rho'])
        self.assertEqual(result['units'][0]['rho_status'],'constant_prediction')

    def test_constant_truth_small_support_and_nonfinite_are_distinct(self):
        result = regression_score(np.ones(30),np.arange(30),['u']*30,['p']*30)
        self.assertIsNone(result['effective_spearman'])
        self.assertEqual(result['units'][0]['rho_status'],'constant_truth')
        result = regression_score(np.arange(29),np.arange(29),['u']*29,['p']*29)
        self.assertEqual(result['units'][0]['rho_status'],'insufficient_truth_support')
        result = regression_score(np.arange(30),np.full(30,np.nan),['u']*30,['p']*30)
        self.assertEqual(result['status'],'run_failure')

    def test_rho_uses_unclipped_predictions(self):
        result = regression_score(np.arange(30),np.arange(30)+2.,['u']*30,['p']*30)
        self.assertAlmostEqual(result['effective_spearman'],1)
        self.assertAlmostEqual(result['mae'],.5)

    def test_unit_cannot_map_to_two_patients(self):
        with self.assertRaises(ProtocolError): regression_score([1,2],[.2,.8],['u','u'],['a','b'])

    def test_frozen_prediction_identity_hash_and_write_once(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as td:
            p = Path(td)/'pred.npz'
            receipt = freeze_predictions(p,['b','a'],[.2,.8],{'purpose':'synthetic'})
            np.testing.assert_allclose(load_frozen_predictions(receipt,['a','b']), [.8,.2])
            with self.assertRaises(ProtocolError): freeze_predictions(p,['a'],[.5],{})
            with self.assertRaises(ProtocolError): load_frozen_predictions(receipt,['a'])
            with p.open('ab') as f: f.write(b'changed')
            with self.assertRaises(ProtocolError): load_frozen_predictions(receipt,['a','b'])


class AggregationTests(unittest.TestCase):
    def test_equal_tasks_not_rows_and_order_invariance(self):
        df = pd.DataFrame({'method':['a']*4,'task':['small','big','big','big'],
                           'repeat':[0,0,1,2],'score':[.2,.8,.8,.8]})
        kw = dict(value='score',levels=[['method','task'],['method']],keys=['method','task','repeat'])
        result = hierarchical_mean(df,**kw)
        self.assertAlmostEqual(result.score.iloc[0],.5)
        pd.testing.assert_frame_equal(result,hierarchical_mean(df.iloc[::-1],**kw))
        with self.assertRaises(ProtocolError): hierarchical_mean(pd.concat([df,df.iloc[:1]]),**kw)

    def test_paired_coverage_mismatch_rejected(self):
        a = pd.DataFrame({'id':['a','b'],'score':[.1,.8]})
        with self.assertRaises(ProtocolError): paired_difference(a,a.iloc[:1],keys=['id'])
        np.testing.assert_allclose(paired_difference(a,a.iloc[::-1],keys=['id']).difference,0)

    def test_all_q_states_and_minimum_budget_count(self):
        rows = resource_crossings([2,5,10],[.2,.5,.4],q_grid=[.1,.3,.6])
        self.assertEqual([r['status'] for r in rows],['met_at_minimum_budget','met','not_reached'])
        self.assertIsNone(rows[0]['minimum_tested_resource'])
        self.assertEqual(rows[0]['resource_upper_bound'],2)
        self.assertEqual(rows[1]['minimum_tested_resource'],5)
        counts=count_crossing_states(rows)
        self.assertEqual(counts[0]['achieved'],1)
        self.assertEqual(counts[0]['met_at_minimum_budget'],1)
        self.assertEqual(len(Q_GRID),19)

    def test_missing_curve_not_silently_skipped(self):
        rows = resource_crossings([2,5],[None,.8])
        self.assertTrue(all(r['status']=='not_evaluable' for r in rows))
        with self.assertRaises(ProtocolError): resource_crossings([2,2],[.1,.2])

    def test_release_and_output_boundary(self):
        with self.assertRaises(ProtocolError): validate_release({'status':'draft'}, {})
        with self.assertRaises(ProtocolError): output_path(ROOT.parent/'benchmark_data'/'bad.json')


class HeadTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(1)
        self.x=rng.normal(size=(60,3))
        self.y=np.where(self.x[:,0]>0,'A','B')
        self.ids=[f'c{i}' for i in range(60)]

    def train(self,candidate, task='classification',y=None):
        y=self.y if y is None else y
        score=(lambda a,b:classification_score(a,b,['valpatient']*len(a),['A','B','C'])['macro_f1']) if task=='classification' else (lambda a,b:-np.abs(a-b).mean())
        return fit_head(self.x[:35],y[:35],self.x[35:50],y[35:50],
            fit_ids=self.ids[:35],validation_ids=self.ids[35:50],task=task,
            candidates=[candidate],source_validation_score=score,classes=['A','B','C'] if task=='classification' else (),seed=3)

    def test_linear_and_knn_classification_missing_class_probabilities(self):
        for family in ['linear','knn']:
            head=self.train(Candidate(family,scale='standard',neighbors=100))
            self.assertEqual(head.predict_proba(self.x[50:]).shape,(10,3))
            np.testing.assert_array_equal(head.predict_proba(self.x[50:])[:,2],0)
            if family=='knn':
                self.assertEqual(head.selection['candidate']['neighbors'],35)
                self.assertIsNone(head.selection['trainable_head_parameters'])

    def test_single_class_fallback_keeps_ontology(self):
        head=self.train(Candidate('linear'),y=np.full(60,'A'))
        self.assertTrue(head.selection['constant_classifier_fallback'])
        self.assertEqual(head.predict_proba(self.x[50:]).shape,(10,3))

    def test_mlp_external_validation_actual_params_and_fixed_seed(self):
        c=Candidate('mlp',width=4,max_epochs=3,patience=2,regularization=.0001)
        a,b=self.train(c),self.train(c)
        self.assertEqual(a.selection['trainable_head_parameters'],31)
        self.assertLessEqual(a.selection['epochs_executed'],3)
        np.testing.assert_allclose(a.predict_proba(self.x[50:]),b.predict_proba(self.x[50:]))

    def test_three_regression_heads(self):
        for family in ['linear','knn','mlp']:
            head=self.train(Candidate(family,width=4,max_epochs=2),task='regression',y=self.x[:,0])
            self.assertTrue(np.isfinite(head.predict(self.x[50:])).all())

    def test_source_scaler_does_not_collapse_target_variation(self):
        self.x[:35,2]=2
        head=self.train(Candidate('linear',scale='standard'))
        self.assertEqual(head.scaler.scale_[2],1)
        values=head.scaler.transform(np.array([[0,0,3.],[0,0,5.]]))[:,2]
        np.testing.assert_allclose(values,[1.,3.])

    def test_validation_failure_not_substituted_by_target(self):
        with self.assertRaises(ProtocolError):
            fit_head(self.x[:35],self.y[:35],self.x[35:50],self.y[35:50],
                fit_ids=self.ids[:35],validation_ids=self.ids[35:50],task='classification',
                candidates=[Candidate('linear')],source_validation_score=lambda a,b:None,classes=['A','B'])


class PretrainingTests(unittest.TestCase):
    def test_mask_flag_and_masked_loss_gradient(self):
        import torch
        torch.manual_seed(5)
        model=make_encoder(4,3)
        x=torch.arange(16,dtype=torch.float32).reshape(4,4)
        mask=torch.tensor(make_mask((4,4),.5,9))
        y=model(x,mask)
        changed=x.clone();changed[mask]=10000
        torch.testing.assert_close(y,model(changed,mask))
        loss=masked_mse(y,x,mask)
        target2=x.clone();target2[~mask]=9999
        torch.testing.assert_close(loss,masked_mse(y,target2,mask))
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        self.assertTrue((mask.sum(dim=1)==2).all())


if __name__ == '__main__':
    unittest.main()
