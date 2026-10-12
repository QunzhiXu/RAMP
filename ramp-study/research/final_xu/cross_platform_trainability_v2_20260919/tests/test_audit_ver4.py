import json
import unittest
from unittest.mock import patch
import numpy as np
from trainability import heads
from trainability.core import ProtocolError,file_sha256
from trainability.adapters import fit_representation
from trainability.heads import Candidate,fit_head
from trainability.j4 import source_validation_diagnostics
from trainability.validation import SourceValidation
from test_integration import toy
from _artifacts import temporary_artifacts


class CacheIdentityTests(unittest.TestCase):
    def build(self,out):
        t=toy()
        return fit_representation(t.input('unaligned'),'unaligned',fitting_pool=t.pool,
            input_contract=t.receipt,family='synthetic',out=out)

    def test_valid_cache_is_reused_without_refitting(self):
        with temporary_artifacts() as root:
            first=self.build(root/'cache')
            with patch('trainability.adapters.simple_representation',side_effect=AssertionError('Unexpected fit')):
                cached=self.build(root/'cache')
            np.testing.assert_array_equal(first.x,cached.x)
            self.assertEqual(first.cell_ids,cached.cell_ids)

    def test_modified_cell_id_file_is_rejected(self):
        with temporary_artifacts() as root:
            self.build(root/'cache');p=root/'cache/cell_ids.npy'
            ids=np.load(p,allow_pickle=False);ids[[0,1]]=ids[[1,0]];np.save(p,ids)
            with self.assertRaisesRegex(ProtocolError,'cell-ID file changed'):
                self.build(root/'cache')

    def test_wrong_order_rejected_even_when_file_hash_matches_receipt(self):
        with temporary_artifacts() as root:
            self.build(root/'cache');p=root/'cache/cell_ids.npy'
            ids=np.load(p,allow_pickle=False);ids[[0,1]]=ids[[1,0]];np.save(p,ids)
            receipt=root/'cache/COMPLETE.json';data=json.loads(receipt.read_text())
            data['cell_ids_sha256']=file_sha256(p);receipt.write_text(json.dumps(data))
            with self.assertRaisesRegex(ProtocolError,'cell IDs/order differ'):
                self.build(root/'cache')


class OptimizationDiagnosticsTests(unittest.TestCase):
    def diagnose(self,scores,selected=None):
        history=[dict(updates=(i+1)*5,source_macro_f1=s) for i,s in enumerate(scores)]
        selected=selected if selected is not None else (int(np.argmax(scores))+1)*5
        return source_validation_diagnostics(history,selected,len(scores)*5,len(scores)*5)

    def test_last_step_improvement_not_called_nonconvergence(self):
        d=self.diagnose([.1,.2,.3])
        self.assertTrue(d['selected_at_update_budget'])
        self.assertTrue(d['last_interval_improved'])
        self.assertEqual(d['convergence_status'],'not_assessed_fixed_update_budget')

    def test_plateau_earlier_checkpoint_not_called_convergence(self):
        d=self.diagnose([.1,.3,.3])
        self.assertFalse(d['selected_at_final_update'])
        self.assertFalse(d['last_interval_improved'])
        self.assertTrue(d['update_budget_exhausted'])
        self.assertEqual(d['convergence_status'],'not_assessed_fixed_update_budget')

    def test_nonmonotone_curve_does_not_confuse_tail_improvement_with_optimum(self):
        d=self.diagnose([.4,.2,.3])
        self.assertFalse(d['selected_at_final_update'])
        self.assertTrue(d['last_interval_improved'])
        self.assertAlmostEqual(d['best_minus_last_source_validation_score'],.1)

    def test_single_check_has_unknown_trend(self):
        d=self.diagnose([.3])
        self.assertIsNone(d['last_interval_improved'])
        self.assertIsNone(d['last_interval_score_change'])

    def test_inconsistent_checkpoint_rejected(self):
        with self.assertRaises(ProtocolError):self.diagnose([.1,.3,.2],selected=15)

    def test_logistic_limit_and_warning_recorded(self):
        # Force real sklearn LBFGS to exhaust one iteration, exercising the
        # warning path rather than inserting fabricated diagnostic fields.
        rng=np.random.default_rng(12);x=rng.normal(size=(120,6))
        y=np.where(x[:,0]+.3*x[:,1]>0,'a','b');ids=[f'c{i}' for i in range(120)]
        vi=ids[80:];classes=('a','b')
        val=SourceValidation.from_cell_maps(cell_ids=vi,x=x[80:],
            targets_by_cell=dict(zip(vi,y[80:])),patients_by_cell={i:'p' for i in vi},
            task='classification',classes=classes,identity_level='patient',identity_reviewed=True)
        original=Candidate.effective
        def one_iteration(candidate,task,n_fit):
            e=original(candidate,task,n_fit);e['max_iter']=1;return e
        with patch.object(Candidate,'effective',one_iteration):
            h=fit_head(x[:80],y[:80],val,fit_ids=ids[:80],task='classification',
                candidates=[Candidate('linear')],classes=classes,seed=1)
        d=h.selection['optimization_diagnostics']
        self.assertTrue(d['iteration_limit_reached'])
        self.assertTrue(d['convergence_warning'])
        self.assertIn('ConvergenceWarning',h.selection['warning_types'])
        self.assertEqual(d['status'],'warning_or_iteration_limit')

if __name__=='__main__':unittest.main()
