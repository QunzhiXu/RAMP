"""Tiny synthetic integration checks: split -> head -> freeze -> score."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from trainability.core import ROOT
from trainability.inputs import build_assay_input, simple_representation
from trainability.sampling import Supervision
from trainability.targets import SourceCalibration
from trainability.heads import Candidate, fit_head
from trainability.scoring import (freeze_predictions, load_frozen_predictions,
                                 classification_score, regression_score)


class WorkflowTests(unittest.TestCase):
    def test_j1_freeze_before_target_scoring(self):
        rng = np.random.default_rng(82)
        x = rng.normal(size=(100, 3))
        source_y = np.where(x[:80, 0] > 0, 'T', 'B')
        ids = [f'cell{i}' for i in range(100)]
        ledger = Supervision(tuple(ids[:50]), tuple(ids[50:80]))
        ledger.assert_allowed(ids[:80], ids[80:])
        head = fit_head(x[:50], source_y[:50], x[50:80], source_y[50:80],
            fit_ids=ledger.fit, validation_ids=ledger.validation, task='classification',
            candidates=[Candidate('linear'), Candidate('knn', neighbors=5)], classes=['B','T'],
            source_validation_score=lambda a,b:classification_score(a,b,['source_val']*30,['B','T'])['macro_f1'])
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as td:
            receipt = freeze_predictions(Path(td)/'j1.npz', ids[80:], head.predict(x[80:]),
                {'purpose':'synthetic integration check', 'supervised_cells':ledger.total})
            # Target answers become available here, after prediction export.
            target_y = np.where(x[80:, 0] > 0, 'T', 'B')
            score = classification_score(target_y, load_frozen_predictions(receipt, ids[80:]),
                                         ['target']*20, ['B','T'])
        self.assertEqual(score['cells'], 20)
        self.assertTrue(0 <= score['macro_f1'] <= 1)

    def test_j2_calibration_mask_fit_freeze_and_score(self):
        rng = np.random.default_rng(16)
        raw = rng.uniform(0, 10, (200, 3))
        raw[:,0] = 2*raw[:,1]+raw[:,2]+rng.normal(0,.1,200)
        ids = [f'c{i}' for i in range(200)]
        ds = ['source']*140+['target']*60
        units = ['source|fit']*80+['source|val']*60+['target|1']*30+['target|2']*30
        contract = {d:dict(review_status='verified_no_target_dependency',
            evidence_path='synthetic independent input',input_contract_version='toy1',operator='identity') for d in set(ds)}
        data = build_assay_input(raw,cell_ids=ids,datasets=ds,units=units,markers=['A','B','C'],
            canonical_marker=dict(zip(['A','B','C'],['A','B','C'])),masked_marker='A',contracts=contract)
        x = simple_representation(data, 'unaligned')
        cal_idx = np.r_[0:10,80:90]
        fit_idx, val_idx = np.arange(10,80), np.arange(90,140)
        ledger = Supervision(tuple(ids[i] for i in fit_idx),tuple(ids[i] for i in val_idx),
                             tuple(ids[i] for i in cal_idx))
        ledger.assert_allowed(ids[:140],ids[140:])
        self.assertEqual(ledger.total, 140)
        cal = SourceCalibration.fit(ledger.calibration,raw[cal_idx,0],np.asarray(units)[cal_idx],ledger.calibration)
        fit_y = cal.transform(ledger.fit,raw[fit_idx,0],np.asarray(units)[fit_idx])
        val_y = cal.transform(ledger.validation,raw[val_idx,0],np.asarray(units)[val_idx])
        # Only source raw validation A is used for source validation correlation.
        source_score = lambda y,p:regression_score(raw[val_idx,0],p,np.asarray(units)[val_idx],['source_val']*50)['effective_spearman']
        head = fit_head(x[fit_idx],fit_y,x[val_idx],val_y,fit_ids=ledger.fit,
            validation_ids=ledger.validation,task='regression',
            candidates=[Candidate('linear'),Candidate('knn',neighbors=5)],source_validation_score=source_score)
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as td:
            receipt = freeze_predictions(Path(td)/'j2.npz',ids[140:],head.predict(x[140:]),
                {'purpose':'synthetic integration check','masked_marker':'A','supervised_cells':140})
            score = regression_score(raw[140:,0],load_frozen_predictions(receipt,ids[140:]),
                                     units[140:],['patient1']*30+['patient2']*30)
        self.assertEqual(score['status'],'ok')
        self.assertEqual(score['rho_patients'],2)
        self.assertEqual(x.shape[1],2)


if __name__ == '__main__':
    unittest.main()
