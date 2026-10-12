"""P2 regression tests: saved diagnostics are a contract, not optional metadata."""
import copy
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import replay as r
from common import code_records, record, write
from scores import classification, Regression


def j1():
    return classification(['a', 'b', 'a', 'b'], ['a', 'a', 'a', 'b'], ['p', 'p', 'q', 'q'], ['a', 'b'])


def j2():
    truth = np.concatenate([np.arange(40), np.ones(40), np.arange(10)])
    units = ['a']*40 + ['b']*40 + ['c']*10
    patients = ['p']*40 + ['q']*40 + ['q']*10
    scorer = Regression(truth, units, patients)
    result = scorer.score(np.linspace(0, 1, 90))
    result['source_constant_baseline'] = scorer.score(np.full(90, .4))
    return result


class DifferenceContract(unittest.TestCase):
    def test_boundary_accepted(self):
        for x in (0, 0.0, 3.33e-16, 2e-14):r.difference(x)

    def test_invalid_errors_rejected(self):
        for x in (1.0, 2.01e-14, -1e-20, float('nan'), float('inf'), -float('inf'), True, '0', None):
            with self.subTest(x=x), self.assertRaises(ValueError):r.difference(x)

    def test_exact_historical_controller(self):r.check_controller_records(copy.deepcopy(r.ACCEPTED_REPLAY_CODE))
    def test_current_controller(self):r.check_controller_records(code_records())

    def test_historical_hash_mutations(self):
        for name in r.ACCEPTED_REPLAY_CODE:
            for field, value in [('sha256', '0'*64), ('bytes', 0)]:
                records = copy.deepcopy(r.ACCEPTED_REPLAY_CODE); records[name][field] = value
                with self.subTest(name=name, field=field), self.assertRaises(ValueError):r.check_controller_records(records)

    def test_missing_controller(self):
        records = copy.deepcopy(r.ACCEPTED_REPLAY_CODE); del records['scores.py']
        with self.assertRaises(ValueError):r.check_controller_records(records)


class ClassificationDiagnostics(unittest.TestCase):
    def check(self, score):return r.classification_diagnostics(score, ['a', 'b'])
    def test_valid(self):self.assertEqual(self.check(j1())['cells'], 4)

    def test_every_field_required(self):
        for name in j1():
            score = j1(); del score[name]
            with self.subTest(name=name), self.assertRaises(ValueError):self.check(score)

    def test_extra_field(self):
        score = j1(); score['unexpected'] = 1
        with self.assertRaises(ValueError):self.check(score)

    def test_missing_patient_matrix(self):
        score = j1(); del score['patient_confusion_matrices'][0]['matrix']
        with self.assertRaises(ValueError):self.check(score)

    def test_duplicate_patients(self):
        score = j1(); score['patient_confusion_matrices'][1]['patient'] = 'p'
        with self.assertRaises(ValueError):self.check(score)

    def test_missing_patient(self):
        score = j1(); score['patient_confusion_matrices'].pop()
        with self.assertRaises(ValueError):self.check(score)

    def test_support_types(self):
        for name, value in [('cells', 0), ('cells', 1), ('cells', 4.0), ('patients', 3), ('patients', True), ('zero_division', False)]:
            score = j1(); score[name] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):self.check(score)

    def test_class_order(self):
        score = j1(); score['classes'].reverse()
        with self.assertRaises(ValueError):self.check(score)

    def test_matrix_shapes(self):
        score = j1(); score['confusion_matrix'][0].pop()
        with self.assertRaises(ValueError):self.check(score)

    def test_matrix_values(self):
        for value in (float('nan'), float('inf'), -0.1, True, '0.5'):
            score = j1(); score['patient_confusion_matrices'][0]['matrix'][0][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):self.check(score)

    def test_matrix_mass(self):
        score = j1(); score['confusion_matrix'][0][0] += .1
        with self.assertRaises(ValueError):self.check(score)

    def test_patient_mean(self):
        score = j1(); score['confusion_matrix'] = [[.5, 0.0], [0.0, .5]]
        with self.assertRaises(ValueError):self.check(score)

    def test_scalar_consistency(self):
        for name in ('accuracy', 'balanced_accuracy', 'macro_f1'):
            score = j1(); score[name] = .01
            with self.subTest(name=name), self.assertRaises(ValueError):self.check(score)


class RegressionDiagnostics(unittest.TestCase):
    def test_valid(self):self.assertEqual(len(r.regression_diagnostics(j2())), 3)

    def test_every_root_field_required(self):
        for name in j2():
            score = j2(); del score[name]
            with self.subTest(name=name), self.assertRaises(ValueError):r.regression_diagnostics(score)

    def test_every_unit_field_required(self):
        for name in j2()['units'][0]:
            score = j2(); del score['units'][0][name]
            with self.subTest(name=name), self.assertRaises(ValueError):r.regression_diagnostics(score)

    def test_baseline_diagnostic_required(self):
        score = j2(); del score['source_constant_baseline']['units']
        with self.assertRaises(ValueError):r.regression_diagnostics(score)

    def test_empty_units(self):
        score = j2(); score['units'] = []
        with self.assertRaises(ValueError):r.regression_diagnostics(score)

    def test_duplicate_units(self):
        score = j2(); score['units'][1]['unit'] = 'a'
        with self.assertRaises(ValueError):r.regression_diagnostics(score)

    def test_status_counts(self):
        score = j2(); score['rho_status_counts']['ok'] = 0
        with self.assertRaises(ValueError):r.regression_diagnostics(score)

    def test_status_count_bool(self):
        score = j2(); score['rho_status_counts']['ok'] = True
        with self.assertRaises(ValueError):r.regression_diagnostics(score)

    def test_patient_counts(self):
        for name in ('rho_patients', 'mae_patients'):
            score = j2(); score[name] = 0
            with self.subTest(name=name), self.assertRaises(ValueError):r.regression_diagnostics(score)

    def test_aggregates(self):
        for name in ('effective_spearman', 'mae', 'mae_rho_support'):
            score = j2(); score[name] = -.3
            with self.subTest(name=name), self.assertRaises(ValueError):r.regression_diagnostics(score)

    def test_nonfinite(self):
        for name in ('mae', 'rho', 'effective_rho'):
            score = j2(); score['units'][0][name] = float('nan')
            with self.subTest(name=name), self.assertRaises(ValueError):r.regression_diagnostics(score)

    def test_minimum_support(self):
        score = j2(); score['units'][0]['cells'] = 29
        with self.assertRaises(ValueError):r.regression_diagnostics(score)

    def test_unsupported_correlation(self):
        score = j2(); score['units'][1]['effective_rho'] = 0.0
        with self.assertRaises(ValueError):r.regression_diagnostics(score)

    def test_invalid_status(self):
        score = j2(); score['units'][0]['rho_status'] = 'unknown'
        with self.assertRaises(ValueError):r.regression_diagnostics(score)

    def test_constant_baseline_required(self):
        score = j2(); score['source_constant_baseline'] = {k:v for k,v in score.items() if k != 'source_constant_baseline'}
        with self.assertRaises(ValueError):r.regression_diagnostics(score)

    def test_baseline_support_mismatch(self):
        score = j2(); score['source_constant_baseline']['units'][0]['cells'] += 1
        with self.assertRaises(ValueError):r.regression_diagnostics(score)

    def test_null_rho_support_valid(self):
        scorer = Regression(np.ones(40), ['u']*40, ['p']*40)
        score = scorer.score(np.ones(40)); score['source_constant_baseline'] = scorer.score(np.ones(40))
        r.regression_diagnostics(score)


class BranchContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='prediction_verifier_')
        self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name).resolve()
        self.job = dict(task='example', task_type='J1', method='ramp', source='a', target='b',
                        family='knn', seed=42, id='job', supervision_sha256='a'*64)
        self.expected = dict(job=self.job, folder='prediction')
        self.catalog = dict(jobs=[self.expected], cohorts={'example': {'classes':['a','b']}},
                            files={'prediction/predictions.npz': {'sha256':'b'*64}})
        self.saved = dict(job=self.job, score=j1(), max_abs_difference=0., supervision_sha256='a'*64,
                          target_ids_sha256='c'*64, prediction_sha256='b'*64, ledger_sha256='d'*64)
        self.report = dict(case='example', task_type='J1', jobs=1, metric_rows=3, max_abs_difference=0.,
                           elapsed_seconds=1., environment={})

    def build(self, table_limit=None):
        write(self.root/'REQUEST.json', dict(case='example',task_type='J1',job_keys=[list(r.key(self.expected))],
              index_sha256='index',code_records=code_records(),environment={},pilot=False))
        with gzip.open(self.root/'scores.jsonl.gz','wt') as stream:stream.write(json.dumps(self.saved)+'\n')
        table=[]
        for metric in ('macro_f1','balanced_accuracy','accuracy'):
            table.append(dict({k:self.job.get(k,'') for k in ('task','task_type','method','marker','source','target','family','seed')},
                         direction='a -> b',job_id='job',metric=metric,score=self.saved['score'][metric]))
        pd.DataFrame(table[:table_limit])[list(r.METRIC_FIELDS)].to_csv(self.root/'metrics.csv',index=False)
        write(self.root/'report.json',self.report)
        write(self.root/'RUN_MANIFEST.json',{p.name:record(p) for p in self.root.iterdir()})
        write(self.root/'COMPLETE.json',dict(status='complete',manifest=record(self.root/'RUN_MANIFEST.json')))

    def verify(self):return r.verify_branch(self.root,self.catalog,'index',code_records(),'example','J1')
    def test_valid(self):self.build();self.assertEqual(self.verify()['metric_rows'],3)
    def test_metric_count(self):
        self.report['metric_rows']=0;self.build()
        with self.assertRaisesRegex(ValueError,'metric count'):self.verify()
    def test_metric_count_float(self):
        self.report['metric_rows']=3.0;self.build()
        with self.assertRaises(ValueError):self.verify()
    def test_missing_metric_row(self):
        self.build(table_limit=2)
        with self.assertRaises(AssertionError):self.verify()
    def test_record_error(self):
        self.saved['max_abs_difference']=1.0;self.report['max_abs_difference']=1.0;self.build()
        with self.assertRaisesRegex(ValueError,'max_abs_difference'):self.verify()
    def test_report_error(self):
        self.report['max_abs_difference']=float('inf')
        with self.assertRaises(ValueError):r.difference(self.report['max_abs_difference'])
    def test_diagnostic_removed(self):
        del self.saved['score']['cells'];self.build()
        with self.assertRaisesRegex(ValueError,'J1 score fields'):self.verify()
    def test_prediction_binding(self):
        self.saved['prediction_sha256']='e'*64;self.build()
        with self.assertRaisesRegex(ValueError,'prediction binding'):self.verify()
    def test_supervision_binding(self):
        self.saved['supervision_sha256']='e'*64;self.build()
        with self.assertRaisesRegex(ValueError,'supervision'):self.verify()
    def test_report_identity(self):
        self.report['case']='foreign';self.build()
        with self.assertRaisesRegex(ValueError,'identity'):self.verify()


if __name__ == '__main__':unittest.main()
