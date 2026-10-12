import json
import unittest
from dataclasses import asdict
from unittest.mock import patch
import numpy as np
from trainability.core import ProtocolError,file_sha256,Q_GRID,RELEASE_CHECKS
from trainability.inputs import prepare_representation
from trainability.j4 import run_j4,PretrainConfig
from trainability.heads import Candidate
from trainability.scoring import classification_score,regression_score
from trainability.reporting import PatientBootstrap,nested_average,paired_summary,summarize_suite,patient_interval_allowed
from trainability.experiment import execute
from test_integration import toy
from _artifacts import temporary_artifacts


class FormalReportingTests(unittest.TestCase):
    def test_global_patient_weights_shared_and_order_invariant(self):
        b=PatientBootstrap(40,5)
        a=b.weights(['cohort::1','cohort::2']);c=b.weights(['cohort::2','cohort::1'])
        np.testing.assert_array_equal(a[:,0],c[:,1])
        np.testing.assert_array_equal(a,PatientBootstrap(40,5).weights(['cohort::1','cohort::2']))

    def test_bootstrap_duplicate_ids_rejected(self):
        with self.assertRaises(ProtocolError):PatientBootstrap().weights(['p','p'])

    def test_classification_stats_preserve_equal_patient_point_estimate(self):
        s=classification_score(['a','a','b','b'],['a','b','b','b'],['p','p','q','q'],['a','b'],include_sufficient_stats=True)
        matrices=np.array([r['matrix'] for r in s['patient_confusion_matrices']])
        np.testing.assert_array_equal(matrices.mean(axis=0),s['confusion_matrix'])
        draws=PatientBootstrap(50).classification(s)
        self.assertTrue(all(np.isfinite(x).all() for x in draws.values()))

    def test_roi_then_patient_regression_weights(self):
        s=dict(units=[dict(patient='p',effective_rho=.2,mae=.1),
                      dict(patient='p',effective_rho=.6,mae=.3),dict(patient='q',effective_rho=.8,mae=.4)])
        b=PatientBootstrap(40);w=b.weights(['p','q']);r=b.regression(s)
        np.testing.assert_allclose(r['effective_spearman'],w@np.array([.4,.8])/w.sum(axis=1))

    def test_equal_task_weight_not_pooled_row_weight(self):
        rows=[dict(task='a',direction='d',marker='',seed=s,score=0.) for s in range(4)]
        rows.append(dict(task='b',direction='e',marker='',seed=0,score=1.))
        self.assertEqual(nested_average(rows),.5)
        self.assertEqual(nested_average(list(reversed(rows))),.5)

    def test_paired_shared_draws_cancel_exactly(self):
        a=[dict(task='a',direction='d',marker='',seed=1,score=.5,bootstrap=np.linspace(.2,.8,50))]
        result=paired_summary(a,a)
        self.assertEqual(result['difference'],0.)
        self.assertEqual(result['interval95'],[0.,0.])
        with self.assertRaises(ProtocolError):paired_summary(a,[])

    def test_unit_descriptive_has_no_patient_interval(self):
        a=[dict(task='a',direction='d',marker='',seed=1,score=.5,bootstrap=None)]
        self.assertIsNone(paired_summary(a,a)['interval95'])

    def test_single_patient_not_given_false_zero_width_patient_interval(self):
        self.assertFalse(patient_interval_allowed(dict(macro_f1=.8,patients=1),'patient','macro_f1'))
        self.assertTrue(patient_interval_allowed(dict(macro_f1=.8,patients=2),'patient','macro_f1'))
        self.assertFalse(patient_interval_allowed(dict(rho_patients=1,mae_patients=4),'patient','effective_spearman'))
        self.assertTrue(patient_interval_allowed(dict(rho_patients=1,mae_patients=4),'patient','mae'))

    def test_formal_run_rejects_changed_code_before_fitting(self):
        with temporary_artifacts() as root:
            evidence=root/'evidence.json';evidence.write_text('{}')
            config=root/'config.json';release=root/'release.json'
            cfg=dict(mode='formal',status='frozen',formal_run_enabled=True,q_grid=Q_GRID,jobs=[],
                output=str(root/'run'),release_evidence=str(release))
            config.write_text(json.dumps(cfg))
            checks={k:dict(status='passed',evidence_path=str(evidence),sha256=file_sha256(evidence)) for k in RELEASE_CHECKS}
            checks['configuration_frozen'].update(configuration_sha256=file_sha256(config),code_sha256='wrong')
            release.write_text(json.dumps(checks))
            with self.assertRaisesRegex(ProtocolError,'source code differs'):
                execute(config,pilot=False)

    def test_source_only_j4_never_freezes_or_scores_target(self):
        t=toy();r=prepare_representation(t.input('unaligned'),'unaligned',fitting_pool=t.pool)
        cfg=PretrainConfig(width=4,probe_fit=20,probe_validation=20,corpus_per_half=20,
            pretrain_validation_per_source=10,pretrain_updates=2,supervised_updates=2,validation_every=1)
        with temporary_artifacts() as root, \
                patch('trainability.j4.classification_score',side_effect=AssertionError('Target score accessed')), \
                patch('trainability.j4.freeze_predictions',side_effect=AssertionError('Target prediction computed')):
            result=run_j4(t,r,source_a='A',source_b='B',target='C',cfg=cfg,seed=3,out=root/'pilot',source_only=True)
            self.assertFalse(result['target_scores_computed'])
            self.assertFalse(list((root/'pilot').glob('*.npz')))

    def test_dispatch_to_aggregation_and_tamper_rejection(self):
        # Real trainers on synthetic values; only the data/adapter access is mocked.
        t=toy()
        with temporary_artifacts() as root:
            shards=[]
            for method in ['unaligned','ramp']:
                jobs=[]
                for marker in [None,'W']:
                    for seed in [8,9]:
                        j=dict(id=f'{marker}_{seed}',task='toy',method=method,task_type='J1' if marker is None else 'J2',
                            source='A',target='C',seed=seed,fit_budget=80,validation_budget=60,calibration_per_unit=10,
                            candidates=[asdict(Candidate('linear'))],family='linear',width=32,analysis_scopes=['primary'])
                        if marker:j['marker']=marker
                        jobs.append(j)
                cfg=dict(name='test',mode='pilot',seed=8,pilot_cells_per_dataset=360,jobs=jobs,output=str(root/method))
                path=root/(method+'.json');path.write_text(json.dumps(cfg))
                def adapter(data,method,*,fitting_pool,**kw):
                    return prepare_representation(data,'unaligned',fitting_pool=fitting_pool)
                with patch('trainability.experiment.load_task',return_value=t),patch('trainability.experiment.fit_representation',side_effect=adapter):
                    result=execute(path,pilot=True)
                self.assertTrue(all(r['status']=='passed' for r in result))
                shards.append(dict(config=str(path),sha256=file_sha256(path)))
            profile=root/'profile.json';profile.write_text(json.dumps(dict(methods=['unaligned','ramp'],seeds=[8,9],bootstrap=dict(draws=20,seed=5))))
            suite=root/'suite.json';suite.write_text(json.dumps(dict(profile=str(profile),profile_sha256=file_sha256(profile),shards=shards)))
            output=summarize_suite(suite,destination=root/'summary')
            self.assertEqual(json.loads((output/'REPORT.json').read_text())['status'],'complete')
            comparisons=json.loads((output/'paired_comparisons.json').read_text())
            self.assertTrue(comparisons)
            # Input contracts differ (native RAMP vs assay-ready unaligned),
            # even though this synthetic adapter skips external alignment.
            self.assertTrue(all(np.isfinite(r['difference']) for r in comparisons))
            self.assertTrue(all(r['interval95'][0]<=r['interval95'][1] for r in comparisons))
            # Removing a completion receipt must suppress aggregate output, not shrink support silently.
            (root/'ramp/jobs/None_8/JOB_COMPLETE.json').unlink()
            output=summarize_suite(suite,destination=root/'partial')
            self.assertEqual(json.loads((output/'REPORT.json').read_text())['status'],'incomplete_no_aggregate_claim')
            self.assertFalse((output/'primary_means.json').exists())
            (root/'unaligned/jobs/None_8/RESULT.json').write_text('{}')
            with self.assertRaises(ProtocolError):summarize_suite(suite,destination=root/'tampered')


if __name__=='__main__':unittest.main()
