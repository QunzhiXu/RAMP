"""Actual-scorer regression for a perfect classification accuracy one ULP above 1."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import replay
from common import SNAPSHOT,phase1,record,write
from scores import classification

CLASSES=['a','b','c']
TRUTH=['a']*6+['b']*23+['c']


def perfect_score():
    return classification(TRUTH,TRUTH,['p']*30,CLASSES)


def complete_branch(root):
    """Create one synthetic hashed branch; never touch an existing directory."""
    root.mkdir(exist_ok=False)
    job=dict(task='example',task_type='J1',method='ramp',source='source',target='target',
             family='knn',seed=42,id='job',supervision_sha256='a'*64)
    expected=dict(job=job,folder='prediction')
    catalog=dict(jobs=[expected],cohorts={'example':dict(classes=CLASSES)},
                 files={'prediction/predictions.npz':dict(sha256='b'*64)})
    score=perfect_score()
    saved=dict(job=job,score=score,max_abs_difference=0.,supervision_sha256='a'*64,
               target_ids_sha256='c'*64,prediction_sha256='b'*64,ledger_sha256='d'*64)
    write(root/'REQUEST.json',dict(case='example',task_type='J1',job_keys=[list(replay.key(expected))],
          index_sha256='synthetic',code_records=replay.ACCEPTED_REPLAY_CODE,environment={},pilot=False))
    with gzip.open(root/'scores.jsonl.gz','wt') as stream:stream.write(json.dumps(saved,allow_nan=False)+'\n')
    rows=[]
    for metric in ('macro_f1','balanced_accuracy','accuracy'):
        rows.append(dict({k:job.get(k,'') for k in ('task','task_type','method','marker','source','target','family','seed')},
                         direction='source -> target',job_id='job',metric=metric,score=score[metric]))
    pd.DataFrame(rows)[list(replay.METRIC_FIELDS)].to_csv(root/'metrics.csv',index=False)
    write(root/'report.json',dict(case='example',task_type='J1',jobs=1,metric_rows=3,
          max_abs_difference=0.,elapsed_seconds=1.,environment={}))
    write(root/'RUN_MANIFEST.json',{p.name:record(p) for p in root.iterdir()})
    write(root/'COMPLETE.json',dict(status='complete',manifest=record(root/'RUN_MANIFEST.json')))
    return catalog


def probe(controller,root):
    """Subprocess probe against the real archived/current verification function."""
    spec=importlib.util.spec_from_file_location('boundary_probe_controller',controller)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    catalog=complete_branch(root)
    return module.verify_branch(root,catalog,'synthetic',replay.ACCEPTED_REPLAY_CODE,'example','J1')


class ProbabilityBoundary(unittest.TestCase):
    def test_actual_scorer_edge(self):
        score=perfect_score()
        self.assertEqual(score['accuracy'],1.0000000000000002)
        replay.classification_diagnostics(score,CLASSES)

    def test_original_scorer_equal(self):
        sys.path.insert(0,str(SNAPSHOT/'tree'/phase1.ROOT/'src'))
        from trainability.scoring import classification_score
        original=classification_score(TRUTH,TRUTH,['p']*30,CLASSES,include_sufficient_stats=True)
        self.assertEqual(perfect_score(),original)
        replay.classification_diagnostics(original,CLASSES)

    def test_no_clipping_or_mutation(self):
        score=perfect_score();before=copy.deepcopy(score)
        replay.classification_diagnostics(score,CLASSES)
        self.assertEqual(score,before);self.assertGreater(score['accuracy'],1.)

    def test_complete_hashed_branch(self):
        with tempfile.TemporaryDirectory(prefix='probability_branch_') as tmp:
            root=Path(tmp).resolve()/'branch';catalog=complete_branch(root)
            before={p.name:record(p) for p in root.iterdir()}
            result=replay.verify_branch(root,catalog,'synthetic',replay.ACCEPTED_REPLAY_CODE,'example','J1')
            self.assertEqual(result['max_abs_difference'],0.)
            self.assertEqual(result['metric_rows'],3)
            self.assertEqual(before,{p.name:record(p) for p in root.iterdir()})

    def test_upper_tolerance_accepted(self):
        score=perfect_score();score['accuracy']=1+replay.SCORE_TOLERANCE
        replay.classification_diagnostics(score,CLASSES)

    def test_lower_tolerance_accepted(self):
        score=classification(TRUTH,['b']*6+['c']*23+['a'],['p']*30,CLASSES)
        for name in ('macro_f1','balanced_accuracy','accuracy'):score[name]=-replay.SCORE_TOLERANCE
        replay.classification_diagnostics(score,CLASSES)

    def test_upper_outside_tolerance_rejected(self):
        for value in (1+1e-10,float(np.nextafter(1+replay.SCORE_TOLERANCE,np.inf))):
            score=perfect_score();score['accuracy']=value
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'outside allowed range'):
                replay.classification_diagnostics(score,CLASSES)

    def test_lower_outside_tolerance_rejected(self):
        score=perfect_score();score['accuracy']=float(np.nextafter(-replay.SCORE_TOLERANCE,-np.inf))
        with self.assertRaisesRegex(ValueError,'outside allowed range'):replay.classification_diagnostics(score,CLASSES)

    def test_nonfinite_or_nonnumeric_still_rejected(self):
        for value in (float('nan'),float('inf'),-float('inf'),True,'1',None):
            score=perfect_score();score['accuracy']=value
            with self.subTest(value=value),self.assertRaises(ValueError):replay.classification_diagnostics(score,CLASSES)

    def test_diagnostic_consistency_retained(self):
        score=perfect_score();score['accuracy']=.9
        with self.assertRaisesRegex(ValueError,'inconsistent with diagnostics'):replay.classification_diagnostics(score,CLASSES)

    def test_error_threshold_not_expanded(self):
        self.assertEqual(replay.SCORE_TOLERANCE,2e-14)
        replay.difference(2e-14)
        with self.assertRaises(ValueError):replay.difference(float(np.nextafter(2e-14,np.inf)))

    def test_error_nonnegative_stays_strict(self):
        with self.assertRaises(ValueError):replay.difference(-1e-30)

    def test_confusion_entries_stay_strict(self):
        with self.assertRaises(ValueError):replay.confusion([[1.0000000000000002]],1)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='probe':
        print(json.dumps(probe(Path(sys.argv[2]),Path(sys.argv[3])),sort_keys=True))
    else:unittest.main()
