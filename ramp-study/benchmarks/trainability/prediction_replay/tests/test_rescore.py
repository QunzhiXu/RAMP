import copy
import ast
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import multiprocessing
import unittest
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from common import *
import binding
from scores import classification,Regression,compare_tree,Cohort,score_job,LUNG_MAPPING

def claim(path,source,barrier,queue):
    barrier.wait()
    try:output(path,source);queue.put('won')
    except (ValueError,FileExistsError):queue.put('refused')

class Arithmetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Only pure score/calibration functions from the byte-frozen phase-1 source.
        sys.path.insert(0,str(SNAPSHOT/'tree'/phase1.ROOT/'src'))
        from trainability.scoring import classification_score,regression_score
        cls.official_class=staticmethod(classification_score);cls.official_reg=staticmethod(regression_score)

    def test_classification_patient_weighting(self):
        y=['a']*99+['b'];p=['a']*100;patient=['x']*99+['y']
        x=classification(y,p,patient,['a','b']);self.assertEqual(x['accuracy'],.5)
        self.assertEqual(x,self.official_class(y,p,patient,['a','b'],include_sufficient_stats=True))

    def test_classification_missing_class(self):
        self.assertEqual(classification(['a']*3,['a']*3,['p']*3,['a','b'])['balanced_accuracy'],.5)

    def test_classification_ontology(self):
        with self.assertRaises(ValueError):classification(['a'],['c'],['p'],['a','b'])

    def test_classification_duplicates(self):
        with self.assertRaises(ValueError):classification(['a'],['a'],['p'],['a','a'])

    def test_classification_empty(self):
        with self.assertRaises(ValueError):classification([],[],[],['a'])

    def test_classification_shape(self):
        with self.assertRaises(ValueError):classification(['a'],['a','a'],['p'],['a'])

    def test_classification_missing_identity(self):
        with self.assertRaises(ValueError):classification(['a'],['a'],[''],['a'])

    def test_random_regression_equivalence(self):
        rng=np.random.default_rng(42)
        for seed in range(12):
            y=rng.integers(-5,9,150).astype(float);p=rng.normal(.5,.8,150)
            units=np.repeat(['a','b','c'],50);patients=np.repeat(['p','p','q'],50)
            compare_tree(Regression(y,units,patients).score(p),self.official_reg(y,p,units,patients))

    def test_regression_constant_truth(self):
        s=Regression(np.ones(50),['u']*50,['p']*50).score(np.arange(50));self.assertIsNone(s['effective_spearman']);self.assertEqual(s['rho_status_counts'],{'constant_truth':1})

    def test_regression_constant_prediction(self):
        s=Regression(np.arange(50),['u']*50,['p']*50).score(np.ones(50));self.assertEqual(s['effective_spearman'],0)

    def test_regression_insufficient(self):
        s=Regression(np.arange(29),['u']*29,['p']*29).score(np.arange(29));self.assertEqual(s['rho_status_counts'],{'insufficient_truth_support':1})

    def test_regression_threshold(self):
        s=Regression(np.arange(30),['u']*30,['p']*30).score(np.arange(30));self.assertEqual(s['rho_status_counts'],{'ok':1})

    def test_regression_nonfinite_prediction(self):
        s=Regression([0,1],['u','u'],['p','p']).score([0,np.nan]);self.assertEqual(s['status'],'run_failure')

    def test_regression_nonfinite_truth(self):
        with self.assertRaises(ValueError):Regression([0,np.nan],['u','u'],['p','p'])

    def test_regression_patient_crossing(self):
        with self.assertRaises(ValueError):Regression([0,1],['u','u'],['p','q'])

    def test_regression_no_unit(self):
        with self.assertRaises(ValueError):Regression([0,1],['',''],['p','p'])

    def test_regression_shape(self):
        with self.assertRaises(ValueError):Regression([0,1],['u'],['p','p'])

    def test_regression_prediction_shape(self):
        with self.assertRaises(ValueError):Regression([0,1],['u','u'],['p','p']).score([0])

    def test_compare_missing_field(self):
        with self.assertRaises(ValueError):compare_tree({'a':0.}, {})

    def test_compare_numeric_mutation(self):
        with self.assertRaises(ValueError):compare_tree({'a':0.}, {'a':.001})

    def test_compare_status_mutation(self):
        with self.assertRaises(ValueError):compare_tree({'a':'ok'}, {'a':'constant_truth'})

    def test_compare_support_mutation(self):
        with self.assertRaises(ValueError):compare_tree({'a':30}, {'a':31})

class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.plan=binding.routes()
    def test_full_grid(self):check_grid(self.plan['jobs'])
    def test_duplicate(self):
        with self.assertRaises(ValueError):check_grid(self.plan['jobs']+[self.plan['jobs'][0]])
    def test_missing(self):
        with self.assertRaises(ValueError):check_grid(self.plan['jobs'][1:])
    def test_swapped(self):
        x=copy.deepcopy(self.plan['jobs'][:1]);x[0]['job']['target']='wrong'
        with self.assertRaises(ValueError):check_grid(x,self.plan['jobs'][:1])
    def test_representation_total(self):self.assertEqual(len(self.plan['representations']),860)
    def test_lung_primary_only(self):
        r=[r for r in self.plan['jobs'] if r['job']['task']=='lung_story' and r['job']['task_type']=='J1'];self.assertEqual({x['job']['arm'] for x in r},{'keep_c02'})
    def test_hnscc_full_only(self):
        for r in self.plan['jobs']:
            if r['job']['task']=='hnscc_story' and r['job']['task_type']=='J1':self.assertIn('/jobs/full__',r['folder'])
    def test_current_lung_j2(self):
        for r in self.plan['jobs']:
            if r['job']['task']=='lung_story' and r['job']['task_type']=='J2':self.assertIn(phase1.J2,r['folder'])
    def test_lung_mapping_exact_frozen_literal(self):
        tree=ast.parse((SNAPSHOT/'tree'/phase1.ROOT/phase1.PILOT/'run.py').read_text())
        mappings=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MAPPING' for t in n.targets)]
        self.assertEqual(mappings,[LUNG_MAPPING])
    def test_source_protection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'source';src.mkdir()
            with self.assertRaises(ValueError):output(src/'nested'/'out',src)
            self.assertFalse((src/'nested').exists())
    def test_prior_module_protection(self):
        with self.assertRaises(ValueError):output(PARENT/'old/unsafe',PARENT.parents[2]/'final_xu')
        self.assertFalse((PARENT/'old').exists())
    def test_output_refuses_existing(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):output(Path(d),Path(d)/'source')
    def test_dangling_link(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'alias';p.symlink_to(Path(d)/'missing')
            with self.assertRaises(ValueError):output(p,Path(d)/'source')
    def test_bound_bytes_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a';p.write_bytes(b'a');r={'a':record(p)};p.write_bytes(b'b')
            with self.assertRaises(ValueError):verified_bytes(d,'a',r)
    def test_bound_path_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):verified_bytes(d,'../escape',{})
    def test_noncomplete_index(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'catalog.json';write(p,{})
            with self.assertRaises(FileNotFoundError):binding.load(p,record(p)['sha256'])

    def test_real_concurrent_claim(self):
        ctx=multiprocessing.get_context('spawn')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();barrier=ctx.Barrier(2);queue=ctx.Queue()
            workers=[ctx.Process(target=claim,args=(root/'output',root/'source',barrier,queue)) for _ in range(2)]
            for w in workers:w.start()
            answers=[queue.get(timeout=30) for _ in workers]
            for w in workers:w.join(30);self.assertEqual(w.exitcode,0)
            self.assertEqual(sorted(answers),['refused','won'])

class Ledger(unittest.TestCase):
    def setUp(self):
        c=Cohort.__new__(Cohort);c.spec=dict(markers=['CD19'],canonical_markers={'CD19':'CD19'},pool_sha256='pool',classes=['a']);c.case='synthetic'
        c.arrays=dict(cell_id=np.array([str(i) for i in range(40)]),dataset=np.array(['source']*10+['target']*30),
                      unit_key=np.array(['u1']*5+['u2']*5+['u3']*30),patient_id=np.array(['p1']*5+['p2']*5+['p3']*30),
                      celltype=np.array(['a']*40),identity_level=np.array(['patient']*40))
        c.x=np.arange(40,dtype=float).reshape(-1,1);c.index={x:i for i,x in enumerate(c.arrays['cell_id'])};c.targets={};c.baselines={};self.c=c
        l=dict(sources=['source'],target='target',fit=['0','1'],validation=['5','6'],calibration=['2','7'],marker='CD19',seed=42,
               split_mode='patient_disjoint_validation',identity_level='patient',pool_sha256='pool',candidates=[{'family':'linear'}],mask=['CD19'],total_supervision=6,
               target_cell_ids=[str(i) for i in range(10,40)],supervised_identity_count=2)
        j=dict(source='source',target='target',marker='CD19',seed=42,split_mode=l['split_mode'],fit_budget=2,validation_budget=2,total_supervision=6,calibration_per_unit=1,
               supervision_sha256=digest({k:l[k] for k in ('fit','validation','calibration')}))
        self.ledger=l;self.row=dict(job=j,candidates_sha256=digest(l['candidates']))
    def test_valid_ledger(self):self.assertEqual(len(self.c.check_ledger(self.row,self.ledger,None)[0]),30)
    def test_wrong_direction(self):
        self.ledger['target']='source'
        with self.assertRaises(ValueError):self.c.check_ledger(self.row,self.ledger,None)
    def test_wrong_mask(self):
        self.ledger['mask']=[]
        with self.assertRaises(ValueError):self.c.check_ledger(self.row,self.ledger,None)
    def test_wrong_candidates(self):
        self.ledger['candidates']=[{'family':'knn'}]
        with self.assertRaises(ValueError):self.c.check_ledger(self.row,self.ledger,None)
    def test_wrong_pool(self):
        self.ledger['pool_sha256']='other'
        with self.assertRaises(ValueError):self.c.check_ledger(self.row,self.ledger,None)
    def test_target_reordered(self):
        self.ledger['target_cell_ids'].reverse()
        with self.assertRaises(ValueError):self.c.check_ledger(self.row,self.ledger,None)
    def test_source_overlap(self):
        self.ledger['validation']=['0','6'];self.row['job']['supervision_sha256']=digest({k:self.ledger[k] for k in ('fit','validation','calibration')})
        with self.assertRaises(ValueError):self.c.check_ledger(self.row,self.ledger,None)
    def test_target_leak(self):
        self.ledger['fit']=['10','11'];self.row['job']['supervision_sha256']=digest({k:self.ledger[k] for k in ('fit','validation','calibration')})
        with self.assertRaises(ValueError):self.c.check_ledger(self.row,self.ledger,None)
    def test_fit_validation_patient_overlap(self):
        self.c.arrays['patient_id'][5:10]='p1';self.ledger['supervised_identity_count']=1
        with self.assertRaises(ValueError):self.c.check_ledger(self.row,self.ledger,None)
    def test_calibration_budget(self):
        self.row['job']['calibration_per_unit']=2
        with self.assertRaises(ValueError):self.c.check_ledger(self.row,self.ledger,None)
    def test_plan_changed(self):
        with self.assertRaises(ValueError):self.c.check_ledger(self.row,self.ledger,dict(fit=['1','0'],validation=['5','6'],calibration=['2','7']))
    def test_source_constant_baseline(self):self.assertEqual(self.c.baseline(self.ledger),0.0)
    def test_source_calibration_ties(self):
        self.c.x[2,0]=0.0
        self.assertEqual(self.c.baseline(self.ledger),0.75)
    def test_source_baseline_ignores_target(self):
        self.c.x[10:,0]=1e9
        self.assertEqual(self.c.baseline(self.ledger),0.0)

if __name__=='__main__':unittest.main()
