import copy
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import contract as c
import study
from compare import normalize,tree_differences

def compete(path,barrier,queue):
    barrier.wait()
    try:c.reserve(path);queue.put(True)
    except (ValueError,FileExistsError):queue.put(False)

class Safety(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
    def tearDown(self):self.tmp.cleanup()
    def test_reserve_once(self):
        p=self.root/'output';c.reserve(p)
        with self.assertRaises(ValueError):c.reserve(p)
    def test_reserve_race(self):
        ctx=mp.get_context('spawn');barrier=ctx.Barrier(2);q=ctx.Queue()
        ps=[ctx.Process(target=compete,args=(str(self.root/'race'),barrier,q)) for _ in range(2)]
        for p in ps:p.start()
        for p in ps:p.join(20);self.assertEqual(p.exitcode,0)
        self.assertEqual(sorted([q.get(timeout=2),q.get(timeout=2)]),[False,True])
    def test_dangling_destination(self):
        p=self.root/'alias';p.symlink_to(self.root/'missing')
        with self.assertRaises(ValueError):c.reserve(p)
    def test_protected_no_parent_created(self):
        p=self.root/'source'/'new'/'output'
        with self.assertRaises(ValueError):c.reserve(p,[self.root/'source'])
        self.assertFalse((self.root/'source').exists())
    def test_protected_alias(self):
        real=self.root/'source';real.mkdir();(self.root/'alias').symlink_to(real)
        with self.assertRaises(ValueError):c.reserve(self.root/'alias'/'output',[real])
    def test_member_traversal(self):
        for n in ('../a','/a','.'):
            with self.subTest(n=n),self.assertRaises(ValueError):c.member(self.root,n)
    def test_member_symlink(self):
        (self.root/'a').symlink_to(self.root/'b')
        with self.assertRaises(ValueError):c.member(self.root,'a')
    def test_copy_exclusive(self):
        a=self.root/'a';a.write_text('old');b=self.root/'b';b.write_text('keep')
        with self.assertRaises(FileExistsError):c.copy(a,b,c.record(a))
        self.assertEqual(b.read_text(),'keep')
    def test_copy_bad_source(self):
        a=self.root/'a';a.write_text('old');r=c.record(a);a.write_text('new')
        with self.assertRaises(ValueError):c.copy(a,self.root/'b',r)
        self.assertFalse((self.root/'b').exists())
    def test_seal_roundtrip(self):
        (self.root/'data').write_text('data');c.seal(self.root,dict(kind='test'));self.assertEqual(c.verify_seal(self.root)['details'],dict(kind='test'))
    def test_seal_extra(self):
        c.seal(self.root,{});(self.root/'extra').write_text('x')
        with self.assertRaises(ValueError):c.verify_seal(self.root)
    def test_failed_cannot_seal(self):
        c.write(self.root/'FAILED.json',{})
        with self.assertRaises(ValueError):c.seal(self.root,{})
        self.assertFalse((self.root/'COMPLETE.json').exists())
    def test_changed_payload(self):
        a=self.root/'a';a.write_text('old');c.seal(self.root,{});a.write_text('new')
        with self.assertRaises(ValueError):c.verify_seal(self.root)
    def test_incomplete(self):
        with self.assertRaises(FileNotFoundError):c.verify_seal(self.root)

class Contracts(unittest.TestCase):
    def test_locked_plan(self):self.assertEqual(len(c.load_plan()['tasks']),45)
    def test_coverage(self):
        rows=[t['row']['job'] for t in c.load_plan()['tasks'].values()]
        self.assertEqual(len({r['task'] for r in rows}),7);self.assertEqual(len({r['method'] for r in rows}),10)
        self.assertEqual(len({(r['task_type'],r['task'],r['family']) for r in rows}),39)
        self.assertEqual({r['seed'] for r in rows},{20260919,20260920,20260921})
        self.assertEqual(sum(r['task']=='tnbc' and r['source']=='Tietscher IMC' for r in rows),6)
    def test_no_historical_outputs_in_bundle(self):
        for n,v in c.load_plan()['training_files'].items():
            self.assertNotIn(Path(v['source']).name,('RESULT.json','predictions.npz','predictions.json','model.pkl'))
    def test_source_functions_byte_pinned(self):
        plan=c.load_plan()
        for n,v in plan['training_files'].items():
            if n.startswith('code/'):self.assertEqual(c.record(c.SOURCE/v['source']),v['record'])
    def test_duplicate_grid(self):
        with self.assertRaises(ValueError):c.check_grid(['a','a'],['a','b'])
    def test_missing_grid(self):
        with self.assertRaises(ValueError):c.check_grid(['a'],['a','b'])
    def test_extra_grid(self):
        with self.assertRaises(ValueError):c.check_grid(['a','b','c'],['a','b'])
    def test_wrong_identity_grid(self):
        with self.assertRaises(ValueError):c.check_grid(['a','c'],['a','b'])
    def test_correct_grid(self):c.check_grid(['b','a'],['a','b'])
    def test_single_thread_child(self):self.assertTrue(all(c.child_environment()[n]=='1' for n in c.THREADS))
    def test_bad_runtime_rejected(self):
        with patch.dict(os.environ,{'OMP_NUM_THREADS':'2'}),self.assertRaises(ValueError):c.environment()
    def test_result_diff_not_hidden(self):self.assertEqual(len(tree_differences({'score':.5},{'score':.6})),1)
    def test_selection_diff_not_hidden(self):self.assertEqual(len(tree_differences({'selected_epoch':5},{'selected_epoch':6})),1)
    def test_nan_rejected(self):
        with self.assertRaises(ValueError):tree_differences(float('nan'),0.)
    def test_schema_not_hidden(self):self.assertTrue(tree_differences({'a':1},{}))
    def test_time_whitelist(self):
        r=dict(elapsed_seconds=1.,training_seconds_including_selection=1.,inference_seconds=0.,selection=dict(seconds=.1,score=.5),candidates=[dict(seconds=.2)],prediction_receipt=dict(prediction_file='/a',sha256='abc',provenance=dict(selection=dict(seconds=.1,score=.5))))
        b=copy.deepcopy(r);b['selection']['seconds']=.8;self.assertEqual(normalize(r),normalize(b))
        b['selection']['score']=.7;self.assertTrue(tree_differences(normalize(r),normalize(b)))
    def test_invalid_time(self):
        with self.assertRaises(ValueError):normalize(dict(elapsed_seconds=-1))

if __name__=='__main__':unittest.main()
