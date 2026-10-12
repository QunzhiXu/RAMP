import copy as copying
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import full_common as c
import full_compare

def race(path,barrier,queue):
    barrier.wait()
    try:c.reserve(path);queue.put(True)
    except (ValueError,FileExistsError):queue.put(False)

class FullContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.lockdir=c.HERE/'locks/full_20261011';cls.root=c.load_lock(cls.lockdir)
    def test_full_grid(self):
        from collections import Counter
        keys=[];counts=Counter();ids=[]
        for ident in self.root['groups']:
            g=c.group_lock(self.lockdir,self.root,ident)
            for i,r in g['tasks'].items():
                j=r['job'];ids.append(i);keys.append(tuple(j.get(k,'') for k in ('task_type','task','method','marker','source','target','family','seed')))
                counts[j['task_type']+'__'+j['task']]+=1
        self.assertEqual(len(keys),32370);self.assertEqual(len(set(keys)),32370);self.assertEqual(len(set(ids)),32370)
        self.assertEqual(dict(counts),self.root['branches'])
        self.assertEqual(sum(v for k,v in counts.items() if k.startswith('J1')),3060)
        self.assertEqual(sum(v for k,v in counts.items() if k.startswith('J2')),29310)
    def test_frozen_pilot_unchanged(self):
        for n,r in self.root['vendor'].items():self.assertEqual(c.record(c.HERE/'vendor/pilot'/n),r)
    def test_scientific_source_unchanged(self):
        for n,r in self.root['scientific'].items():self.assertEqual(c.record(c.STUDY.parent/'final_xu'/n),r)
    def test_source_roles_separate(self):
        for ident in self.root['groups']:
            g=c.group_lock(self.lockdir,self.root,ident)
            self.assertFalse(set(g['inputs'])&set(g['references']))
            self.assertFalse(any(Path(n).name in ('predictions.npz','predictions.json','RESULT.json') for n in g['inputs']))
    def test_order_independent_grid(self):c.check_grid(['b','a'],['a','b'])
    def test_duplicate_grid(self):
        with self.assertRaises(ValueError):c.check_grid(['a','a'],['a','b'])
    def test_missing_grid(self):
        with self.assertRaises(ValueError):c.check_grid(['a'],['a','b'])
    def test_extra_grid(self):
        with self.assertRaises(ValueError):c.check_grid(['a','b','c'],['a','b'])
    def test_foreign_grid(self):
        with self.assertRaises(ValueError):c.check_grid(['a','c'],['a','b'])
    def test_unknown_group(self):
        with self.assertRaises(ValueError):c.group_lock(self.lockdir,self.root,'unknown')
    def test_changed_lock_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);c.write(p/'LOCK.json',{} )
            with self.assertRaises(ValueError):c.load_lock(p)
    def test_environment_forces_threads(self):self.assertTrue(all(c.child_environment()[n]=='1' for n in c.THREADS))
    def test_disk_guard(self):
        usage=type('Usage',(),dict(free=c.RESERVE_BYTES-1))()
        with patch.object(c.shutil,'disk_usage',return_value=usage),self.assertRaises(ValueError):c.disk_check(c.HERE)
    def test_prediction_difference_not_hidden(self):self.assertTrue(full_compare.pilot_compare.tree_differences({'score':.1},{'score':.2}))
    def test_selected_epoch_not_hidden(self):self.assertTrue(full_compare.pilot_compare.tree_differences({'selected_epoch':1},{'selected_epoch':2}))
    def test_nonfinite_result_rejected(self):
        with self.assertRaises(ValueError):full_compare.pilot_compare.tree_differences(float('nan'),0.)

class OutputSafety(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
    def tearDown(self):self.tmp.cleanup()
    def test_exclusive(self):
        p=self.root/'out';c.reserve(p)
        with self.assertRaises(ValueError):c.reserve(p)
    def test_competition(self):
        ctx=mp.get_context('spawn');barrier=ctx.Barrier(2);q=ctx.Queue();ps=[ctx.Process(target=race,args=(str(self.root/'out'),barrier,q)) for _ in range(2)]
        for p in ps:p.start()
        for p in ps:p.join(20);self.assertEqual(p.exitcode,0)
        self.assertEqual(sorted([q.get(timeout=2),q.get(timeout=2)]),[False,True])
    def test_source_overlap_before_parent(self):
        p=self.root/'protected'/'new'/'out'
        with self.assertRaises(ValueError):c.reserve(p,[self.root/'protected'])
        self.assertFalse((self.root/'protected').exists())
    def test_alias_overlap(self):
        p=self.root/'protected';p.mkdir();(self.root/'alias').symlink_to(p)
        with self.assertRaises(ValueError):c.reserve(self.root/'alias'/'out',[p])
    def test_dangling(self):
        p=self.root/'out';p.symlink_to(self.root/'missing')
        with self.assertRaises(ValueError):c.reserve(p)
    def test_failed_marker(self):
        c.write(self.root/'FAILED.json',{})
        with self.assertRaises(ValueError):c.seal(self.root,{})
        self.assertFalse((self.root/'COMPLETE.json').exists())
    def test_sealed_changed_data(self):
        p=self.root/'data';p.write_text('a');c.seal(self.root,{});p.write_text('b')
        with self.assertRaises(ValueError):c.verify_seal(self.root)
    def test_sealed_extra_data(self):
        c.seal(self.root,{});(self.root/'extra').write_text('a')
        with self.assertRaises(ValueError):c.verify_seal(self.root)
    def test_seal_roundtrip(self):c.seal(self.root,dict(test=True));self.assertEqual(c.verify_seal(self.root)['details'],dict(test=True))

if __name__=='__main__':unittest.main()
