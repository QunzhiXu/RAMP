import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from common import HERE,write,sha,record
from compare import table,array,tree_differences,normalize
from validate import report_binding

class Comparisons(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name).resolve()
        self.a=self.root/'a.csv'; self.b=self.root/'b.csv'
    def tearDown(self): self.temp.cleanup()
    def test_exact_table(self):
        pd.DataFrame({'id':['a','b'],'x':[1.,2.]}).to_csv(self.a,index=False)
        self.b.write_bytes(self.a.read_bytes()); self.assertTrue(table(self.a,self.b)['exact_values'])
    def test_changed_score_reported(self):
        pd.DataFrame({'id':['a'],'x':[1.]}).to_csv(self.a,index=False)
        pd.DataFrame({'id':['a'],'x':[2.]}).to_csv(self.b,index=False)
        self.assertEqual(table(self.a,self.b)['max_abs_difference'],1.)
    def test_identity_changed_reported(self):
        pd.DataFrame({'id':['a']}).to_csv(self.a,index=False)
        pd.DataFrame({'id':['b']}).to_csv(self.b,index=False)
        self.assertFalse(table(self.a,self.b)['exact_values'])
    def test_missing_column(self):
        pd.DataFrame({'a':[1]}).to_csv(self.a,index=False)
        pd.DataFrame({'b':[1]}).to_csv(self.b,index=False)
        with self.assertRaises(ValueError): table(self.a,self.b)
    def test_missing_values(self):
        pd.DataFrame({'a':[1.,2.]}).to_csv(self.a,index=False)
        pd.DataFrame({'a':[1.,np.nan]}).to_csv(self.b,index=False)
        with self.assertRaises(ValueError): table(self.a,self.b)
    def test_infinite(self):
        pd.DataFrame({'a':[np.inf]}).to_csv(self.a,index=False)
        self.b.write_bytes(self.a.read_bytes())
        with self.assertRaises(ValueError): table(self.a,self.b)
    def test_consumed_fields(self):
        pd.DataFrame({'a':[1],'legacy':[123]}).to_csv(self.a,index=False)
        pd.DataFrame({'a':[1]}).to_csv(self.b,index=False)
        self.assertTrue(table(self.a,self.b,['a'])['exact_values'])
    def test_consumed_field_missing(self):
        pd.DataFrame({'a':[1]}).to_csv(self.a,index=False); self.b.write_bytes(self.a.read_bytes())
        with self.assertRaises(ValueError): table(self.a,self.b,['other'])
    def test_array_dtype(self):
        np.save(self.root/'a.npy',np.ones(2,np.float32)); np.save(self.root/'b.npy',np.ones(2,np.float64))
        with self.assertRaises(ValueError): array(self.root/'a.npy',self.root/'b.npy')
    def test_array_difference_not_hidden(self):
        np.save(self.root/'a.npy',np.ones(2)); np.save(self.root/'b.npy',np.zeros(2))
        self.assertEqual(array(self.root/'a.npy',self.root/'b.npy')['changed_elements'],2)
    def test_json_extra_field(self): self.assertTrue(tree_differences({'a':1},{'a':1,'b':2}))
    def test_json_changed_scientific_field(self): self.assertEqual(tree_differences({'n':28},{'n':27})[0]['abs_difference'],1)
    def test_json_relocation_only(self): self.assertEqual(normalize({'path':'/old/a'},'/old',{}),{'path':'<TREE>/a'})

class Bindings(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name).resolve()
        self.run=self.root/'run'; self.lock=self.root/'lock'; self.comp=self.root/'comp'
        for p in (self.run,self.lock,self.comp): p.mkdir()
        write(self.run/'RUN_MANIFEST.json',{}); write(self.lock/'LOCK.json',{})
        self.report={'run_manifest_sha256':sha(self.run/'RUN_MANIFEST.json'),'lock_sha256':sha(self.lock/'LOCK.json'),
                     'controller':record(HERE/'compare.py'),'status':'completed_with_differences'}
    def tearDown(self): self.temp.cleanup()
    def seal(self):
        write(self.comp/'report.json',self.report)
        write(self.comp/'COMPLETE.json',{'status':'complete','report_sha256':sha(self.comp/'report.json')})
    def test_valid(self): self.seal(); self.assertEqual(report_binding(self.lock,self.run,self.comp)[0],self.report)
    def test_wrong_run_resealed(self):
        self.report['run_manifest_sha256']='0'*64; self.seal()
        with self.assertRaisesRegex(ValueError,'run_manifest'): report_binding(self.lock,self.run,self.comp)
    def test_wrong_lock_resealed(self):
        self.report['lock_sha256']='0'*64; self.seal()
        with self.assertRaisesRegex(ValueError,'lock_sha'): report_binding(self.lock,self.run,self.comp)
    def test_unknown_comparer(self):
        self.report['controller']['sha256']='0'*64; self.seal()
        with self.assertRaises(ValueError): report_binding(self.lock,self.run,self.comp)
    def test_unknown_status(self):
        self.report['status']='passed'; self.seal()
        with self.assertRaises(ValueError): report_binding(self.lock,self.run,self.comp)
    def test_failed_report(self):
        self.seal(); write(self.comp/'FAILED.json',{})
        with self.assertRaises(ValueError): report_binding(self.lock,self.run,self.comp)
    def test_missing_binding(self):
        del self.report['run_manifest_sha256']; self.seal()
        with self.assertRaises(ValueError): report_binding(self.lock,self.run,self.comp)

if __name__=='__main__': unittest.main()
