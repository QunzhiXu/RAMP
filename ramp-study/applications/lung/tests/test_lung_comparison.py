import json
from pathlib import Path
import sys
import unittest
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from common import EXP, STORY
from compare import canonical,scientific_json,arrays,tree_differences,json_comparison
from validate import comparison_coverage

class LungComparison(unittest.TestCase):
    def test_identical_inherited_design_not_relocated(self):
        with tempfile.TemporaryDirectory() as t:
            a=Path(t)/'a'; b=Path(t)/'b'; text=json.dumps({'input_sha256':{'/old/upstream/file':'abc'}})
            a.write_text(text); b.write_text(text)
            diff,excluded,method=json_comparison(a,b,EXP+'/inputs/DESIGN.json','/old','/new',{})
            self.assertEqual(diff,[]); self.assertEqual(method,'byte_identity'); self.assertEqual(excluded,{'historical':{},'new':{}})
    def test_changed_design_not_hidden(self):
        with tempfile.TemporaryDirectory() as t:
            a=Path(t)/'a'; b=Path(t)/'b'; a.write_text('{"n": 1}'); b.write_text('{"n": 2}')
            diff,_,method=json_comparison(a,b,EXP+'/inputs/DESIGN.json','/old','/new',{})
            self.assertTrue(diff); self.assertEqual(method,'recorded_provenance_relocation')
    def test_old_test_release_path(self):
        self.assertEqual(canonical('/old/'+STORY+'/test/new_senosain_20260921/a','/old',{}),'<TREE>/'+EXP+'/a')
    def test_signature_is_parsed_not_dropped(self):
        old={'signature':'a|b|c|'+json.dumps({'/old/'+EXP+'/x':'oldhash'}),'n_cells':10}
        new={'signature':'a|b|c|'+json.dumps({'/new/'+EXP+'/x':'newhash'}),'n_cells':10}
        a,_=scientific_json(old,EXP+'/cords/patients/a.json','/old',{'oldhash':'newhash'})
        b,_=scientific_json(new,EXP+'/cords/patients/a.json','/new',{})
        self.assertEqual(a,b)
    def test_unknown_signature_code_not_hidden(self):
        a,_=scientific_json({'signature':'a|b|c|{}'},EXP+'/cords/patients/a.json','/old',{})
        b,_=scientific_json({'signature':'wrong|b|c|{}'},EXP+'/cords/patients/a.json','/old',{})
        self.assertTrue(tree_differences(a,b))
    def test_invalid_signature(self):
        with self.assertRaises(ValueError): scientific_json({'signature':'bad'},EXP+'/cords/patients/a.json','/',{})
    def test_seconds_only_projection_completion(self):
        a,e=scientific_json({'seconds':3,'n_cells':20},EXP+'/cords/COMPLETE.json','/old',{})
        self.assertEqual(a,{'n_cells':20}); self.assertEqual(e,{'seconds':3})
    def test_negative_duration_rejected(self):
        with self.assertRaises(ValueError): scientific_json({'seconds':-1},EXP+'/cords/COMPLETE.json','/old',{})
    def test_nan_duration_rejected(self):
        with self.assertRaises(ValueError): scientific_json({'seconds':float('nan')},EXP+'/cords/COMPLETE.json','/old',{})
    def test_scientific_extra_key_preserved(self):
        a,_=scientific_json({'seconds':1,'events':97},EXP+'/analysis/a.json','/old',{})
        self.assertEqual(a,{'seconds':1,'events':97})
    def test_array_shape_rejected(self):
        with self.assertRaises(ValueError): arrays(np.ones((2,2)),np.ones((2,3)))
    def test_array_dtype_rejected(self):
        with self.assertRaises(ValueError): arrays(np.ones(2,np.float32),np.ones(2,np.float64))
    def test_nonfinite_array_rejected(self):
        with self.assertRaises(ValueError): arrays(np.array([np.nan]),np.array([np.nan]))
    def test_all_changed_values_counted(self):
        self.assertEqual(arrays(np.ones(4),np.zeros(4))['changed_elements'],4)

class Coverage(unittest.TestCase):
    def setUp(self):
        self.lock={'files':{'a':{'role':'reference'},'b':{'role':'reference'},'c':{'role':'input'}}}
        self.report={'tables':[{'path':'a','byte_identical':True},{'path':'b','byte_identical':False}],
            'arrays':[],'npzs':[],'jsons':[],'texts':[],'reference_files_compared':2,'byte_identical_files':1}
    def test_complete(self): self.assertEqual(comparison_coverage(self.report,self.lock),{'reference_files':2})
    def test_duplicate(self):
        self.report['tables'][1]=self.report['tables'][0]
        with self.assertRaises(ValueError): comparison_coverage(self.report,self.lock)
    def test_missing(self):
        self.report['tables'].pop()
        with self.assertRaises(ValueError): comparison_coverage(self.report,self.lock)
    def test_wrong_count(self):
        self.report['reference_files_compared']=0
        with self.assertRaises(ValueError): comparison_coverage(self.report,self.lock)
    def test_wrong_bytes_count(self):
        self.report['byte_identical_files']=2
        with self.assertRaises(ValueError): comparison_coverage(self.report,self.lock)
    def test_wrong_category(self):
        self.report['arrays']=None
        with self.assertRaises(ValueError): comparison_coverage(self.report,self.lock)

if __name__=='__main__': unittest.main()
