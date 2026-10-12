import copy
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from validate import comparison_status

class StatusConsistency(unittest.TestCase):
    def setUp(self):
        self.report={'status':'exact_with_disclosed_reuse_scope','tables':[{'exact_values':True}],
            'arrays':[{'exact_values':True}],'npzs':[{'arrays':{'x':{'exact_values':True}}}],
            'jsons':[{'exact_after_recorded_provenance_relocation':True}],
            'texts':[{'equal_except_disclosed_os_line':True}]}
    def test_exact(self): self.assertEqual(comparison_status(self.report),self.report['status'])
    def test_false_differences_status(self):
        self.report['status']='completed_with_differences'
        with self.assertRaises(ValueError): comparison_status(self.report)
    def test_every_category_false_rejected(self):
        for kind,key in [('tables','exact_values'),('arrays','exact_values'),('jsons','exact_after_recorded_provenance_relocation'),('texts','equal_except_disclosed_os_line')]:
            with self.subTest(kind=kind):
                r=copy.deepcopy(self.report); r[kind][0][key]=False
                with self.assertRaisesRegex(ValueError,'disagrees'): comparison_status(r)
    def test_npz_false_rejected(self):
        self.report['npzs'][0]['arrays']['x']['exact_values']=False
        with self.assertRaisesRegex(ValueError,'disagrees'): comparison_status(self.report)
    def test_legitimate_differences(self):
        self.report['tables'][0]['exact_values']=False; self.report['status']='completed_with_differences'
        self.assertEqual(comparison_status(self.report),'completed_with_differences')
    def test_non_boolean_rejected(self):
        for value in ('false',1,None):
            with self.subTest(value=value):
                self.report['tables'][0]['exact_values']=value
                with self.assertRaises(ValueError): comparison_status(self.report)
    def test_missing_flag_rejected(self):
        del self.report['jsons'][0]['exact_after_recorded_provenance_relocation']
        with self.assertRaises(ValueError): comparison_status(self.report)
    def test_empty_npz_rejected(self):
        self.report['npzs'][0]['arrays']={}
        with self.assertRaises(ValueError): comparison_status(self.report)
    def test_npz_non_boolean_rejected(self):
        self.report['npzs'][0]['arrays']['x']['exact_values']=1
        with self.assertRaises(ValueError): comparison_status(self.report)

if __name__=='__main__': unittest.main()

