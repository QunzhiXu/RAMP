import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from validate import comparison_coverage
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

