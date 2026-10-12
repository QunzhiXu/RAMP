import copy
from pathlib import Path
import sys
import unittest
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import combined as c

class Grid(unittest.TestCase):
    def setUp(self):
        self.root={'groups':{'group_0000':{},'group_0001':{}}}
        self.expected=[dict(id='group_0000',origin='original',manifest={'sha256':'a'}),dict(id='group_0001',origin='continuation',manifest={'sha256':'b'})]
    def rejects(self,actual):
        with self.assertRaises(ValueError):c.check_routes(actual,self.root,self.expected)
    def test_valid(self):c.check_routes(self.expected,self.root,self.expected)
    def test_duplicate(self):self.rejects([self.expected[0],self.expected[0]])
    def test_missing(self):self.rejects(self.expected[:1])
    def test_extra(self):self.rejects(self.expected+[self.expected[0]])
    def test_foreign(self):
        a=copy.deepcopy(self.expected);a[0]['id']='foreign';self.rejects(a)
    def test_traversal(self):
        a=copy.deepcopy(self.expected);a[0]['id']='../group_0000';self.rejects(a)
    def test_wrong_origin(self):
        a=copy.deepcopy(self.expected);a[0]['origin']='continuation';self.rejects(a)
    def test_unknown_origin(self):
        a=copy.deepcopy(self.expected);a[0]['origin']='pilot';self.rejects(a)
    def test_swapped_hash(self):
        a=copy.deepcopy(self.expected);a[0]['manifest']=a[1]['manifest'];self.rejects(a)
    def test_extra_field(self):
        a=copy.deepcopy(self.expected);a[0]['path']='other';self.rejects(a)
    def test_unknown_shape(self):self.rejects([None])
    def test_reorder_rejected(self):self.rejects(list(reversed(self.expected)))

if __name__=='__main__':unittest.main()
