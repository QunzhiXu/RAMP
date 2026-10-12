import ast
import copy as copying
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import full_common as old
import resume_common as c
import resume_run as run

class Continuation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.root=c.load_lock(c.HERE/'locks/full_20261011')
    def test_exact_remaining_grid(self):
        self.assertEqual(c.remaining(self.root),[f'group_{i:04d}' for i in range(845,860)])
    def test_no_original_parameter_change(self):
        self.assertEqual(old.RESERVE_BYTES,20*1024**3);self.assertEqual(c.RESERVE_BYTES,15*1024**3)
        self.assertEqual(old.codes(),c.interruption()['code_records'])
    def test_worker_only_changes_contract_import(self):
        original=(c.HERE/'full_worker.py').read_text()
        current=(c.HERE/'resume_worker.py').read_text()
        self.assertEqual(current,original.replace('from full_common import *','from resume_common import *'))
    def test_worker_scientific_function_unchanged(self):
        a=ast.parse((c.HERE/'full_worker.py').read_text());b=ast.parse((c.HERE/'resume_worker.py').read_text())
        self.assertEqual(ast.dump(next(n for n in a.body if isinstance(n,ast.FunctionDef))),ast.dump(next(n for n in b.body if isinstance(n,ast.FunctionDef))))
    def test_new_controller_explicit(self):
        self.assertEqual(set(c.codes()),set(old.CODE)|set(c.EXTRA_CODE))
        self.assertNotEqual(c.codes(),old.codes())
    def test_remaining_refuses_missing_group(self):
        root=copying.deepcopy(self.root);del root['groups']['group_0859']
        with self.assertRaises(ValueError):c.remaining(root)
    def test_remaining_refuses_foreign_group(self):
        root=copying.deepcopy(self.root);root['groups']['unexpected']=root['groups']['group_0859']
        with self.assertRaises(ValueError):c.remaining(root)
    def test_request_records_resource_authorization(self):
        r=c.expected_request('group_0845',c.HERE,c.HERE,c.codes())
        self.assertEqual(r['resource_policy'],c.RESOURCE_POLICY)
        self.assertEqual(r['continuation_of'],c.INTERRUPTION_SHA)
    def test_resource_lower_boundary_rejected(self):
        usage=type('Usage',(),dict(free=c.RESERVE_BYTES-1))()
        with patch.object(c.shutil,'disk_usage',return_value=usage),self.assertRaises(ValueError):c.disk_check(c.HERE)
    def test_resource_boundary_accepted(self):
        usage=type('Usage',(),dict(free=c.RESERVE_BYTES))()
        with patch.object(c.shutil,'disk_usage',return_value=usage):self.assertEqual(c.disk_check(c.HERE),usage.free)
    def test_resource_needed_included(self):
        usage=type('Usage',(),dict(free=c.RESERVE_BYTES+7))()
        with patch.object(c.shutil,'disk_usage',return_value=usage),self.assertRaises(ValueError):c.disk_check(c.HERE,8)
    def test_prior_failed_directory_cannot_be_promoted(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);c.write(p/'COMPLETE.json',{'status':'complete'})
            with self.assertRaisesRegex(ValueError,'relabelled'):c.prior_binding(p,self.root)
    def test_continuation_incomplete_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'incomplete'):run.binding(c.HERE/'locks/full_20261011',d)

if __name__=='__main__':unittest.main()
