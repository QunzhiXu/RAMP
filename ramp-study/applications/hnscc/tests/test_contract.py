import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import prepare
import specs
import study

class Safety(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name).resolve()
        self.source = self.root/'source'; self.source.mkdir()
        self.file = self.source/'data'; self.file.write_bytes(b'123')
    def tearDown(self): self.temp.cleanup()
    def test_exclusive_json(self):
        p=self.root/'record'; common.write(p, {'ok': True})
        with self.assertRaises(FileExistsError): common.write(p, {})
    def test_finite_json(self):
        with self.assertRaises(ValueError): common.write(self.root/'bad', {'x': float('nan')})
    def test_existing_output(self):
        with self.assertRaises(FileExistsError): common.reserve(self.source, [])
    def test_protected_child(self):
        with self.assertRaises(ValueError): common.reserve(self.source/'out', [self.source])
    def test_protected_parent(self):
        with self.assertRaises(ValueError): common.reserve(self.root, [self.source])
    def test_dangling_link(self):
        link=self.root/'link'; link.symlink_to(self.root/'absent')
        with self.assertRaises(ValueError): common.reserve(link, [])
    def test_alias_parent(self):
        link=self.root/'alias'; link.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(ValueError): common.reserve(link/'output', [])
    def test_no_preliminary_mkdir(self):
        with self.assertRaises(ValueError): common.reserve(self.source/'new'/'out', [self.source])
        self.assertFalse((self.source/'new').exists())
    def test_regular_symlink(self):
        p=self.root/'symlink'; p.symlink_to(self.file)
        with self.assertRaises(ValueError): common.record(p)
    def test_regular_hardlink(self):
        p=self.root/'hardlink'; os.link(self.file, p)
        with self.assertRaises(ValueError): common.record(p)
    def test_resource_study_limit(self):
        with self.assertRaises(RuntimeError): common.check_resources({'study_allocated_bytes':60_000_000_000,'filesystem_free_bytes':50_000_000_000})
    def test_resource_free_limit(self):
        with self.assertRaises(RuntimeError): common.check_resources({'study_allocated_bytes':1,'filesystem_free_bytes':19_999_999_999})
    def test_resource_valid(self):
        self.assertEqual(common.check_resources({'study_allocated_bytes':1,'filesystem_free_bytes':20_000_000_000})['study_allocated_bytes'],1)
    def test_environment(self):
        with patch.dict(os.environ, {'R_LIBS':'bad','R_LIBS_USER':'bad','R_LIBS_SITE':'bad','PYTHONPATH':'bad'}):
            env=common.environment(self.root)
        self.assertTrue(all(k not in env for k in ('R_LIBS','R_LIBS_USER','R_LIBS_SITE','PYTHONPATH')))
        self.assertTrue(all(env[k]=='1' for k in common.THREADS)); self.assertEqual(env['LC_ALL'],'C.UTF-8')
    def test_write_guard(self):
        out=self.root/'run'; out.mkdir(); link=out/'data'; link.symlink_to(self.file)
        code='from common import write_guard; from pathlib import Path; write_guard('+repr(str(out))+',[]); Path('+repr(str(link))+').write_text("bad")'
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=common.HERE,capture_output=True)
        self.assertNotEqual(result.returncode,0); self.assertEqual(self.file.read_bytes(),b'123')
    def test_failed_run(self):
        common.write(self.root/'FAILED.json',{})
        with self.assertRaisesRegex(ValueError,'Failed run'): study.verify_run(self.root,self.root)
    def test_invalid_lock(self):
        common.write(self.root/'LOCK.json',{})
        with self.assertRaisesRegex(ValueError,'Unrecognized'): prepare.verify(self.root)
    def test_missing_lock(self):
        with self.assertRaises(FileNotFoundError): prepare.verify(self.root)
    def test_cli_missing_lock_no_output(self):
        out=self.root/'out'
        p=subprocess.run([sys.executable,'-B',str(common.HERE/'study.py'),'fit','--lockdir',str(self.root),'--out',str(out)],capture_output=True)
        self.assertNotEqual(p.returncode,0); self.assertFalse(out.exists())


