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

class Scope(unittest.TestCase):
    def test_unique_stages(self): self.assertEqual(len(specs.STAGES),len(set(specs.STAGES)))
    def test_scope(self): self.assertEqual(len(specs.STAGES),8); self.assertEqual(len(common.MARKERS),9)
    def test_pin_and_code(self): self.assertEqual(len(prepare.LOCK_SHA),64); self.assertEqual(len(common.codes()),6)
    def test_frozen_lock(self):
        lock=prepare.verify(common.HERE/'frozen_20261011')
        self.assertEqual(len(lock['files']),1222); self.assertEqual(len(specs.CODES),12)
    def test_alignment_pins(self): self.assertEqual(set(prepare.ALIGN_SHA),set(specs.ALIGN_FILES))
    def test_prefix_no_fit(self):
        import adapt, ast
        path=common.HERE/'frozen_20261011/code'/specs.SCRIPTS/'align.py'
        p=adapt.prefix(path.read_text()); source=ast.unparse(p)
        self.assertNotIn('ramp.fit_transform(',source)
        original=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
        count=len(p.body[0].body)-2
        self.assertEqual(ast.dump(ast.Module(body=original.body[:count],type_ignores=[])),ast.dump(ast.Module(body=p.body[0].body[:count],type_ignores=[])))
    def test_wrong_prefix_rejected(self):
        import adapt
        with self.assertRaises(ValueError): adapt.prefix('def main():\n    return 1')
    def test_projection_no_outcomes(self):
        source=(common.HERE/'frozen_20261011/code'/specs.SCRIPTS/'project.py').read_text()
        self.assertNotIn('read_csv(STORY/\'survival',source)
        self.assertNotIn('PREDICTIONS_AND_TRUTH',source)
    def test_clinical_after_freeze(self):
        source=(common.HERE/'frozen_20261011/code'/specs.SCRIPTS/'analyze.py').read_text()
        self.assertLess(source.index("rt.save(OUT/'MOLECULAR_FREEZE"),source.index('clinical=pd.read_csv'))
    def test_accepted_alignment_not_refit_stage(self): self.assertNotIn('align',specs.STAGES)
    def test_dfs_not_os(self):
        source=(common.HERE/'frozen_20261011/code'/specs.SCRIPTS/'analyze.py').read_text()
        self.assertIn('len(p)==280 and p.event.sum()==96',source)
        self.assertIn('p.dfs_days/365.25',source)

if __name__=='__main__': unittest.main()
