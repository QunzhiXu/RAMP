import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import compare_execution as comp


class ComparisonTests(unittest.TestCase):
    def test_os_only_accepted(self):comp.sessions('R 4\nRunning under: macOS 26.6.1\nlocked','R 4\nRunning under: macOS 26.7.1\nlocked')
    def test_package_change_rejected(self):
        with self.assertRaises(ValueError):comp.sessions('Running under: macOS 26.6.1\nold','Running under: macOS 26.7.1\nnew')
    def test_linux_change_rejected(self):
        with self.assertRaises(ValueError):comp.sessions('Running under: macOS 26.6.1','Running under: Linux')
    def test_elapsed_only(self):comp.elapsed({'elapsed_seconds':1,'seed':42},{'elapsed_seconds':2,'seed':42})
    def test_elapsed_seed_rejected(self):
        with self.assertRaises(ValueError):comp.elapsed({'elapsed_seconds':1,'seed':42},{'elapsed_seconds':2,'seed':43})
    def test_invalid_elapsed(self):
        for value in (True,-1,float('nan'),float('inf'),'1'):
            with self.subTest(value=value),self.assertRaises(ValueError):comp.elapsed({'elapsed_seconds':1},{'elapsed_seconds':value})
    def test_nested_elapsed(self):comp.elapsed({'runtime':{'elapsed_seconds':1},'hash':'x'},{'runtime':{'elapsed_seconds':2},'hash':'x'},True)
    def test_nested_hash_rejected(self):
        with self.assertRaises(ValueError):comp.elapsed({'runtime':{'elapsed_seconds':1},'hash':'x'},{'runtime':{'elapsed_seconds':2},'hash':'y'},True)
    def warning(self,prefix,pid=12,name='Norm_batch_01_fsom1.fcs'):
        return f"cannot remove file '{prefix}.staging-{pid}/work/normalize/{name}', reason 'No such file or directory'"
    def test_cleanup_path_only(self):
        a='identity fallback\n'+self.warning('/old/alignment');b='identity fallback\n'+self.warning('/new/result',34)
        comp.warnings_relocation(a,b,'/old/alignment','/new/result')
    def test_other_warning_rejected(self):
        with self.assertRaises(ValueError):comp.warnings_relocation('identity fallback','different fallback','/old','/new')
    def test_warning_missing_line_rejected(self):
        with self.assertRaises(ValueError):comp.warnings_relocation(self.warning('/old'),'','/old','/new')
    def test_warning_changed_basename_rejected(self):
        with self.assertRaises(ValueError):comp.warnings_relocation(self.warning('/old'),self.warning('/new',name='Norm_batch_02_fsom1.fcs'),'/old','/new')
    def test_warning_other_path_rejected(self):
        with self.assertRaises(ValueError):comp.warnings_relocation(self.warning('/old'),self.warning('/other'),'/old','/new')
    def test_warning_other_reason_rejected(self):
        with self.assertRaises(ValueError):comp.warnings_relocation(self.warning('/old'),self.warning('/new').replace('No such file or directory','Permission denied'),'/old','/new')


if __name__=='__main__':unittest.main()
