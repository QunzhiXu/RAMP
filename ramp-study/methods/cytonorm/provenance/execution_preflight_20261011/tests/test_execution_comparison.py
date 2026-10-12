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


if __name__=='__main__':unittest.main()
