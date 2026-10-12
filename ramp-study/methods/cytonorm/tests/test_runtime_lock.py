import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fixtures import inv,io
import probe_runtime as probe


class RuntimeLock(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='cytonorm_lock_');self.root=Path(self.tmp.name).resolve()
        self.lib=self.root/'runtime/.Rlib';self.lib.mkdir(parents=True)
        self.contract=probe.runtime_contract()
    def tearDown(self):self.tmp.cleanup()

    def test_authentic_lock(self):
        io.copy_new(inv.SNAPSHOT/'environment/runtime_lock.csv',self.lib/'runtime_lock.csv',self.contract['lock'])
        self.assertEqual(probe.runtime_contract(self.lib),self.contract)
        self.assertEqual(len(self.contract['versions']),69)

    def test_shortened_lock_before_any_output_or_R(self):
        (self.lib/'runtime_lock.csv').write_text('package,version,library\nbase,4.3.2,ignored\n')
        with patch.object(probe.subprocess,'run') as run,self.assertRaises(ValueError):
            probe.probe(self.lib,Path('/usr/local/bin/Rscript'),self.root,self.root/'out')
        run.assert_not_called();self.assertFalse((self.root/'out').exists())

    def test_changed_version(self):
        text=(inv.SNAPSHOT/'environment/runtime_lock.csv').read_text().replace('"CytoNorm","2.0.6"','"CytoNorm","9.9.9"')
        (self.lib/'runtime_lock.csv').write_text(text)
        with self.assertRaises(ValueError):probe.runtime_contract(self.lib)

    def test_missing_lock(self):
        with self.assertRaises(FileNotFoundError):probe.runtime_contract(self.lib)

    def test_changed_lock_after_subprocess(self):
        io.copy_new(inv.SNAPSHOT/'environment/runtime_lock.csv',self.lib/'runtime_lock.csv',self.contract['lock'])
        def changed(*a,**k):
            (self.lib/'runtime_lock.csv').write_text('changed')
            return type('Result',(),{'returncode':0})()
        with patch.object(probe.subprocess,'run',side_effect=changed),self.assertRaises(ValueError):
            probe.probe(self.lib,Path('/usr/local/bin/Rscript'),self.root,self.root/'out')
        self.assertTrue((self.root/'out/FAILED.json').exists());self.assertFalse((self.root/'out/report.json').exists())

    def summary(self):
        return {'status':'passed','runtime_versions':copy.deepcopy(self.contract['versions']),
            'runtime_lock_sha256':self.contract['lock']['sha256'],
            'results':[{'maximum_absolute_difference':0,'preserialization_model_replay_maximum_difference':0}]*2}

    def test_complete_summary(self):self.assertEqual(probe.check_summary(self.summary()),self.contract)

    def test_short_summary_map(self):
        s=self.summary();s['runtime_versions']={'base':'4.3.2'}
        with self.assertRaises(ValueError):probe.check_summary(s)

    def test_extra_summary_version(self):
        s=self.summary();s['runtime_versions']['unknown']='1'
        with self.assertRaises(ValueError):probe.check_summary(s)

    def test_summary_hash_mismatch(self):
        s=self.summary();s['runtime_lock_sha256']='0'*64
        with self.assertRaises(ValueError):probe.check_summary(s)

    def test_preserialization_difference_rejected(self):
        s=self.summary();s['results'][0]['preserialization_model_replay_maximum_difference']=1e-15
        with self.assertRaises(ValueError):probe.check_summary(s)


if __name__=='__main__':unittest.main()
