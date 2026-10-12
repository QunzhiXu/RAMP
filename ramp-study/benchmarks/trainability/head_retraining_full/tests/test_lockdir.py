"""Small routing regressions; real relocated CLI/payload proof is in validate_lockdir_fix.py."""
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import combined as c


class StopAtRoutes(Exception): pass


class ExternalLocks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.controller = self.base/'relocated controller'; self.controller.mkdir()
        self.external = self.base/'external locks with spaces'; self.external.mkdir()
        self.default = self.controller/'locks/full_20261011'
        self.root = dict(groups={'group_0000':{}, 'group_0001':{}})
        self.prior = dict(sealed_groups={'group_0000':dict(manifest=dict(sha256='old'))})
        self.batch = dict(groups={'group_0001':dict(sha256='new')})
        c.write(self.external/'LOCK.json', dict(token='valid'))
        self.expected = [dict(id='group_0000', origin='original', manifest=dict(sha256='old')),
                         dict(id='group_0001', origin='continuation', manifest=dict(sha256='new'))]

    def routes(self):
        # Only pin/provenance setup is mocked. The binding stub really reads the
        # argument's file; a hidden read from the default path fails these tests.
        def load(directory, continuation):
            c.require(c.read(Path(directory)/'LOCK.json') == dict(token='valid'), 'Explicit lock invalid')
            return self.root, self.batch
        with patch.object(c,'HERE',self.controller), patch.object(c,'controller_binding'), \
             patch.object(c,'prior_binding',return_value=self.prior), \
             patch.object(c.resume_run,'binding',side_effect=load) as binding:
            result = c.expected_routes(self.external,self.root,self.base/'original',self.base/'continuation')
            binding.assert_called_once_with(self.external,self.base/'continuation')
            return result

    def test_external_lock_without_default_directory(self):
        self.assertFalse((self.controller/'locks').exists())
        self.assertEqual(self.routes(), self.expected)
        self.assertFalse((self.controller/'locks').exists())

    def test_invalid_default_is_not_read(self):
        self.default.mkdir(parents=True); c.write(self.default/'LOCK.json',dict(token='poison'))
        self.assertEqual(self.routes(), self.expected)

    def test_missing_explicit_lock_cannot_fall_back(self):
        self.default.mkdir(parents=True); c.write(self.default/'LOCK.json',dict(token='valid'))
        self.external = self.base/'missing external locks'
        with self.assertRaises(FileNotFoundError): self.routes()
        self.assertFalse(self.external.exists())

    def test_invalid_explicit_lock_cannot_fall_back(self):
        self.default.mkdir(parents=True); c.write(self.default/'LOCK.json',dict(token='valid'))
        self.external = self.base/'invalid external locks'; self.external.mkdir()
        c.write(self.external/'LOCK.json',dict(token='invalid'))
        with self.assertRaisesRegex(ValueError,'Explicit lock invalid'): self.routes()

    def test_binding_forwards_external_lock(self):
        with patch.object(c,'load_lock',return_value=self.root), \
             patch.object(c,'expected_routes',side_effect=StopAtRoutes) as routes:
            with self.assertRaises(StopAtRoutes): c.binding(self.external,'original','continuation','combined')
            routes.assert_called_once_with(self.external,self.root,'original','continuation')

    def test_verify_run_forwards_external_lock(self):
        with patch.object(c,'binding',side_effect=StopAtRoutes) as binding:
            with self.assertRaises(StopAtRoutes): c.verify_run(self.external,'source','original','continuation','combined')
            binding.assert_called_once_with(self.external,'original','continuation','combined')

    def assemble(self, changed=False):
        continuation=self.base/'continuation'; continuation.mkdir()
        c.write(continuation/'BATCH_MANIFEST.json',dict(fixture=True))
        output=self.base/'new index'
        values=[self.expected, [] if changed else self.expected]
        with ExitStack() as stack:
            stack.enter_context(patch.object(c,'load_lock',return_value=self.root))
            routes=stack.enter_context(patch.object(c,'expected_routes',side_effect=values))
            stack.enter_context(patch.object(c,'verify_payloads',return_value=dict(jobs=32370,predictions=123)))
            stack.enter_context(patch.object(c,'interruption',return_value=dict(code_records={})))
            stack.enter_context(patch.object(c,'codes',return_value={}))
            if changed:
                with self.assertRaisesRegex(ValueError,'Group bindings changed'):
                    c.assemble(self.external,self.base/'source',self.base/'original',continuation,output)
            else:
                c.assemble(self.external,self.base/'source',self.base/'original',continuation,output)
            self.assertEqual(routes.call_count,2)
            for call in routes.call_args_list:
                self.assertEqual(call.args,(self.external,self.root,self.base/'original',continuation))
        return output

    def test_assemble_and_final_recheck_forward_same_lock(self):
        output=self.assemble(); c.verify_seal(output)

    def test_assemble_final_recheck_still_rejects_changed_routes(self):
        output=self.assemble(changed=True)
        self.assertTrue((output/'FAILED.json').is_file())
        self.assertFalse((output/'COMPLETE.json').exists())


if __name__ == '__main__': unittest.main()
