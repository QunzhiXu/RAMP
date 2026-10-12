import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from release_check import verify, BUDGET, unique_object


class SourceCandidate(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        (self.root/'README.md').write_text('synthetic source fixture\n')
        self.m={'status':'internal_source_candidate_publication_blocked','publish_allowed':False,'budget_bytes':BUDGET,
                'full_study_runnable_without_external_evidence':False,'data_payloads_copied':False,
                'files':{'README.md':self.record(self.root/'README.md')}}
        self.save()

    def record(self,p):
        b=p.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

    def save(self):
        (self.root/'RELEASE_MANIFEST.json').write_text(json.dumps(self.m))

    def test_valid_candidate_remains_unapproved(self):
        self.assertFalse(verify(self.root)['publish_allowed'])

    def test_changed_payload_rejected(self):
        (self.root/'README.md').write_text('changed')
        with self.assertRaisesRegex(ValueError,'hash differs'):verify(self.root)

    def test_missing_payload_rejected(self):
        (self.root/'README.md').unlink()
        with self.assertRaisesRegex(ValueError,'file set differs'):verify(self.root)

    def test_extra_payload_rejected(self):
        (self.root/'new.csv').write_text('not declared')
        with self.assertRaisesRegex(ValueError,'file set differs'):verify(self.root)

    def test_link_rejected(self):
        (self.root/'link').symlink_to(self.root/'README.md')
        with self.assertRaisesRegex(ValueError,'Nonregular'):verify(self.root)

    def test_directory_link_rejected(self):
        (self.root/'link').symlink_to(self.root,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'Directory link'):verify(self.root)

    def test_no_publication_approval_inferred(self):
        self.m['publish_allowed']=True;self.save()
        with self.assertRaisesRegex(ValueError,'does not approve'):verify(self.root)

    def test_scope_cannot_be_overstated(self):
        self.m['full_study_runnable_without_external_evidence']=True;self.save()
        with self.assertRaisesRegex(ValueError,'scope differs'):verify(self.root)

    def test_budget_cannot_be_raised(self):
        self.m['budget_bytes']=10*BUDGET;self.save()
        with self.assertRaisesRegex(ValueError,'scope differs'):verify(self.root)

    def test_scientific_payload_even_if_sealed_rejected(self):
        p=self.root/'matrix.npy';p.write_bytes(b'synthetic placeholder')
        self.m['files']['matrix.npy']=self.record(p);self.save()
        with self.assertRaisesRegex(ValueError,'Disallowed'):verify(self.root)

    def test_pdf_even_if_sealed_rejected(self):
        p=self.root/'paper.pdf';p.write_bytes(b'synthetic placeholder')
        self.m['files']['paper.pdf']=self.record(p);self.save()
        with self.assertRaisesRegex(ValueError,'Disallowed'):verify(self.root)

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):unique_object([('files',{}),('files',{})])

    def test_invalid_python_rejected(self):
        p=self.root/'bad.py';p.write_text('def broken(:')
        self.m['files']['bad.py']=self.record(p);self.save()
        with self.assertRaises(SyntaxError):verify(self.root)

    def test_root_git_directory_ignored(self):
        (self.root/'.git').mkdir();(self.root/'.git/config').write_text('synthetic metadata')
        self.assertEqual(verify(self.root)['files'],2)


if __name__=='__main__':unittest.main()
