import json
from pathlib import Path
import tempfile
import unittest
import prepare_manifest as p
import resume_approved as r


class PrefixRecovery(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'receipts'
        self.rows=[dict(bytes=3,sha256='abc',canonical=dict(bytes=3,sha256='abc'))]
        self.intent=dict(event='intent',ordinal=0,row_sha256=p.digest(self.rows[0]),utc='before',
                         target=r.e.rec(self.rows[0]),canonical=r.e.rec(self.rows[0]['canonical']))
        self.done=dict(event='deleted',ordinal=0,utc='after')

    def check(self,values):
        self.path.write_bytes(b''.join(p.encode(x) for x in values))
        r.verify_prefix(self.rows,self.path,1)

    def test_exact_prefix(self): self.check([self.intent,self.done])
    def test_missing_done(self):
        with self.assertRaises(StopIteration): self.check([self.intent])
    def test_extra_event(self):
        with self.assertRaises(ValueError): self.check([self.intent,self.done,self.done])
    def test_wrong_index(self):
        with self.assertRaises(ValueError): self.check([self.intent,dict(self.done,ordinal=1)])
    def test_wrong_row(self):
        with self.assertRaises(ValueError): self.check([dict(self.intent,row_sha256='wrong'),self.done])
    def test_wrong_live_hash(self):
        with self.assertRaises(ValueError): self.check([dict(self.intent,target={'bytes':3,'sha256':'wrong'}),self.done])
    def test_reversed_events(self):
        with self.assertRaises(ValueError): self.check([self.done,self.intent])


if __name__=='__main__': unittest.main()
