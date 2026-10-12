import ast
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import adapt, common, prepare, specs

class Scope(unittest.TestCase):
    def test_stages(self):
        self.assertEqual(len(specs.STAGES),9); self.assertEqual(len(set(specs.STAGES)),9)
    def test_markers(self):
        self.assertEqual(len(common.MARKERS),22)
        for x in ['LAG3','CD39','TIGIT']: self.assertNotIn(x,common.MARKERS)
    def test_roles_disjoint(self):
        lock=prepare.verify(common.HERE/'frozen_20261011')
        self.assertFalse(set(specs.INPUTS)&set(specs.references(Path(lock['source_root']))))
        self.assertEqual(len(specs.CODES),9)
    def test_no_refit(self): self.assertNotIn('align',specs.STAGES)
    def test_no_reference_mirror(self):
        self.assertFalse(any('predictions' in x or '/screen/' in x or '/projection/' in x for x in specs.INPUTS))
    def test_primary_survival(self):
        code=(common.HERE/'frozen_20261011/code'/specs.CODES[1]).read_text()
        self.assertIn('nrow(analysis) != 74L || sum(analysis$os_event) != 24L',code)
        self.assertIn('Surv(OS, os_event) ~ spatial_score_z',code)
    def test_molecular_prefix(self):
        source=(common.HERE/'frozen_20261011/code'/specs.CODES[6]).read_text()
        selected=adapt.fixed_fraction_function(source).body[0]
        original=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='compare')
        for x,y in zip(selected.body[:-1],original.body): self.assertEqual(ast.dump(x),ast.dump(y))
        self.assertNotIn('baseline',ast.unparse(selected))
    def test_no_candidate_roi_scan(self):
        source=(common.HERE/'adapt.py').read_text()
        self.assertNotIn('rank_rois',source); self.assertIn("'521_S1reg012'",source)
    def test_links_allow_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); base=root/'tree'; base.mkdir(); orig=root/'source'; orig.write_bytes(b'abc')
            link=base/'input'; link.symlink_to(orig); rel=common.EXP+'/input'
            lock={'input_paths':{rel:str(orig)},'files':{rel:common.record(orig)}}
            adapt.linked_formal_path(link,base,lock)
            orig.write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'changed'): adapt.linked_formal_path(link,base,lock)
    def test_links_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); base=root/'tree'; base.mkdir(); orig=root/'source'; orig.write_bytes(b'abc')
            link=base/'input'; link.symlink_to(orig)
            with self.assertRaisesRegex(ValueError,'Unrecognized'): adapt.linked_formal_path(link,base,{'input_paths':{}})
    def test_forbidden_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); folder=root/'test'; folder.mkdir(); p=folder/'input'; p.write_bytes(b'abc')
            with self.assertRaisesRegex(ValueError,'Forbidden'): adapt.linked_formal_path(p,root,{})

if __name__=='__main__': unittest.main()
