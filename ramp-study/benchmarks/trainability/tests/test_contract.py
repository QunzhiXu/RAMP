import copy
import importlib
import importlib.util
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import inventory as i
import contract as c
import safe_io as s


def compete(path, barrier, queue):
    barrier.wait()
    try:s.reserve(path);queue.put('won')
    except (ValueError, FileExistsError):queue.put('refused')


class Safety(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        self.source=self.root/'source';self.source.mkdir();s.write(self.source/'data.json',{'x':1})
        self.cat={'schema':1,'files':{'data.json':s.record(self.source/'data.json')},'copies':['data.json'],'links':[]}
    def tearDown(self):self.temp.cleanup()
    def freeze(self):
        with patch.object(i,'collect',return_value=self.cat):return i.freeze(self.source,self.root/'out')
    def test_freeze_verify(self):
        report=self.freeze();self.assertEqual(i.verify(self.root/'out',report['catalog_sha256'],self.source)['copied_files'],1)
    def test_repeat_refused(self):
        self.freeze()
        with self.assertRaises(ValueError):self.freeze()
    def test_wrong_anchor(self):
        self.freeze()
        with self.assertRaises(ValueError):i.verify(self.root/'out','0'*64)
    def test_missing_completion(self):
        d=self.freeze();(self.root/'out/COMPLETE.json').unlink()
        with self.assertRaises(FileNotFoundError):i.verify(self.root/'out',d['catalog_sha256'])
    def test_extra_payload(self):
        d=self.freeze();s.write(self.root/'out/tree/extra.json',{})
        with self.assertRaises(ValueError):i.verify(self.root/'out',d['catalog_sha256'])
    def test_mutation(self):
        d=self.freeze();(self.root/'out/tree/data.json').write_text('{}')
        with self.assertRaises(ValueError):i.verify(self.root/'out',d['catalog_sha256'])
    def test_source_changed_after_collect(self):
        (self.source/'data.json').write_text('{}')
        with self.assertRaises(ValueError):self.freeze()
        self.assertTrue((self.root/'out/FAILED.json').exists());self.assertFalse((self.root/'out/COMPLETE.json').exists())
    def test_source_changes_after_copy(self):
        old=i.copy_checked
        def changed(src,dst,expected):old(src,dst,expected);src.write_text('{}')
        with patch.object(i,'copy_checked',side_effect=changed),self.assertRaises(ValueError):self.freeze()
        self.assertFalse((self.root/'out/COMPLETE.json').exists())
    def test_final_validation_failure(self):
        with patch.object(i,'payload',side_effect=ValueError('bad')),self.assertRaises(ValueError):self.freeze()
        self.assertFalse((self.root/'out/COMPLETE.json').exists())
    def test_completion_is_last(self):
        old=i.payload
        def checked(out,cat):self.assertFalse((out/'COMPLETE.json').exists());return old(out,cat)
        with patch.object(i,'payload',side_effect=checked):self.freeze()
    def test_dangling_symlink(self):
        out=self.root/'out';out.symlink_to(self.root/'absent')
        with self.assertRaises(ValueError):s.reserve(out)
    def test_protected_alias(self):
        alias=self.root/'alias';alias.symlink_to(self.source,target_is_directory=True)
        with self.assertRaises(ValueError):s.reserve(alias/'out',[self.source])
        self.assertFalse((self.source/'out').exists())
    def test_protected_ancestor(self):
        with self.assertRaises(ValueError):s.reserve(self.root/'parent',[self.root/'parent/child'])
        self.assertFalse((self.root/'parent').exists())
    def test_copy_hardlink_conflict(self):
        dst=self.root/'link';os.link(self.source/'data.json',dst)
        with self.assertRaises(FileExistsError):s.copy_checked(self.source/'data.json',dst,self.cat['files']['data.json'])
        self.assertEqual(s.record(dst),self.cat['files']['data.json'])
    def test_symlink_member(self):
        (self.source/'link').symlink_to(self.root/'outside')
        with self.assertRaises(ValueError):s.member(self.source,'link/file')
    def test_traversal(self):
        for name in ('../bad','/absolute','a/../../x','.'):
            with self.subTest(name=name),self.assertRaises(ValueError):s.relative(name)
    def test_bind_root(self):
        self.assertEqual(i.bind('/some/old/final_xu/a/b'),'a/b')
        with self.assertRaises(ValueError):i.bind('/other/a')
        with self.assertRaises(ValueError):i.bind('/old/final_xu/../bad')
    def test_two_process_claim(self):
        ctx=mp.get_context('spawn');bar=ctx.Barrier(2);q=ctx.Queue()
        procs=[ctx.Process(target=compete,args=(self.root/'out',bar,q)) for _ in range(2)]
        for p in procs:p.start()
        for p in procs:p.join(30);self.assertEqual(p.exitcode,0)
        self.assertEqual(sorted([q.get(timeout=3),q.get(timeout=3)]),['refused','won'])
    def test_move_no_source(self):
        d=self.freeze();moved=self.root/'moved with spaces';shutil.move(self.root/'out',moved);shutil.rmtree(self.source)
        self.assertEqual(i.verify(moved,d['catalog_sha256'])['copied_files'],1)
    def test_exact_field_not_note(self):
        s.write(self.source/'manifest.json',{'raw':'wrong','note':self.cat['files']['data.json']['sha256']})
        self.cat['files']['manifest.json']=s.record(self.source/'manifest.json')
        self.cat['links']=[{'owner':'manifest.json','fields':['raw'],'target':'data.json'}]
        with self.assertRaises(ValueError):i.verify_sources(self.cat,self.source)


class Aggregation(unittest.TestCase):
    def setUp(self):self.expected=pd.DataFrame({'id':['a','b'],'score':[.2,.3],'rows':[1,1],'rank':[1,2]})
    def test_good_grid(self):c.exact_grid(self.expected.iloc[::-1],self.expected,['id'])
    def test_duplicate(self):
        with self.assertRaises(ValueError):c.exact_grid(pd.concat([self.expected.iloc[:1]]*2),self.expected,['id'])
    def test_missing(self):
        with self.assertRaises(ValueError):c.exact_grid(self.expected.iloc[:1],self.expected,['id'])
    def test_foreign(self):
        wrong=self.expected.copy();wrong.loc[1,'id']='c'
        with self.assertRaises(ValueError):c.exact_grid(wrong,self.expected,['id'])
    def test_score_diff(self):
        wrong=self.expected.copy();wrong.loc[0,'score']+=.001
        with self.assertRaises(ValueError):c.compare_scores(wrong,self.expected,['id'])
    def test_nonfinite(self):
        with self.assertRaises(ValueError):c.finite([1,np.nan])
    def test_rows_diff(self):
        wrong=self.expected.copy();wrong.loc[0,'rows']=2
        with self.assertRaises(ValueError):c.compare_scores(wrong,self.expected,['id'])
    def test_rank_diff(self):
        wrong=self.expected.copy();wrong.loc[0,'rank']=2
        with self.assertRaises(ValueError):c.compare_scores(wrong,self.expected,['id'])
    def test_nested_equal_tasks(self):
        rows=[dict(task='a',direction='a1',marker='x',score=0)]*9+[dict(task='b',direction='b1',marker='x',score=1)]
        self.assertEqual(c.nested(pd.DataFrame(rows)),.5)


class FrozenPrimitives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lock=s.read(i.HERE/'LOCK.json');cls.root=i.HERE/lock['snapshot']
        i.verify(cls.root,lock['catalog_sha256'])
        folder=cls.root/'tree'/i.ROOT/'src/trainability'
        name='_figure3_frozen'
        spec=importlib.util.spec_from_file_location(name,folder/'__init__.py',submodule_search_locations=[str(folder)])
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        cls.inputs=importlib.import_module(name+'.inputs');cls.sampling=importlib.import_module(name+'.sampling');cls.targets=importlib.import_module(name+'.targets')
    def build(self,x,op='identity'):
        return self.inputs.build_assay_input(x,cell_ids=['a','b'],datasets=['d','d'],units=['u','u'],markers=['X','Y','Z'],
            canonical_marker={'X':'hidden','Y':'hidden','Z':'visible'},masked_marker='hidden',
            contracts={'d':dict(operator=op,review_status='verified_no_target_dependency',evidence_path='frozen',input_contract_version='test')})
    def test_alias_mask(self):
        d=self.build(np.array([[np.nan,np.inf,2],[9,8,3]]));self.assertEqual(d.markers,('Z',));np.testing.assert_array_equal(d.x,[[2],[3]])
    def test_mask_before_clr(self):
        d=self.build(np.array([[100.,999.,2],[1000,8,3]]),'clr_shared')
        np.testing.assert_allclose(d.x,np.log1p(np.array([[2],[3]])/(1+np.array([[2],[3]]))),rtol=0,atol=1e-15)
    def test_mask_perturb_invariance(self):
        a=np.array([[1.,2,3],[4,5,6]]);b=a.copy();b[:,:2]=1e20
        np.testing.assert_array_equal(self.build(a,'clr_shared').x,self.build(b,'clr_shared').x)
    def test_visible_nonfinite_rejected(self):
        with self.assertRaises(ValueError):self.build(np.array([[1,2,np.nan],[2,3,4]]))
    def test_signed_identity_retained(self):self.assertEqual(self.build(np.array([[1,2,-3],[2,3,4]])).x[0,0],-3)
    def test_supervision_disjoint(self):
        with self.assertRaises(ValueError):self.sampling.Supervision(('a',),('a',))
    def test_target_supervision_rejected(self):
        p=self.sampling.Supervision(('a',),('b',))
        with self.assertRaises(ValueError):p.assert_allowed(['a','b'],['b','c'])
    def test_within_patient_mode(self):
        p=self.sampling.source_split(['a','b','c','d'],dict.fromkeys('abcd','p'),.5,1,patient_mapping_verified=True)
        self.assertEqual(p['mode'],'within_patient_cell_validation')
    def test_patient_disjoint_mode(self):
        m=dict(a='p1',b='p1',c='p2',d='p2');p=self.sampling.source_split(list(m),m,.5,1,patient_mapping_verified=True)
        self.assertFalse({m[x] for x in p['fit_pool']} & {m[x] for x in p['validation_pool']})
    def test_budget_no_truncation(self):
        with self.assertRaises(ValueError):self.sampling.nested_draws(['a','b'],[3],1)
    def test_calibration_no_reuse(self):
        cal=self.targets.SourceCalibration.fit(['a'],[1],['u'],['a'])
        with self.assertRaises(ValueError):cal.transform(['a'],[1],['u'])
    def test_calibration_cannot_borrow_unit(self):
        cal=self.targets.SourceCalibration.fit(['a'],[1],['u'],['a'])
        with self.assertRaises(ValueError):cal.transform(['b'],[1],['other'])


if __name__=='__main__':unittest.main()
