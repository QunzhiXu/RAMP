import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import inventory as inv
import safe_io as io
import study
import worker


def contend(path, barrier, q):
    barrier.wait()
    try:io.reserve(path);q.put('owner')
    except FileExistsError:q.put('refused')


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()

    def test_all_seven_contracts(self):
        for c in inv.CASES:
            with self.subTest(case=c):
                s=inv.spec(c);p=study.plan(c)
                self.assertEqual(s['case'],c);self.assertEqual(p['case'],c)
                self.assertEqual(set(s['inputs']),{'raw.npy','fit_metadata.csv.gz','markers.txt'})
                self.assertEqual(set(s['references']),set(s['matrices'])|set(s['audits'])|set(s['extras']))
                self.assertEqual(s['reference_only'],c=='bone_marrow')

    def test_lung_policies_distinct(self):
        self.assertEqual(inv.spec('lung_cancer')['pair_fit'],{'minimum':7000,'per_unit':0,'maximum':7000})
        self.assertEqual(inv.spec('lung_story')['pair_fit'],{'minimum':4000,'per_unit':100,'maximum':8000})
        self.assertIn('Monkman CODEX',inv.spec('lung_cancer')['datasets']);self.assertIn('Zheng mIF',inv.spec('lung_story')['datasets'])

    def test_melanoma_dtype_preserved(self):self.assertEqual(inv.spec('melanoma')['input_dtype'],'float32')
    def test_tnbc_cohort(self):self.assertEqual(inv.spec('tnbc')['dataset_cells']['Meyer IMC'],22701)
    def test_bad_case(self):
        with self.assertRaises(ValueError):study.plan('unknown')
    def test_no_fit_labels_in_plan(self):
        for c in inv.CASES:
            self.assertEqual([p for p in study.plan(c)['files'] if p.startswith('input/')],['input/raw.npy','input/fit_metadata.csv.gz','input/markers.txt'])

    def test_protected_sources(self):
        for p in study.protected():
            with self.subTest(path=p),self.assertRaises((ValueError,FileExistsError)):
                io.reserve(Path(p)/'not-created',study.protected())
            self.assertFalse((Path(p)/'not-created').exists())

    def test_same_target_refused(self):
        p=io.reserve(self.root/'out')
        with self.assertRaises(FileExistsError):io.reserve(p)
    def test_dangling_symlink_refused(self):
        p=self.root/'link';p.symlink_to(self.root/'missing')
        with self.assertRaises(FileExistsError):io.reserve(p)
    def test_protected_alias_refused(self):
        p=self.root/'source';p.mkdir();alias=self.root/'alias';alias.symlink_to(p,target_is_directory=True)
        with self.assertRaises(ValueError):io.reserve(alias/'out',(p,))
        self.assertFalse((p/'out').exists())
    def test_protected_ancestor_refused(self):
        with self.assertRaises(ValueError):io.reserve(self.root/'parent',(self.root/'parent/source',))
        self.assertFalse((self.root/'parent').exists())
    def test_copy_never_truncates_hardlink(self):
        src=self.root/'src';src.write_bytes(b'input');target=self.root/'target';os.link(src,target)
        with self.assertRaises(FileExistsError):io.copy(src,target,io.record(src))
        self.assertEqual(src.read_bytes(),b'input')
    def test_copy_mismatch_preserves_partial(self):
        src=self.root/'src';src.write_bytes(b'input');row=io.record(src);row['sha256']='0'*64
        with self.assertRaises(ValueError):io.copy(src,self.root/'dest',row)
        self.assertTrue((self.root/'dest').exists())
    def test_exclusive_json(self):
        p=self.root/'r.json';io.write(p,{'a':1})
        with self.assertRaises(FileExistsError):io.write(p,{'a':2})
        self.assertEqual(io.read(p),{'a':1})
    def test_real_two_process_reservation(self):
        ctx=mp.get_context('spawn');barrier=ctx.Barrier(2);q=ctx.Queue();target=self.root/'race'
        jobs=[ctx.Process(target=contend,args=(target,barrier,q)) for _ in range(2)]
        for p in jobs:p.start()
        for p in jobs:p.join(20);self.assertEqual(p.exitcode,0)
        self.assertEqual(sorted(q.get(timeout=5) for _ in jobs),['owner','refused'])
    def test_payload_rejects_extra(self):
        p=self.root/'x';p.write_bytes(b'x');rows=io.files(self.root);(self.root/'extra').write_bytes(b'e')
        with self.assertRaises(ValueError):io.payload(self.root,rows)
    def test_payload_rejects_symlink(self):
        (self.root/'link').symlink_to(self.root/'missing')
        with self.assertRaises(ValueError):io.files(self.root)
    def test_child_traversal(self):
        for x in ('../out','/tmp/out','a/../../x'):
            with self.subTest(x=x),self.assertRaises(ValueError):io.child(self.root,x)
    def test_child_internal_symlink_rejected(self):
        (self.root/'actual').mkdir();(self.root/'alias').symlink_to(self.root/'actual')
        with self.assertRaises(ValueError):io.child(self.root,'alias/x')

    def fixture(self):
        p=self.root/'bundle';p.mkdir();s={'case':'synthetic','reference_only':False}
        (p/'input').mkdir();(p/'input/raw.npy').write_bytes(b'x')
        contract={'case':'synthetic','catalog_sha256':'pinned','files':io.files(p)}
        io.write(p/'MANIFEST.json',contract);io.write(p/'COMPLETE.json',{'status':'complete','manifest_sha256':io.record(p/'MANIFEST.json')['sha256']})
        return p,s,contract

    def test_bundle_rehash_not_authority(self):
        p,s,c=self.fixture();changed=copy.deepcopy(c);changed['files']['input/raw.npy']=dict(bytes=1,sha256='0'*64)
        (p/'MANIFEST.json').write_text(json.dumps(changed));(p/'COMPLETE.json').write_text(json.dumps({'status':'complete','manifest_sha256':io.record(p/'MANIFEST.json')['sha256']}))
        with patch.object(study,'plan',return_value=c),patch.object(inv,'spec',return_value=s),self.assertRaises(ValueError):study.verify_bundle(p)
    def test_incomplete_bundle(self):
        p,s,c=self.fixture();(p/'COMPLETE.json').write_text(json.dumps({'status':'partial','manifest_sha256':io.record(p/'MANIFEST.json')['sha256']}))
        with patch.object(study,'plan',return_value=c),patch.object(inv,'spec',return_value=s),self.assertRaises(ValueError):study.verify_bundle(p)
    def test_move_bundle(self):
        p,s,c=self.fixture();q=self.root/'moved bundle';shutil.move(p,q)
        with patch.object(study,'plan',return_value=c),patch.object(inv,'spec',return_value=s):self.assertEqual(study.verify_bundle(q),s)
    def test_bm_fit_refused_before_reserve(self):
        with patch.object(study,'verify_bundle',return_value={'reference_only':True}),self.assertRaises(ValueError):study.fit(self.root,self.root/'never')
        self.assertFalse((self.root/'never').exists())
    def test_batch_case_swap_before_reserve(self):
        with patch.object(study,'verify_bundle',return_value={'case':'wrong'}),self.assertRaises(ValueError):study.batch(self.root,self.root/'never')
        self.assertFalse((self.root/'never').exists())
    def test_code_change_refused(self):
        r=study.code_records();r['worker.py']['sha256']='0'*64
        with self.assertRaises(ValueError):study.check_code(r)
    def test_environment_scrub(self):
        with patch.dict(os.environ,{'PYTHONPATH':'bad','PYTHONHOME':'bad','OMP_NUM_THREADS':'7'}):e=study.child_environment(self.root)
        self.assertNotIn('PYTHONPATH',e);self.assertNotIn('PYTHONHOME',e)
        self.assertEqual(e['OMP_NUM_THREADS'],'1');self.assertEqual(e['PYTHONDONTWRITEBYTECODE'],'1')

    def test_subprocess_failure_keeps_logs_no_complete(self):
        p,s,c=self.fixture();out=self.root/'run'
        with patch.object(study,'verify_bundle',return_value=s),patch.object(study,'plan',return_value=c),patch.object(study.subprocess,'run',return_value=types.SimpleNamespace(returncode=7)):
            with self.assertRaises(RuntimeError):study.fit(p,out)
        self.assertTrue((out/'FAILED.json').exists());self.assertTrue((out/'REQUEST.json').exists())
        self.assertTrue((out/'stdout.log').exists());self.assertFalse((out/'COMPLETE.json').exists())
        with self.assertRaises(FileExistsError):io.reserve(out)

    def test_output_validation_failure_no_complete(self):
        p,s,c=self.fixture();out=self.root/'run'
        with patch.object(study,'verify_bundle',return_value=s),patch.object(study,'plan',return_value=c),patch.object(study.subprocess,'run',return_value=types.SimpleNamespace(returncode=0)),patch.object(study,'validate_outputs',side_effect=ValueError('invalid outputs')):
            with self.assertRaises(ValueError):study.fit(p,out)
        self.assertTrue((out/'FAILED.json').exists());self.assertFalse((out/'COMPLETE.json').exists())

    def test_worker_reentry_refused(self):
        p,s,c=self.fixture();out=self.root/'worker';out.mkdir()
        req={'run':str(out),'bundle':str(p),'case':'synthetic','code_records':study.code_records(),'bundle_manifest':io.record(p/'MANIFEST.json'),'python':io.record(sys.executable),'thread_environment':worker.thread_values('synthetic')}
        io.write(out/'REQUEST.json',req);io.write(out/'STARTED.json',{'owner':'first'})
        with patch.object(study,'verify_bundle',return_value=s),patch.object(study,'check_runtime'),self.assertRaises(FileExistsError):worker.execute(out)
        self.assertEqual(io.read(out/'STARTED.json'),{'owner':'first'})

    def test_worker_wrong_request_path_refused(self):
        out=self.root/'worker';out.mkdir();io.write(out/'REQUEST.json',{'run':str(self.root/'wrong')})
        with self.assertRaises(ValueError):worker.execute(out)
        self.assertFalse((out/'STARTED.json').exists())

    def test_prepare_source_tamper_no_complete(self):
        source=self.root/'sources';(source/'synthetic').mkdir(parents=True)
        path=source/'synthetic/raw.npy';path.write_bytes(b'input')
        c={'case':'synthetic','files':{'input/raw.npy':io.record(path)}}
        original=io.copy
        def mutate(src,target,row):
            original(src,target,row);path.write_bytes(b'changed')
        with patch.object(inv,'spec',return_value={'case':'synthetic'}),patch.object(study,'plan',return_value=c),patch.object(io,'copy',side_effect=mutate),self.assertRaises(ValueError):study.prepare('synthetic',source,self.root/'out')
        self.assertTrue((self.root/'out/FAILED.json').exists());self.assertFalse((self.root/'out/COMPLETE.json').exists())


class ScientificTests(unittest.TestCase):
    def setUp(self):
        self.md=pd.DataFrame({'cell_id':['a','b','c','d'],'dataset':['A','A','B','B'],'unit_key':['A1','A1','B1','B1']})
        self.x=np.array([[0.,2.],[0.,3.],[-1.,2.],[1.,3.]])
        self.spec={'case':'tnbc','shape':[4,2],'input_dtype':'float64','fit_columns':list(self.md.columns),'datasets':['A','B'],'dataset_cells':{'A':2,'B':2},'dataset_units':{'A':1,'B':1},'markers':['M1','M2'],'family':'family','pair_fit':{'minimum':4000,'per_unit':100,'maximum':8000}}
    def test_signed_and_zero_input_retained(self):
        fit=worker.validate_input(self.x,self.md,['M1','M2'],self.spec)
        self.assertEqual(list(fit.columns),['dataset','unit_key']);self.assertEqual(self.x[2,0],-1)
    def test_extra_labels_rejected(self):
        md=self.md.assign(celltype='T')
        with self.assertRaises(ValueError):worker.validate_input(self.x,md,['M1','M2'],self.spec)
    def test_marker_order_rejected(self):
        with self.assertRaises(ValueError):worker.validate_input(self.x,self.md,['M2','M1'],self.spec)
    def test_dataset_order_rejected(self):
        with self.assertRaises(ValueError):worker.validate_input(self.x,self.md.iloc[::-1],['M1','M2'],self.spec)
    def test_cross_dataset_unit_rejected(self):
        md=self.md.copy();md['unit_key']='same'
        with self.assertRaises(ValueError):worker.validate_input(self.x,md,['M1','M2'],self.spec)
    def test_duplicate_cell_rejected(self):
        md=self.md.copy();md.loc[1,'cell_id']='a'
        with self.assertRaises(ValueError):worker.validate_input(self.x,md,['M1','M2'],self.spec)
    def test_nan_rejected(self):
        x=self.x.copy();x[0,0]=np.nan
        with self.assertRaises(ValueError):worker.validate_input(x,self.md,['M1','M2'],self.spec)
    def test_dtype_rejected(self):
        with self.assertRaises(ValueError):worker.validate_input(self.x.astype('float32'),self.md,['M1','M2'],self.spec)
    def test_invariants_pass(self):self.assertEqual(worker.independent_invariants(self.x,self.x,self.x,self.md)['strict_inversions'],0)
    def test_inversion_rejected(self):
        y=self.x.copy();y[:2,1]=[3,2]
        with self.assertRaises(ValueError):worker.independent_invariants(self.x,y,y,self.md)
    def test_split_tie_rejected(self):
        y=self.x.copy();y[0,0]=0.1
        with self.assertRaises(ValueError):worker.independent_invariants(self.x,self.x,y,self.md)
    def test_displacement_rejected(self):
        with self.assertRaises(ValueError):worker.independent_invariants(self.x,self.x,self.x+0.8,self.md)
    def fake(self,fail=False):
        def fit(raw,md,markers,**kw):
            self.assertTrue(np.array_equal(raw,self.x));self.assertEqual(kw,{'datasets':['A','B'],'family':'family'})
            self.assertEqual(markers,['M1','M2']);self.assertEqual(list(md.columns),['dataset','unit_key'])
            if fail:raise RuntimeError('intentional')
            return 'fitted'
        return types.SimpleNamespace(PAIR_FIT_MINIMUM=4000,PAIR_FIT_PER_UNIT=100,PAIR_FIT_MAXIMUM=8000,apply_monotone_profiles=lambda b,m,p:b,fit_transform=fit)
    def test_call_no_transformation(self):
        r=self.fake();result,captured=worker.calculate(self.x,self.md[['dataset','unit_key']],self.spec,r)
        self.assertEqual(result,'fitted');self.assertEqual(captured,{})
    def test_pair_override_restored_after_failure(self):
        s=copy.deepcopy(self.spec);s['pair_fit']={'minimum':7000,'per_unit':0,'maximum':7000};r=self.fake(True)
        with self.assertRaises(RuntimeError):worker.calculate(self.x,self.md[['dataset','unit_key']],s,r)
        self.assertEqual([r.PAIR_FIT_MINIMUM,r.PAIR_FIT_PER_UNIT,r.PAIR_FIT_MAXIMUM],[4000,100,8000])
    def test_runtime_wrong_threads_rejected(self):
        with patch.dict(os.environ,{'OMP_NUM_THREADS':'2'}),self.assertRaises(ValueError):worker.environment()

    def test_story_default_environment_scrubs_inherited_caps(self):
        with patch.dict(os.environ,{k:'3' for k in worker.THREADS}):
            env=study.child_environment(Path('/tmp/new-diagnostic'),'melanoma_story')
        self.assertTrue(all(k not in env for k in worker.THREADS))
        self.assertEqual(env['PYTHONDONTWRITEBYTECODE'],'1')

    def test_other_cases_remain_single_thread(self):
        for case in inv.CASES:
            if case!='melanoma_story':
                env=study.child_environment(Path('/tmp/new-diagnostic'),case)
                self.assertEqual({k:env[k] for k in worker.THREADS},{k:'1' for k in worker.THREADS})

    def test_story_default_host_accepted(self):
        worker.check_thread_policy('melanoma_story',{k:None for k in worker.THREADS},[{'user_api':'openmp','num_threads':14}])

    def test_story_changed_thread_cap_rejected(self):
        with self.assertRaises(ValueError):worker.check_thread_policy('melanoma_story',{k:'1' for k in worker.THREADS},[{'user_api':'openmp','num_threads':14}])

    def test_story_other_host_not_claimed(self):
        with self.assertRaises(ValueError):worker.check_thread_policy('melanoma_story',{k:None for k in worker.THREADS},[{'user_api':'openmp','num_threads':8}])

    def test_single_thread_effective_runtime_rejected(self):
        with self.assertRaises(ValueError):worker.check_thread_policy('tnbc',{k:'1' for k in worker.THREADS},[{'user_api':'openmp','num_threads':14}])


if __name__=='__main__':unittest.main()
