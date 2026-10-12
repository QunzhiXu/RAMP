import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from fixtures import fixture,io,inv,study
from probe_runtime import environment


class Safety(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory(prefix='fastmnn test ');self.root=Path(self.tmp.name).resolve()
    def tearDown(self):self.tmp.cleanup()

    def test_existing_output(self):
        out=self.root/'out';out.mkdir();(out/'old').write_text('kept')
        with self.assertRaises(FileExistsError):io.reserve(out)
        self.assertEqual((out/'old').read_text(),'kept')

    def test_dangling_symlink(self):
        out=self.root/'out';out.symlink_to(self.root/'missing')
        with self.assertRaises(FileExistsError):io.reserve(out)

    def test_symlink_parent_overlap(self):
        protected=self.root/'source';protected.mkdir();alias=self.root/'alias';alias.symlink_to(protected,target_is_directory=True)
        with self.assertRaises(ValueError):io.reserve(alias/'output',(protected,))

    def test_ancestor_overlap(self):
        with self.assertRaises(ValueError):io.reserve(self.root/'new',(self.root/'new/input',))

    def test_hardlink_not_overwritten(self):
        src=self.root/'src';src.write_text('kept');dest=self.root/'dest';os.link(src,dest)
        with self.assertRaises(FileExistsError):io.copy_new(src,dest,io.record(src))
        self.assertEqual(src.read_text(),'kept')

    def test_path_traversal(self):
        for name in ('../escape','/absolute'):
            with self.subTest(name=name),self.assertRaises(ValueError):io.within(self.root,name)

    def test_escaping_file_symlink(self):
        d=self.root/'dir';d.mkdir();f=self.root/'file';f.write_text('x');(d/'link').symlink_to(f)
        with self.assertRaises(ValueError):io.within(d,'link')

    def test_double_process_reserve(self):
        code='''import sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1]);import cytonorm_io as io
root=Path(sys.argv[2]);name=sys.argv[3];(root/('ready'+name)).write_text('ready')
while not (root/'go').exists():time.sleep(.01)
try:out=io.reserve(root/'out')
except FileExistsError:sys.exit(17)
(out/'winner').write_text(name)
'''
        ps=[subprocess.Popen([sys.executable,'-B','-c',code,str(inv.HERE),str(self.root),str(i)])for i in range(2)]
        try:
            deadline=time.monotonic()+15
            while not all((self.root/('ready'+str(i))).exists()for i in range(2)):
                if time.monotonic()>deadline:self.fail('barrier timeout')
                time.sleep(.01)
            (self.root/'go').write_text('go');self.assertEqual(sorted(p.wait(timeout=15)for p in ps),[0,17])
        finally:
            for p in ps:
                if p.poll() is None:p.kill();p.wait()


class Snapshot(unittest.TestCase):
    def test_seven_branches(self):
        c=inv.verify_local();self.assertEqual({s['case']for s in c['branches']},set(inv.TECHNICAL)|set(inv.STORIES))
        for s in c['branches']:
            self.assertEqual(s['parameters'],inv.PARAMETERS)
            self.assertEqual(s['identity_check']['output_type'],'corrected_marker_values')

    def test_no_correction_or_label_inputs(self):
        for s in inv.verify_local()['branches']:
            spec,files=study.plan(s['case']);source=[r['source']for r in files if r['origin']=='source']
            self.assertFalse(any('/methods/'in p or '/alignment/data/'in p or 'evaluation_metadata'in p for p in source))
            if spec['kind']=='story':self.assertTrue(spec['input'].endswith('/analysis_ready.csv.gz'))

    def test_lung_platform_separation(self):
        a,_=study.plan('lung_story');b,_=study.plan('lung_cancer')
        self.assertIn('Zheng mIF',a['datasets']);self.assertIn('Monkman CODEX',b['datasets']);self.assertNotEqual(a['input'],b['input'])

    def test_unknown_case(self):
        with self.assertRaises(ValueError):study.plan('missing')

    def test_exact_hash_field(self):
        with tempfile.TemporaryDirectory()as name:
            root=Path(name);(root/'data').write_text('x');io.create_json(root/'m.json',{'expected':'0'*64,'note':io.sha(root/'data')})
            with self.assertRaises(ValueError):inv.Builder(root).link('m.json',['expected'],'data')

    def test_rehashed_catalog_not_trusted(self):
        with tempfile.TemporaryDirectory()as name:
            root=Path(name)/'snapshot';shutil.copytree(inv.SNAPSHOT,root);p=root/'catalog.json';m=io.read_json(p);m['branches'][0]['cells']=1;p.write_text(json.dumps(m))
            (root/'COMPLETE.json').write_text(json.dumps({'status':'complete','manifest_sha256':io.sha(p)}))
            with self.assertRaisesRegex(ValueError,'anchor'):inv.verify_local(root)

    def test_registry_change(self):
        with tempfile.TemporaryDirectory()as name:
            p=Path(name)/'reg.json';p.write_text('{}')
            with patch.object(inv,'REGISTRY',p),self.assertRaises(ValueError):inv.registry()

    def test_freeze_registry_mutation_no_complete(self):
        with tempfile.TemporaryDirectory()as name:
            root=Path(name).resolve();source=root/'source';source.mkdir();(source/'script.py').write_text('pass\n')
            reg=root/'registry.json';shutil.copyfile(inv.SNAPSHOT/'registry.json',reg)
            c={'status':'complete','registry':io.record(reg),'files':[{'source':'script.py','copy':'adapter/script.py',**io.record(source/'script.py')}],'links':[],'branches':[]}
            def changed(_):reg.write_text('{}');return c
            with patch.object(inv,'REGISTRY',reg),patch.object(inv,'build',side_effect=changed),self.assertRaises(ValueError):inv.freeze(source,root/'out')
            self.assertFalse((root/'out/COMPLETE.json').exists());self.assertTrue((root/'out/FAILED.json').exists())

    def test_freeze_payload_failure_no_complete(self):
        with tempfile.TemporaryDirectory()as name:
            root=Path(name).resolve();source=root/'source';source.mkdir();reg=root/'registry.json';shutil.copyfile(inv.SNAPSHOT/'registry.json',reg)
            c={'status':'complete','registry':io.record(reg),'files':[],'links':[],'branches':[]}
            with patch.object(inv,'REGISTRY',reg),patch.object(inv,'build',return_value=c),patch.object(inv,'verify_payload',side_effect=ValueError('payload')),self.assertRaises(ValueError):inv.freeze(source,root/'out')
            self.assertFalse((root/'out/COMPLETE.json').exists())

    def test_complete_written_after_payload(self):
        with tempfile.TemporaryDirectory()as name:
            root=Path(name).resolve();source=root/'source';source.mkdir();reg=root/'registry.json';shutil.copyfile(inv.SNAPSHOT/'registry.json',reg)
            c={'status':'complete','registry':io.record(reg),'files':[],'links':[],'branches':[]};seen=[];orig=inv.verify_payload
            def checked(out,catalog):self.assertFalse((out/'COMPLETE.json').exists());orig(out,catalog);seen.append(True)
            with patch.object(inv,'REGISTRY',reg),patch.object(inv,'build',return_value=c),patch.object(inv,'verify_payload',side_effect=checked):inv.freeze(source,root/'out')
            self.assertEqual(seen,[True]);self.assertTrue((root/'out/COMPLETE.json').exists())


class Bundle(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='fastmnn test ');self.root=Path(self.tmp.name).resolve();self.ctx=fixture(self.root)
        self.source,self.spec=self.ctx.__enter__();self.bundle=self.root/'bundle'
    def tearDown(self):self.ctx.__exit__(None,None,None);self.tmp.cleanup()
    def prepare(self):return study.prepare(self.source,self.bundle,self.spec['case'])
    def rewrite(self,edit):
        m=io.read_json(self.bundle/'MANIFEST.json');edit(m);(self.bundle/'MANIFEST.json').write_text(json.dumps(m))
        (self.bundle/'COMPLETE.json').write_text(json.dumps({'status':'complete','manifest_sha256':io.sha(self.bundle/'MANIFEST.json')}))

    def test_moved_without_source(self):
        self.prepare();moved=self.root/'moved';self.bundle.rename(moved);self.source.rename(self.root/'unavailable')
        self.assertEqual(study.verify_bundle(moved)[0]['cells'],36)

    def test_explicit_symlink_tmp(self):
        alias=self.root/'alias';alias.symlink_to(self.root,target_is_directory=True)
        study.prepare(self.source,alias/'bundle',self.spec['case']);study.verify_bundle(self.bundle)

    def test_input_change_before_copy(self):
        (self.source/self.spec['input']).write_bytes(b'bad')
        with self.assertRaises(ValueError):self.prepare()
        self.assertFalse(self.bundle.exists())

    def test_source_change_during_copy(self):
        orig=io.copy_new
        def copied(src,dest,row):
            orig(src,dest,row)
            if src==(self.source/self.spec['input']).resolve():src.write_bytes(b'changed')
        with patch.object(io,'copy_new',side_effect=copied),self.assertRaises(ValueError):self.prepare()
        self.assertFalse((self.bundle/'COMPLETE.json').exists());self.assertTrue((self.bundle/'FAILED.json').exists())

    def test_disk_failure(self):
        with patch.object(io,'copy_new',side_effect=OSError('disk')),self.assertRaises(OSError):self.prepare()
        self.assertFalse((self.bundle/'COMPLETE.json').exists());self.assertTrue((self.bundle/'FAILED.json').exists())

    def test_repeated_prepare_rejected(self):
        self.prepare();before=io.sha(self.bundle/'MANIFEST.json')
        with self.assertRaises(FileExistsError):self.prepare()
        self.assertEqual(io.sha(self.bundle/'MANIFEST.json'),before)

    def test_missing_complete(self):
        self.prepare();(self.bundle/'COMPLETE.json').unlink()
        with self.assertRaises(FileNotFoundError):study.verify_bundle(self.bundle)

    def test_partial_status(self):
        self.prepare();(self.bundle/'COMPLETE.json').write_text(json.dumps({'status':'partial','manifest_sha256':io.sha(self.bundle/'MANIFEST.json')}))
        with self.assertRaises(ValueError):study.verify_bundle(self.bundle)

    def test_rehashed_partial_manifest(self):
        self.prepare();self.rewrite(lambda m:m.update(status='partial'))
        with self.assertRaises(ValueError):study.verify_bundle(self.bundle)

    def test_rehashed_input(self):
        self.prepare();name='inputs/'+self.spec['input'];(self.bundle/name).write_bytes(b'bad')
        self.rewrite(lambda m:[r.update(io.record(self.bundle/name))for r in m['files']if r['path']==name])
        with self.assertRaisesRegex(ValueError,'contract'):study.verify_bundle(self.bundle)

    def test_extra_file(self):
        self.prepare();(self.bundle/'extra').write_text('x')
        with self.assertRaises(ValueError):study.verify_bundle(self.bundle)

    def test_symlink_payload(self):
        self.prepare();p=self.bundle/'adapter/README.md';p.unlink();p.symlink_to(inv.SNAPSHOT/'adapter/README.md')
        with self.assertRaises(ValueError):study.verify_bundle(self.bundle)

    def test_source_output_overlap(self):
        with self.assertRaises(ValueError):study.prepare(self.source,self.source/'out',self.spec['case'])

    def test_counts_not_self_reported(self):
        self.prepare();self.rewrite(lambda m:m['counts'].update(cells=1))
        with self.assertRaises(ValueError):study.verify_bundle(self.bundle)

    def test_zero_cells_retained(self):
        self.prepare();self.assertEqual(study.validate_inputs(self.bundle,self.spec)['zero_l2_cells'],1)

    def test_nonfinite_values_rejected(self):
        import pandas as pd
        self.prepare();p=self.bundle/'inputs'/self.spec['input'];df=io.table(p);df.loc[0,self.spec['markers'][0]]=float('inf');df.to_csv(p,index=False)
        with self.assertRaisesRegex(ValueError,'Nonfinite'):study.validate_inputs(self.bundle,self.spec)

    def test_wrong_dataset_order(self):
        self.prepare();spec=copy.deepcopy(self.spec);spec['datasets'].reverse()
        with self.assertRaises(ValueError):study.validate_inputs(self.bundle,spec)


class Story(unittest.TestCase):
    def check(self,n):
        with tempfile.TemporaryDirectory()as name:
            root=Path(name).resolve()
            with fixture(root,'story',n)as(source,spec):
                bundle=study.prepare(source,root/'bundle',spec['case']);_,files=study.verify_bundle(bundle)
                self.assertEqual(len(files),len(study.plan(spec['case'])[1]));self.assertEqual(study.validate_inputs(bundle,spec)['cells'],12*n)
                self.assertTrue(any(r['path']=='scripts/run_r.R'for r in files))
    def test_two_platforms(self):self.check(2)
    def test_three_platforms(self):self.check(3)


class Locale(unittest.TestCase):
    def test_C_parent(self):
        parent={'LC_ALL':'C','LANG':'C'};env=environment(parent);self.assertEqual(parent['LC_ALL'],'C');self.assertEqual(env['LC_ALL'],'C.UTF-8')
    def test_unset_parent(self):
        env=environment({'R_LIBS':'uncontrolled'});self.assertEqual(env['LANG'],'C.UTF-8');self.assertNotIn('R_LIBS',env)

    def test_user_site_overrides_removed(self):
        env=environment({'R_LIBS':'a','R_LIBS_USER':'b','R_LIBS_SITE':'c'})
        for key in ('R_LIBS','R_LIBS_USER','R_LIBS_SITE'):self.assertNotIn(key,env)


if __name__=='__main__':unittest.main()
