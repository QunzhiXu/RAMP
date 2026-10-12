"""Control-flow tests use mocked subprocesses; real numerical evidence is separate."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from fixtures import fixture, io, inv, study
import execution as ex
import compare_execution as comparison


def env_record(kind):
    categories = ('LC_COLLATE','LC_CTYPE','LC_MONETARY','LC_NUMERIC','LC_TIME','LC_MESSAGES')
    return {'environment':ex.LOCALE,'locale':{k:'C' if k=='LC_NUMERIC' else 'C.UTF-8' for k in categories},
            'runtime_lock':io.table(inv.SNAPSHOT/'adapter/runtime_lock.csv').to_dict('records'),
            'extra_serialization':{'data.table':'1.17.8'} if kind=='story' else None}


class ExecutionSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='fastmnn execution '); self.root = Path(self.tmp.name).resolve()
        self.library = self.root/'library/.Rlib'; self.library.mkdir(parents=True)
        self.rscript = self.root/'Rscript'; self.rscript.write_text('mock executable')
        self.scratch = Path(tempfile.gettempdir()).resolve()
    def tearDown(self): self.tmp.cleanup()

    def exercise(self, mutation=None, expected_error=None):
        with fixture(self.root) as (source,spec):
            bundle = study.prepare(source,self.root/'bundle',spec['case']); out = self.root/'fit'
            def command(argv, root, name, env):
                self.assertEqual({k:env[k] for k in ex.LOCALE},ex.LOCALE)
                self.assertFalse(any('alignment/data' in str(v) for v in argv))
                (root/(name+'.stderr.log')).write_text('')
                if name.startswith('environment'):
                    io.create_json(root/(name+'.stdout.log'),env_record('technical'))
                else:
                    (root/(name+'.stdout.log')).write_text('mocked; no fit')
                    (root/'result').mkdir(); io.create_json(root/'result/COMPLETE.json',{'status':'complete'})
                if mutation: mutation(name,root,bundle)
            with patch.object(ex,'run_command',side_effect=command),patch.object(ex,'check_output',return_value={'mocked':True}):
                if expected_error:
                    with self.assertRaises(expected_error): ex.fit(bundle,self.library,self.rscript,out,self.scratch)
                    self.assertTrue((out/'FAILED.json').exists()); self.assertFalse((out/'COMPLETE.json').exists())
                    # A corrupted input may be rejected even before the occupied
                    # output guard; either way the partial output is not reused.
                    with self.assertRaises((FileExistsError,ValueError)):
                        ex.fit(bundle,self.library,self.rscript,out,self.scratch)
                else:
                    ex.fit(bundle,self.library,self.rscript,out,self.scratch)
                    self.assertEqual(ex.verify_run(out,bundle),{'mocked':True})
                    moved = self.root/'moved bundle'; shutil.copytree(bundle,moved)
                    self.assertEqual(ex.verify_run(out,moved),{'mocked':True})
                    codes = ex.code_records()
                    with patch.object(ex,'HERE',self.root/'moved controller'),patch.object(ex,'code_records',return_value=codes):
                        self.assertEqual(ex.verify_run(out,moved),{'mocked':True})
                    with self.assertRaises(FileExistsError): ex.fit(bundle,self.library,self.rscript,out,self.scratch)
            return out

    def test_success_and_moved_bundle(self): self.exercise()

    def test_subprocess_failure_keeps_nested_not_outer_complete(self):
        def mutate(name,root,bundle):
            if name=='correction': raise RuntimeError('mocked R failure')
        out = self.exercise(mutate,RuntimeError)
        self.assertTrue((out/'result/COMPLETE.json').exists())

    def test_input_changed_during_fit(self):
        def mutate(name,root,bundle):
            if name=='correction':
                spec,_ = study.plan('synthetic_technical'); (bundle/'inputs'/spec['input']).write_bytes(b'changed')
        self.exercise(mutate,ValueError)

    def test_bundle_manifest_changed_during_fit(self):
        def mutate(name,root,bundle):
            if name=='correction': (bundle/'MANIFEST.json').write_text('{}')
        self.exercise(mutate,ValueError)

    def test_executable_changed_during_fit(self):
        def mutate(name,root,bundle):
            if name=='correction': self.rscript.write_text('changed')
        self.exercise(mutate,ValueError)

    def test_runtime_adapter_changed(self):
        def mutate(name,root,bundle):
            if name=='correction': (root/'runtime/tree/algorithm/fastmnn/fastmnn_repro.R').write_text('changed')
        self.exercise(mutate,ValueError)

    def test_environment_changed(self):
        def mutate(name,root,bundle):
            if name=='environment_after':
                p=root/'environment_after.stdout.log'; value=io.read_json(p);value['changed']=True;p.write_text(json.dumps(value))
        self.exercise(mutate,ValueError)

    def test_locale_wrong(self):
        def mutate(name,root,bundle):
            if name=='environment_before':
                p=root/'environment_before.stdout.log'; value=io.read_json(p);value['environment']={};p.write_text(json.dumps(value))
        self.exercise(mutate,ValueError)

    def test_runtime_link_changed(self):
        def mutate(name,root,bundle):
            if name=='correction':
                p=root/'runtime/tree/algorithm/fastmnn/.Rlib';p.unlink();p.symlink_to(self.root)
        self.exercise(mutate,ValueError)

    def test_code_change_rejected(self):
        # Simulate a code mutation via the hash guard; do not edit actual source.
        original = io.check
        def check(path,row):
            if Path(path)==ex.HERE/'execution.py': raise ValueError('code changed')
            return original(path,row)
        with patch.object(io,'check',side_effect=check): self.exercise(expected_error=ValueError)

    def test_extra_symlink_rejected(self):
        def mutate(name,root,bundle):
            if name=='correction': (root/'unexpected').symlink_to(self.rscript)
        self.exercise(mutate,ValueError)

    def test_space_scratch_rejected_before_claim(self):
        scratch=self.root/'with space';scratch.mkdir()
        with fixture(self.root) as (source,spec):
            bundle=study.prepare(source,self.root/'bundle',spec['case'])
            with self.assertRaises(ValueError): ex.fit(bundle,self.library,self.rscript,self.root/'fit',scratch)
            self.assertFalse((self.root/'fit').exists())

    def test_incomplete_run_rejected(self):
        with fixture(self.root) as (source,spec):
            bundle=study.prepare(source,self.root/'bundle',spec['case']); out=self.root/'fit';out.mkdir()
            with self.assertRaises(FileNotFoundError): ex.verify_run(out,bundle)

    def test_output_contract_failure_blocks_completion(self):
        with patch.object(ex,'check_output',side_effect=ValueError('dimensions differ')):
            # Direct call instead of exercise, which intentionally mocks check_output.
            with fixture(self.root) as (source,spec):
                bundle=study.prepare(source,self.root/'bundle',spec['case']);out=self.root/'fit'
                def command(argv,root,name,env):
                    (root/(name+'.stderr.log')).write_text('')
                    io.create_json(root/(name+'.stdout.log'),env_record('technical'))
                with patch.object(ex,'run_command',side_effect=command), self.assertRaises(ValueError):
                    ex.fit(bundle,self.library,self.rscript,out,self.scratch)
                self.assertTrue((out/'FAILED.json').exists());self.assertFalse((out/'COMPLETE.json').exists())


class Contracts(unittest.TestCase):
    def test_all_seven_command_routes(self):
        for spec in inv.verify_local()['branches']:
            cmd=ex.command_plan(Path('/owned'),spec,'/Rscript')['correction']
            if spec['kind']=='technical':
                self.assertIn('/owned/runtime/tree/'+spec['config'],cmd)
                self.assertIn('--input',cmd)
            else:
                self.assertEqual(cmd[-1],'fastmnn'); self.assertTrue(cmd[2].endswith('/scripts/run_r.R'))
            self.assertFalse(any('analysis_ready' not in v and 'raw.csv' in v for v in cmd))

    def test_configs_have_unchanged_relative_records(self):
        for spec in inv.verify_local()['branches']:
            if spec['kind']!='technical': continue
            config=inv.SNAPSHOT/'study_sources'/spec['config']; c=io.read_json(config)
            relative=Path(c['input_scale']['preprocessing_record_file'])
            self.assertFalse(relative.is_absolute())
            self.assertEqual((config.parent/relative).resolve(),(inv.SNAPSHOT/'study_sources'/spec['record']).resolve())

    def test_runtime_maps_every_file_without_collisions(self):
        for spec in inv.verify_local()['branches']:
            _,rows=study.plan(spec['case']);paths=[ex.runtime_path(r,spec) for r in rows]
            self.assertEqual(len(paths),len(set(paths)))
            self.assertTrue(all(str(p).startswith('runtime/tree/') for p in paths))

    def test_runtime_rejects_unknown_payload(self):
        with self.assertRaises(ValueError): ex.runtime_path({'path':'fitted/values.npy'},{'kind':'story'})

    def test_embedding_not_marker_values(self):
        for spec in inv.verify_local()['branches']:
            self.assertEqual(len(spec['embedding_columns']),spec['dimensions'])
            self.assertEqual(spec['dimensions'],len(spec['markers'])-1)

    def test_case_specific_error_and_scientific_elapsed_rejection(self):
        a={'elapsed_seconds':1,'parameters':{'seed':42}}; b=copy.deepcopy(a); b['elapsed_seconds']=2
        self.assertEqual(comparison.elapsed(a,b)['current_seconds'],2)
        b['parameters']['seed']=1
        with self.assertRaises(ValueError): comparison.elapsed(a,b)

    def test_invalid_elapsed_rejected(self):
        for x in (-1,float('nan'),float('inf'),True,'1'):
            with self.subTest(x=x),self.assertRaises(ValueError): comparison.elapsed({'elapsed_seconds':1},{'elapsed_seconds':x})

    def test_nested_elapsed_only(self):
        a={'runtime':{'elapsed_seconds':1},'values_sha256':'same'};b=copy.deepcopy(a);b['runtime']['elapsed_seconds']=9
        comparison.elapsed(a,b,True);b['values_sha256']='different'
        with self.assertRaises(ValueError): comparison.elapsed(a,b,True)

    def test_sessions_allow_only_macos_line(self):
        a='R 4.3.2\nRunning under: macOS 26.6.1\nlocale C.UTF-8\n'
        comparison.sessions(a,a.replace('26.6.1','26.7.1'))
        with self.assertRaises(ValueError): comparison.sessions(a,a.replace('4.3.2','4.4.0'))
        with self.assertRaises(ValueError): comparison.sessions(a,a.replace('macOS','Linux'))

    def test_config_changes_rejected(self):
        a={'input_scale':{'preprocessing_record_file':'old','status':'analysis_ready'},'seed':42};b=copy.deepcopy(a)
        b['input_scale']['preprocessing_record_file']='new';comparison.resolved_config(a,b)
        b['seed']=43
        with self.assertRaises(ValueError): comparison.resolved_config(a,b)

    def test_manifest_only_two_payload_hashes_normalized(self):
        spec,_=study.plan('bone_marrow');a=io.read_json(inv.SNAPSHOT/'study_sources'/spec['output_manifest']);b=copy.deepcopy(a)
        b['created_at']='later';b['input']['path']='new';b['implementation']['wrapper_file']['path']='new'
        for row in b['output_files']:
            if row['file']=='sessionInfo.txt':row['sha256']='new'
        comparison.technical_manifest(a,b)
        b['output_files'][0]['sha256']='different'
        with self.assertRaises(ValueError): comparison.technical_manifest(a,b)

    def test_manifest_algorithm_change_rejected(self):
        spec,_=study.plan('bone_marrow');a=io.read_json(inv.SNAPSHOT/'study_sources'/spec['output_manifest']);b=copy.deepcopy(a)
        b['algorithm']['resolved_parameters']['seed']=43
        with self.assertRaises(ValueError): comparison.technical_manifest(a,b)

    def test_imports_are_safe(self):
        code="import sys;sys.path.insert(0,sys.argv[1]);import execution,compare_execution;print('safe')"
        p=subprocess.run([sys.executable,'-B','-c',code,str(inv.HERE)],capture_output=True,text=True,check=True)
        self.assertEqual(p.stdout.strip(),'safe')


if __name__=='__main__': unittest.main()
