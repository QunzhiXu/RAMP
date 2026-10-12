"""Execution safety/contracts only; mocked subprocess tests are not numerical evidence."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import execution as ex
import cytonorm_io as io
import inventory as inv
import study
from fixtures import fixture
from probe_runtime import runtime_contract


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
        self.library=self.root/'runtime/.Rlib';self.library.mkdir(parents=True)
        io.copy_new(inv.SNAPSHOT/'environment/runtime_lock.csv',self.library/'runtime_lock.csv',
                    io.record(inv.SNAPSHOT/'environment/runtime_lock.csv'))
        self.rscript=self.root/'Rscript';self.rscript.write_text('fake executable')
        self.pins={};self.pin_patch=patch.object(ex,'accepted_bundle_manifests',return_value=self.pins);self.pin_patch.start()

    def tearDown(self): self.pin_patch.stop();self.tmp.cleanup()

    def environment(self):
        contract=runtime_contract();locale={k:'C.UTF-8' for k in ('LC_COLLATE','LC_CTYPE','LC_MONETARY','LC_NUMERIC','LC_TIME','LC_MESSAGES')};locale['LC_NUMERIC']='C'
        return {'R':'4.3.2','locale':locale,'environment':ex.LOCALE,'runtime_versions':contract['versions'],
            'runtime_lock_sha256':contract['lock']['sha256'],'upstream_lock_sha256':io.sha(inv.SNAPSHOT/'adapter/upstream_lock.json')}

    def prepare(self,source,spec):
        bundle=self.root/'bundle';study.prepare(source,bundle,spec['case'])
        self.pins[spec['case']]=io.sha(bundle/'MANIFEST.json');return bundle

    def fake(self,spec,bundle):
        def command(argv,root,name,env):
            self.assertEqual(env['LC_ALL'],'C.UTF-8');self.assertEqual(env['LANG'],'C.UTF-8')
            io.create_json(root/(name+'.stdout.log'),self.environment() if name.startswith('environment') else {'mock':True})
            (root/(name+'.stderr.log')).write_text('')
            if name!='correction':return
            result=root/ex.result_path(spec);result.mkdir(parents=True)
            inp=io.table(bundle/'inputs'/spec['input']);identity=inp.iloc[:,:3]
            inp[['cell_id',*spec['markers']]].to_csv(result/'corrected_expression.csv.gz',index=False)
            identity.to_csv(result/'row_metadata.csv.gz',index=False)
            identity.assign(cluster=1).to_csv(result/'cell_clusters.csv.gz',index=False)
            identity.to_csv(result/'aggregate_cells.csv.gz',index=False)
            for n in ex.CORE_FILES:
                if not (result/n).exists():(result/n).write_text('mock only')
            (result/'sessionInfo.txt').write_text('mock only')
            if spec['kind']=='technical':
                io.create_json(result/'integrity.json',spec['integrity'])
                shutil.copyfile(bundle/'inputs'/spec['marker_file'],result/'markers.txt')
                io.create_json(result/'resolved_config.json',io.read_json(root/'runtime_config.json'))
                (result/'warnings.txt').write_text('')
                io.create_json(result/'COMPLETE.json',{'status':'COMPLETE','outputs':{p.name:io.sha(p) for p in result.iterdir()}})
            else:
                io.create_json(result/'R_COMPLETE.json',{'method':'cytonorm','input_sha256':io.sha(bundle/'inputs'/spec['input']),
                    'helper_sha256':io.sha(bundle/'scripts/cytonorm_repro.R'),'parameters':inv.PARAMETERS,
                    'result':spec['integrity'],'elapsed_seconds':1,'output_sha256':io.sha(result/'corrected_expression.csv.gz')})
        return command

    def run_mock(self,kind='technical',alter=None):
        with fixture(self.root,kind) as (source,spec):
            bundle=self.prepare(source,spec);fn=self.fake(spec,bundle)
            def command(*args):
                fn(*args)
                if alter:alter(*args)
            with patch.object(ex,'run_command',side_effect=command):
                run=ex.fit(bundle,self.library,self.rscript,self.root/'run',self.root)
            return ex.verify_run(run,bundle)

    def test_technical_mock_control_flow(self):self.assertEqual(self.run_mock()['cells'],36)
    def test_story_mock_control_flow(self):self.assertEqual(self.run_mock('story')['cells'],36)
    def test_runtime_mapping_technical(self):
        s,files=study.plan('bone_marrow')
        self.assertEqual(len({ex.runtime_path(r,s) for r in files}),len(files))
        self.assertTrue(str(ex.result_path(s)).startswith('runtime/tree/algorithm_test/'))
    def test_runtime_mapping_seven(self):
        for s in inv.verify_local()['branches']:
            _,files=study.plan(s['case']);self.assertEqual(len({ex.runtime_path(r,s) for r in files}),len(files))
            commands=ex.command_plan(self.root,s,self.rscript)
            self.assertNotIn('raw.csv.gz',' '.join(commands['correction']))
            self.assertNotIn('final_xu',' '.join(commands['correction']))
    def test_unknown_runtime_file(self):
        with self.assertRaises(ValueError):ex.runtime_path({'path':'unknown'}, {'kind':'technical'})
    def test_relocation_only_one_field(self):
        with fixture(self.root) as (source,s):
            b=self.prepare(source,s);old=io.read_json(b/'inputs'/s['config']);new=ex.relocated_config(self.root,b,s)
            changed=[k for k in old if old[k]!=new[k]];self.assertEqual(changed,['preprocessing_file'])
            self.assertEqual(new['preprocessing_sha256'],io.sha(b/'inputs'/s['input_manifest']))
    def test_technical_command_uses_runtime_config(self):
        s,_=study.plan('bone_marrow');c=ex.command_plan(self.root,s,self.rscript)['correction']
        self.assertEqual(c[c.index('--config')+1],str(self.root/'runtime_config.json'))
    def test_story_command_uses_original_runner(self):
        s,_=study.plan('hnscc_story');c=ex.command_plan(self.root,s,self.rscript)['correction']
        self.assertEqual(c[-1],'cytonorm');self.assertTrue(c[2].endswith('/scripts/run_r.R'))
    def test_changed_batch_controller_rejected_before_output(self):
        with fixture(self.root) as (source,s):
            b=self.prepare(source,s)
            with self.assertRaisesRegex(ValueError,'Controller changed'):
                ex.fit(b,self.library,self.rscript,self.root/'run',self.root,expected_codes={})
            self.assertFalse((self.root/'run').exists())
    def test_existing_output_rejected(self):
        with fixture(self.root) as (source,s):
            b=self.prepare(source,s);(self.root/'run').mkdir()
            with self.assertRaises(FileExistsError):ex.fit(b,self.library,self.rscript,self.root/'run',self.root)
    def test_self_rehashed_source_root_refused_before_output(self):
        with fixture(self.root) as (source,s):
            b=self.prepare(source,s);p=b/'MANIFEST.json';v=io.read_json(p);v['source_root_at_preparation']=str(self.root/'false_source')
            p.write_text(json.dumps(v));(b/'COMPLETE.json').write_text(json.dumps({'status':'complete','manifest_sha256':io.sha(p)}))
            with self.assertRaisesRegex(ValueError,'audited phase-1'):
                ex.fit(b,self.library,self.rscript,source/'nested/run',self.root)
            self.assertFalse((source/'nested').exists())
    def test_rehashed_source_root_refused_by_batch(self):
        with fixture(self.root) as (source,s):
            b=self.prepare(source,s);bundles=self.root/'bundles';bundles.mkdir();target=bundles/(s['case']+'_inputs_20261011');b.rename(target)
            p=target/'MANIFEST.json';v=io.read_json(p);v['source_root_at_preparation']=str(self.root/'false_source');p.write_text(json.dumps(v))
            (target/'COMPLETE.json').write_text(json.dumps({'status':'complete','manifest_sha256':io.sha(p)}))
            with self.assertRaisesRegex(ValueError,'audited phase-1'):
                ex.batch(bundles,self.library,self.rscript,source/'nested/run',self.root)
            self.assertFalse((source/'nested').exists())
    def test_output_under_source_rejected(self):
        with fixture(self.root) as (source,s):
            b=self.prepare(source,s)
            with self.assertRaises(ValueError):ex.fit(b,self.library,self.rscript,source/'nested/run',self.root)
            self.assertFalse((source/'nested').exists())
    def test_space_scratch_rejected_before_output(self):
        with fixture(self.root) as (source,s):
            b=self.prepare(source,s);scratch=self.root/'space scratch';scratch.mkdir()
            with self.assertRaises(ValueError):ex.fit(b,self.library,self.rscript,self.root/'run',scratch)
            self.assertFalse((self.root/'run').exists())
    def test_truncated_library_lock_rejected_before_output(self):
        with fixture(self.root) as (source,s):
            b=self.prepare(source,s);(self.library/'runtime_lock.csv').write_text('package,version\n')
            with self.assertRaises(ValueError):ex.fit(b,self.library,self.rscript,self.root/'run',self.root)
            self.assertFalse((self.root/'run').exists())
    def assert_failed(self,alter,kind='technical'):
        with self.assertRaises((ValueError,RuntimeError)):self.run_mock(kind,alter)
        self.assertTrue((self.root/'run/FAILED.json').exists());self.assertFalse((self.root/'run/COMPLETE.json').exists())
    def test_subprocess_failure_preserved(self):
        def alter(argv,root,name,env):
            if name=='correction':raise RuntimeError('mock failure')
        self.assert_failed(alter)
    def test_nested_complete_not_enough(self):
        def alter(argv,root,name,env):
            if name=='environment_after':raise RuntimeError('environment unavailable')
        self.assert_failed(alter)
    def test_executable_mutation_rejected(self):
        def alter(argv,root,name,env):
            if name=='environment_after':self.rscript.write_text('changed')
        self.assert_failed(alter)
    def test_runtime_copy_mutation_rejected(self):
        def alter(argv,root,name,env):
            if name=='environment_after':(root/'runtime/tree/algorithm/CytoNorm/README.md').write_text('changed')
        self.assert_failed(alter)
    def test_wrong_environment_rejected(self):
        def alter(argv,root,name,env):
            if name=='environment_after':(root/(name+'.stdout.log')).write_text('{}')
        self.assert_failed(alter)
    def test_wrong_scientific_parameter_rejected(self):
        def alter(argv,root,name,env):
            if name=='environment_after':
                p=root/'runtime_config.json';v=io.read_json(p);v['seed']=43;p.write_text(json.dumps(v))
        self.assert_failed(alter)
    def test_story_extra_result_rejected(self):
        def alter(argv,root,name,env):
            if name=='environment_after':(root/'runtime/tree/story_lung_HLA_ABC/Comparison_senosain140k_20260921/methods/cytonorm/extra').write_text('x')
        self.assert_failed(alter,'story')
    def test_environment_missing_version_rejected(self):
        v=self.environment();v['runtime_versions'].pop(next(iter(v['runtime_versions'])))
        p=self.root/'log.json';io.create_json(p,v)
        with self.assertRaises(ValueError):ex.check_environment_log(p)
    def test_environment_locale_rejected(self):
        v=self.environment();v['locale']['LC_COLLATE']='C';p=self.root/'log.json';io.create_json(p,v)
        with self.assertRaises(ValueError):ex.check_environment_log(p)
    def test_environment_valid(self):
        p=self.root/'log.json';io.create_json(p,self.environment());ex.check_environment_log(p)
    def test_batch_protects_all_sources_and_symlinks(self):
        # Distinct source roots, every case, including a symlink alias; no fit.
        specs=inv.verify_local()['branches'];bundles=self.root/'bundles';bundles.mkdir();sources=[]
        for s in specs:
            source=self.root/('source_'+s['case']);source.mkdir();sources.append(source)
            b=bundles/(s['case']+'_inputs_20261011');b.mkdir();io.create_json(b/'MANIFEST.json',{'source_root_at_preparation':str(source)})
        def verified(p):return next(s for s in specs if p.name==s['case']+'_inputs_20261011'),[]
        with patch.object(ex,'verify_bundle',side_effect=verified),patch.object(ex,'fit') as fit:
            for i,source in enumerate(sources):
                alias=self.root/('alias_'+str(i));alias.symlink_to(source,target_is_directory=True)
                for target in (source,alias):
                    with self.assertRaises(ValueError):ex.batch(bundles,self.library,self.rscript,target/'nested/run',self.root)
                    self.assertFalse((source/'nested').exists())
            fit.assert_not_called()
    def test_batch_case_swap_refused_before_output(self):
        bundles=self.root/'bundles';bundles.mkdir();specs=inv.verify_local()['branches']
        with patch.object(ex,'verify_bundle',return_value=(specs[1],[])),patch.object(ex,'fit') as fit:
            with self.assertRaisesRegex(ValueError,'different cohort'):
                ex.batch(bundles,self.library,self.rscript,self.root/'batch',self.root)
            self.assertFalse((self.root/'batch').exists());fit.assert_not_called()


if __name__=='__main__':unittest.main()
