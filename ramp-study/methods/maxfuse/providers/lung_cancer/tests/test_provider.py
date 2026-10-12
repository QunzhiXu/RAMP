"""Synthetic parity/safety tests for the two-stage lung_cancer technical port."""

import ast
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import rebuild_source_active as source
import build_inputs as method
import lung_support as support


def claim_worker(out,bmk,raw,barrier,queue):
    barrier.wait(timeout=20)
    try:
        support.claim(Path(out),"source",Path(bmk),raw_root=Path(raw))
        queue.put("won")
    except FileExistsError: queue.put("lost")


def publish_worker(stage,out,manifest,barrier,queue):
    with patch("lung_support.verify_reference",return_value=manifest):
        barrier.wait(timeout=20)
        try:
            support.publish_new(Path(stage),Path(out),"source")
            queue.put("won")
        except FileExistsError: queue.put("lost")


class ProviderTest(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.root=Path(t.name).resolve()
        self.raw,self.bmk=self.root/"raw",self.root/"benchmark"
        self.raw.mkdir(); self.bmk.mkdir()

    def test_pinned_six_source_files(self): support.check_code()

    def test_11_source_and_6_method_functions_ast_identical(self):
        for filename,changed,count in (("rebuild_source_active.py",{"main"},11),
                                       ("build_inputs.py",{"main","ensure_source_active"},6)):
            def funcs(p):
                return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)}
            old,new=funcs(HERE/"source"/filename),funcs(HERE/filename)
            self.assertEqual(set(old)-set(new),{"publish","verified_existing"})
            self.assertEqual(set(new)-set(old),set())
            names=set(new)-changed
            self.assertEqual(len(names),count)
            for n in names:self.assertEqual(old[n],new[n],filename+":"+n)

    def test_all_scientific_constants_unchanged(self):
        for filename,module,names in (("rebuild_source_active.py",source,["EXPECTED_CELLS","SHARED","DATASETS","SENOSAIN_SHARED_NATIVE","CORDS_SHARED_NATIVE","MONKMAN_SHARED_NATIVE","CORDS_EXPORT_R"]),
                                      ("build_inputs.py",method,["SHARED","DATASETS"])):
            tree=ast.parse((HERE/"source"/filename).read_text())
            for n in tree.body:
                if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in names:
                    self.assertEqual(ast.literal_eval(n.value),getattr(module,n.targets[0].id))

    def test_main_scientific_records_expressions_unchanged(self):
        for filename in ("rebuild_source_active.py","build_inputs.py"):
            def records(path):
                t=ast.parse(path.read_text())
                n=next(n for n in ast.walk(t) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="records")
                return ast.dump(n,include_attributes=False)
            self.assertEqual(records(HERE/"source"/filename),records(HERE/filename))

    def test_sample_sd_and_signed_input(self):
        x=np.array([[-3.,1.],[0.,2.],[4.,5.]])
        z,center,scale=method.zscore(x)
        np.testing.assert_array_equal(z,(x-x.mean(axis=0))/x.std(axis=0,ddof=1))
        np.testing.assert_array_equal(center,x.mean(axis=0))
        np.testing.assert_array_equal(scale,x.std(axis=0,ddof=1))

    def test_constant_features_refused(self):
        with self.assertRaises(AssertionError):method.zscore(np.ones((3,2)))

    def test_published_marker_collision_refused(self):
        with self.assertRaises(AssertionError):
            source.published_markers(["CD8a","CD8"],{"CD8":"CD8a"})

    def test_backed_rows_restores_order_and_refuses_duplicates(self):
        values=np.arange(24).reshape(6,4)
        np.testing.assert_array_equal(source.backed_rows(values,np.array([4,1,3])),values[[4,1,3]])
        with self.assertRaises(AssertionError):source.backed_rows(values,np.array([1,1]))

    def test_senosain_extraction_identity_panel_and_raw_scale(self):
        native=list(source.SENOSAIN_SHARED_NATIVE.values())+[f"extra{i}" for i in range(26)]+["CD90"]
        markers=pd.DataFrame({"raw35_column_zero_based":np.arange(35),"marker":native})
        metadata=pd.DataFrame({"sampled_row_zero_based":np.arange(140000),
                               "source_cell_id":[f"cell{i}" for i in range(140000)]})
        rows=np.array([5,1,9])
        matrix=np.zeros((140000,35))
        matrix[rows]=np.arange(3)[:,None]+np.arange(35)[None,:]+0.25
        canonical=pd.DataFrame({"cell_id":["x","y","z"],"source_cell_id":metadata.iloc[rows].source_cell_id.to_numpy(),
                                "source_row_zero_based":rows})
        stage=self.root/"stage";stage.mkdir()
        def read(path,*args,**kwargs):
            return markers if str(path).endswith("markers35.csv") else metadata
        with patch.object(source,"RAW_ROOT",self.raw),patch.object(source,"EXPECTED_CELLS",3), \
             patch.object(source.np,"load",return_value=matrix),patch.object(source.pd,"read_csv",side_effect=read), \
             patch.object(source,"source_identity",return_value={"path":"synthetic","bytes":0,"sha256":"0"*64}):
            rec=source.build_senosain(canonical,matrix[rows,:8],stage)
        self.assertEqual(rec["active_features"],34)
        self.assertNotIn("CD90",rec["markers"])
        np.testing.assert_array_equal(np.load(stage/"senosain_cytof/active_source_float64.npy"),matrix[rows,:34])
        self.assertEqual(rec["maximum_canonical_raw_shared_reconstruction_error"],0.)

    def test_three_method_routes_match_original_transforms(self):
        root=self.root/"source";root.mkdir()
        n=20000
        values=(np.arange(n)[:,None]+0.3)*(np.arange(8)[None,:]+1)/100
        for dataset,slug,route in method.DATASETS:
            folder=root/slug;folder.mkdir()
            ids=np.array([f"{slug}:{i}" for i in range(n)])
            meta=pd.DataFrame({"cell_id":ids,"dataset":dataset,"donor":"d","unit_key":"u"})
            source.deterministic_csv(pd.DataFrame({"cell_id":ids,"source_cell_id":ids}),folder/"cell_ids.csv.gz")
            np.save(folder/"active_source_float64.npy",values)
            (folder/"active_markers.txt").write_text("\n".join(method.SHARED)+"\n")
            stage=self.root/slug;stage.mkdir()
            with patch.object(method,"SOURCE_ACTIVE",root),patch.object(method,"canonical_metadata",return_value=meta):
                record=method.build_dataset(dataset,slug,route,stage)
            pre=np.arcsinh(values/5.) if slug=="senosain_cytof" else values
            expected=(pre-pre.mean(axis=0))/pre.std(axis=0,ddof=1)
            self.assertEqual(record["pre_z_buffer_sha256_float64"],method.buffer_sha256(pre))
            self.assertEqual(record["processed_buffer_sha256_float64"],method.buffer_sha256(expected))

    def test_pinned_input_hash_not_self_report(self):
        p=self.bmk/"a";p.write_text("wrong")
        records=[{"path":"old/benchmark_data/lung_cancer/a","bytes":5,"sha256":"0"*64}]
        with patch("lung_support.input_records",return_value=([],records)):
            with self.assertRaises(ValueError):support.check_inputs(None,self.bmk)

    def test_claim_refuses_existing_overlapping_or_dangling(self):
        with self.assertRaises(FileExistsError):support.claim(self.raw,"source",self.bmk,raw_root=self.raw)
        with self.assertRaises(ValueError):support.claim(self.raw/"nested","source",self.bmk,raw_root=self.raw)
        alias=self.root/"alias";alias.symlink_to(self.root/"missing")
        with self.assertRaises(FileExistsError):support.claim(alias,"source",self.bmk,raw_root=self.raw)

    def race(self,fn,args):
        ctx=multiprocessing.get_context("spawn");barrier,queue=ctx.Barrier(2),ctx.Queue()
        workers=[ctx.Process(target=fn,args=(*args,barrier,queue)) for _ in range(2)]
        try:
            for p in workers:p.start()
            for p in workers:p.join(25);self.assertEqual(p.exitcode,0)
            self.assertCountEqual([queue.get(timeout=5) for _ in workers],["won","lost"])
        finally:
            for p in workers:
                if p.is_alive():p.terminate();p.join()
            queue.close()

    def stage(self):
        out=support.claim(self.root/"out","source",self.bmk,raw_root=self.raw)
        stage=out/"_work";stage.mkdir()
        (stage/"a").write_text("value")
        m={"outputs":{"a":support.sha(stage/"a")}}
        (stage/"manifest.json").write_text(json.dumps(m))
        return out,stage,m

    def test_real_two_process_claim(self):self.race(claim_worker,(self.root/"out",self.bmk,self.raw))

    def test_real_two_process_publish(self):
        out,stage,m=self.stage();self.race(publish_worker,(stage,out,m))
        self.assertTrue((out/"COMPLETE.json").exists())

    def test_repeat_publish_refused_no_hardlink(self):
        out,stage,m=self.stage()
        with patch("lung_support.verify_reference",return_value=m):
            support.publish_new(stage,out,"source")
            with self.assertRaises(FileExistsError):support.publish_new(stage,out,"source")
        self.assertNotEqual((stage/"a").stat().st_ino,(out/"a").stat().st_ino)

    def test_existing_file_not_overwritten(self):
        out,stage,m=self.stage();(out/"a").write_text("old")
        with patch("lung_support.verify_reference",return_value=m):
            with self.assertRaises(FileExistsError):support.publish_new(stage,out,"source")
        self.assertEqual((out/"a").read_text(),"old")
        self.assertFalse((out/"COMPLETE.json").exists())

    def test_failure_retains_staging(self):
        out,stage,_=self.stage();support.fail(out,RuntimeError("interrupted"))
        self.assertTrue((stage/"a").exists());self.assertTrue((out/"FAILED.json").exists())

    def test_tampered_output_refused(self):
        out,stage,m=self.stage();(stage/"a").write_text("tampered")
        with patch("lung_support.history",return_value=m):
            with self.assertRaises(ValueError):support.verify_reference(stage,"source")

    def source_completion_fixture(self):
        root=self.root/"source_active";root.mkdir()
        manifest={"recipe_sha256":method.sha256(method.SOURCE_RECIPE)}
        (root/"manifest.json").write_text(json.dumps(manifest))
        complete={"status":"complete","stage":"source",
                  "manifest_sha256":method.sha256(root/"manifest.json")}
        return root,manifest,complete

    def test_method_requires_completed_source_stage(self):
        root,manifest,complete=self.source_completion_fixture()
        complete["stage"]="method"
        (root/"COMPLETE.json").write_text(json.dumps(complete))
        with patch.object(method,"SOURCE_ACTIVE",root),patch.object(method,"verify_reference",return_value=manifest):
            with self.assertRaises(AssertionError):method.ensure_source_active()

    def test_method_accepts_complete_source_status(self):
        root,manifest,complete=self.source_completion_fixture()
        (root/"COMPLETE.json").write_text(json.dumps(complete))
        with patch.object(method,"SOURCE_ACTIVE",root),patch.object(method,"verify_reference",return_value=manifest):
            self.assertEqual(method.ensure_source_active(),manifest)

    def test_method_refuses_partial_source_status(self):
        root,manifest,complete=self.source_completion_fixture()
        complete["status"]="partial"
        (root/"COMPLETE.json").write_text(json.dumps(complete))
        with patch.object(method,"SOURCE_ACTIVE",root),patch.object(method,"verify_reference",return_value=manifest):
            with self.assertRaises(AssertionError):method.ensure_source_active()

    def test_method_refuses_missing_source_status(self):
        root,manifest,complete=self.source_completion_fixture()
        del complete["status"]
        (root/"COMPLETE.json").write_text(json.dumps(complete))
        with patch.object(method,"SOURCE_ACTIVE",root),patch.object(method,"verify_reference",return_value=manifest):
            with self.assertRaises(AssertionError):method.ensure_source_active()


if __name__=="__main__":unittest.main()
