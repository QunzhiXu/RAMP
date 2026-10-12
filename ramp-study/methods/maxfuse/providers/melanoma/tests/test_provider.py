"""Synthetic parity/safety tests for the two-stage melanoma technical port."""

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
import melanoma_support as support


def claim_worker(out,bmk,raw,barrier,queue):
    barrier.wait(timeout=20)
    try:
        support.claim(Path(out),"source",Path(bmk),raw_root=Path(raw))
        queue.put("won")
    except FileExistsError: queue.put("lost")


def publish_worker(stage,out,manifest,barrier,queue):
    with patch("melanoma_support.verify_reference",return_value=manifest):
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

    def test_10_source_and_6_method_functions_ast_identical(self):
        for filename,changed,count in (("rebuild_source_active.py",{"main"},10),
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
        for filename,module,names in (("rebuild_source_active.py",source,["EXPECTED_CELLS","SHARED","DATASETS","HICKEY_PANEL","HOCH_PANEL","PTACEK_PANEL"]),
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

    def test_invalid_source_indices_refused(self):
        for v in ([1,1],[1,-1],[1,1.5]):
            with self.assertRaises(AssertionError):source.source_indices(pd.DataFrame({"source_index_zero_based":v}),"test")

    def test_three_extractors_preserve_source_values_and_row_order(self):
        specs=[(source.build_hickey,source.HICKEY_PANEL,"hickey_codex/23_10_11_Melanoma_Marker_Cell_Neighborhood.csv"),
               (source.build_hoch,source.HOCH_PANEL,"hoch_imc/HochSchulz_2022_Melanoma_exprs_cells_by_markers.csv"),
               (source.build_ptacek,source.PTACEK_PANEL,"ptacek_mibi/scaled_mibi_data_w_cell_gating.csv")]
        for index,(fn,panel,relative) in enumerate(specs):
            with self.subTest(dataset=index):
                p=self.raw/relative;p.parent.mkdir()
                markers=[m for _,m in panel]
                values=np.arange(4)[:,None]*(np.arange(len(markers))[None,:]+1)+0.5
                frame=pd.DataFrame(values,columns=markers)
                if index==0:
                    frame["Unnamed: 0"]=["a","b","c","d"];ids=np.array(["hickey::"+x for x in "abcd"])
                elif index==1:
                    frame["cell_id_export"]=["a","b","c","d"];ids=np.array(["hoch::"+x for x in "abcd"])
                else:
                    frame["fov"]="01";frame["label"]=["a","b","c","d"];ids=np.array(["ptacek::01::"+x for x in "abcd"])
                frame.to_csv(p,index=False)
                indices=np.array([2,0,3]); canon=[m for m,_ in panel]
                lock=pd.DataFrame({"cell_id":["new2","new0","new3"],"source_cell_id":ids[indices],"source_index_zero_based":indices})
                expected=values[indices][:,[canon.index(m) for m in source.SHARED]]
                stage=self.root/f"stage{index}";stage.mkdir()
                with patch.object(source,"RAW_ROOT",self.raw),patch.object(source,"EXPECTED_CELLS",3):
                    rec=fn(lock,expected,stage)
                np.testing.assert_array_equal(np.load(stage/rec["slug"]/"active_source_float64.npy"),values[indices])
                self.assertEqual(rec["maximum_canonical_shared_reconstruction_error"],0.)

    def test_pinned_input_hash_not_self_report(self):
        p=self.bmk/"a";p.write_text("wrong")
        records=[{"path":"old/benchmark_data/melanoma/a","bytes":5,"sha256":"0"*64}]
        with patch("melanoma_support.input_records",return_value=([],records)):
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
        with patch("melanoma_support.verify_reference",return_value=m):
            support.publish_new(stage,out,"source")
            with self.assertRaises(FileExistsError):support.publish_new(stage,out,"source")
        self.assertNotEqual((stage/"a").stat().st_ino,(out/"a").stat().st_ino)

    def test_existing_file_not_overwritten(self):
        out,stage,m=self.stage();(out/"a").write_text("old")
        with patch("melanoma_support.verify_reference",return_value=m):
            with self.assertRaises(FileExistsError):support.publish_new(stage,out,"source")
        self.assertEqual((out/"a").read_text(),"old")
        self.assertFalse((out/"COMPLETE.json").exists())

    def test_failure_retains_staging(self):
        out,stage,_=self.stage();support.fail(out,RuntimeError("interrupted"))
        self.assertTrue((stage/"a").exists());self.assertTrue((out/"FAILED.json").exists())

    def test_tampered_output_refused(self):
        out,stage,m=self.stage();(stage/"a").write_text("tampered")
        with patch("melanoma_support.history",return_value=m):
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
