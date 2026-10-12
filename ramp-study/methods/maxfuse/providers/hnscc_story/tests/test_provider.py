"""HNSCC scientific parity and create-only safety; synthetic inputs only."""
import ast
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import prepare as recipe
import hnscc_support as support


def claim_worker(out,comparison,rahim,haist,barrier,queue):
    barrier.wait(timeout=20)
    try:
        support.claim(Path(out),Path(comparison),Path(rahim),Path(haist),{})
        queue.put("won")
    except FileExistsError:queue.put("lost")


def publish_worker(stage,out,history,barrier,queue):
    with patch("hnscc_support.history",return_value=history):
        barrier.wait(timeout=20)
        try:
            support.publish_new(Path(stage),Path(out));queue.put("won")
        except FileExistsError:queue.put("lost")


class ProviderTest(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve()
        self.comp=self.root/"comparison";self.comp.mkdir()
        self.rahim,self.haist=self.root/"adt.h5ad",self.root/"codex.csv"
        self.rahim.write_bytes(b"synthetic");self.haist.write_bytes(b"synthetic")

    def test_six_frozen_sources(self):support.check_code()

    def test_eight_scientific_functions_ast_identical(self):
        def funcs(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
        old,new=funcs(HERE/"source/prepare.py"),funcs(HERE/"prepare.py")
        self.assertEqual(set(old)-set(new),{"lung","melanoma"})
        names=set(new)-{"main","module"}
        self.assertEqual(len(names),8)
        for name in names:self.assertEqual(new[name],old[name],name)

    def test_marker_map_literal_without_sampler_execution(self):
        path=HERE/"source/alignment/01_prepare_alignment_input.py"
        with patch.object(recipe,"RAHIM_ADT",self.rahim),patch.object(recipe,"HAIST_CELLS",self.haist),patch.object(recipe,"SOURCES",{}):
            with patch("builtins.exec",side_effect=AssertionError("sampler execution forbidden")):
                mod=recipe.module(path,"irrelevant")
        self.assertEqual(len(mod.MARKER_MAP),22)
        self.assertEqual(mod.MARKER_MAP["CD4"],("CD4-1","CD4"))
        self.assertEqual(mod.RAHIM_ADT,self.rahim)

    def test_unknown_helper_refused(self):
        with self.assertRaises(ValueError):recipe.module(self.root/"unknown.py","x")

    def test_positions_preserve_requested_order(self):
        np.testing.assert_array_equal(recipe.positions(["b","a","c"],["c","b"]),[2,0])

    def test_positions_missing_or_duplicate_source_refused(self):
        for source,wanted in ((["a","a"],["a"]),(["a","b"],["z"])):
            with self.assertRaises(AssertionError):recipe.positions(source,wanted)

    def test_h5ad_categorical_and_plain_encodings(self):
        path=self.root/"obs.h5ad"
        with h5py.File(path,"w") as f:
            obs=f.create_group("obs")
            group=obs.create_group("grouped");group.create_dataset("categories",data=np.array(["a","b"],dtype=h5py.string_dtype()))
            group.create_dataset("codes",data=[1,0])
            old=obs.create_group("__categories");old.create_dataset("old",data=np.array(["a","b"],dtype=h5py.string_dtype()))
            obs.create_dataset("old",data=[1,0]);obs.create_dataset("plain",data=np.array(["b","a"],dtype=h5py.string_dtype()))
            obs.create_dataset("numeric",data=[3,4])
        with h5py.File(path,"r") as f:
            for key in ("grouped","old","plain"):np.testing.assert_array_equal(recipe.obs(f,key),["b","a"])
            np.testing.assert_array_equal(recipe.obs(f,"numeric"),[3,4])

    def save_fixture(self,op="log1p",bad=None):
        root=self.root/"fixture";root.mkdir();(root/"data").mkdir()
        raw=np.array([[1.,2.,9.,3.],[2.,7.,8.,3.],[4.,4.,5.,3.],[8.,9.,1.,3.]])
        ids=np.array(["a","b","c","d"]);donors=np.array(["P1"]*4)
        md=pd.DataFrame({"dataset":"GSE212797_ADT","cell_id":ids,"unit_key":"GSE212797_P1"})
        if bad=="donor":donors[0]="P2"
        if bad=="id":ids=ids[::-1]
        ref=raw[:,:2].copy()
        if bad=="reference":ref[0,0]+=1
        with patch.object(recipe,"HERE",root),patch.object(recipe,"MD",md),patch.object(recipe,"REF",ref),patch.object(recipe,"SHARED",["A","B"]),patch.object(recipe,"CFG",{"case":"hnscc_story"}),patch.object(recipe,"views",[]) as views:
            recipe.save_view("GSE212797_ADT","synthetic",raw,["A","B","isotype","constant"],ids,donors,{"A":"A","B":"B"},op,{"isotype":"isotype control"})
            records=list(views)
        return root/"data/synthetic",raw,records

    def test_log1p_sample_sd_and_control_exclusion(self):
        out,raw,records=self.save_fixture()
        ready=np.log1p(raw[:,:2]);expected=(ready-ready.mean(axis=0))/ready.std(axis=0,ddof=1)
        np.testing.assert_array_equal(np.load(out/"active.npy"),expected)
        np.testing.assert_array_equal(np.load(out/"shared.npy"),expected)
        self.assertEqual(records[0]["active_features"],2)
        self.assertIn("constant",records[0]["excluded_features"])

    def test_identity_does_not_add_arcsinh(self):
        out,raw,_=self.save_fixture("identity")
        np.testing.assert_array_equal(np.load(out/"active_preprocessed.npy"),raw[:,:2])

    def test_donor_mismatch_refused(self):
        with self.assertRaises(AssertionError):self.save_fixture(bad="donor")

    def test_cell_order_mismatch_refused(self):
        with self.assertRaises(AssertionError):self.save_fixture(bad="id")

    def test_shared_reference_mismatch_refused(self):
        with self.assertRaises(AssertionError):self.save_fixture(bad="reference")

    def test_pinned_input_not_self_reported(self):
        with patch("hnscc_support.input_contract",return_value={"adt":(self.rahim,"0"*64)}):
            with self.assertRaises(ValueError):support.check_inputs(self.comp,self.rahim,self.haist)

    def test_existing_overlap_and_dangling_outputs_refused(self):
        for out,error in ((self.comp,FileExistsError),(self.comp/"nested",ValueError)):
            with self.assertRaises(error):support.claim(out,self.comp,self.rahim,self.haist,{})
        alias=self.root/"alias";alias.symlink_to(self.root/"missing")
        with self.assertRaises(FileExistsError):support.claim(alias,self.comp,self.rahim,self.haist,{})

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

    def test_real_two_process_claim(self):
        self.race(claim_worker,(self.root/"out",self.comp,self.rahim,self.haist))

    def stage(self):
        out=support.claim(self.root/"out",self.comp,self.rahim,self.haist,{})
        stage=out/"_work";stage.mkdir();data=stage/"data";data.mkdir()
        (data/"a").write_text("value")
        m={"case":"hnscc_story","output_sha256":{"a":support.sha(data/"a")},
           "source_sha256":{str(HERE/"source/alignment/01_prepare_alignment_input.py"):support.SOURCE_HASHES["source/alignment/01_prepare_alignment_input.py"]}}
        (data/"manifest.json").write_text(json.dumps(m))
        cfg=json.loads((HERE/"source/full_run_config.json").read_text())
        cfg.update(comparison=str(out),full_input_manifest_sha256=support.sha(data/"manifest.json"))
        (stage/"run_config.json").write_text(json.dumps(cfg))
        return out,stage,m

    def test_real_two_process_publish(self):
        out,stage,m=self.stage();self.race(publish_worker,(stage,out,m))
        self.assertTrue((out/"data/COMPLETE.json").exists())

    def test_repeat_publish_refused_and_no_hardlink(self):
        out,stage,m=self.stage()
        with patch("hnscc_support.history",return_value=m):
            support.publish_new(stage,out)
            with self.assertRaises(FileExistsError):support.publish_new(stage,out)
        self.assertNotEqual((stage/"data/a").stat().st_ino,(out/"data/a").stat().st_ino)

    def test_existing_data_preserved(self):
        out,stage,m=self.stage();(out/"data").mkdir();(out/"data/keep").write_text("old")
        with patch("hnscc_support.history",return_value=m):
            with self.assertRaises(FileExistsError):support.publish_new(stage,out)
        self.assertEqual((out/"data/keep").read_text(),"old")
        self.assertFalse((out/"COMPLETE.json").exists())

    def test_tampered_output_rejected(self):
        out,stage,m=self.stage();(stage/"data/a").write_text("changed")
        with patch("hnscc_support.history",return_value=m):
            with self.assertRaises(ValueError):support.publish_new(stage,out)
        self.assertFalse((out/"COMPLETE.json").exists())

    def test_interruption_retains_stage(self):
        out,stage,_=self.stage();support.fail(out,RuntimeError("interrupted"))
        self.assertTrue((stage/"data/a").exists());self.assertTrue((out/"FAILED.json").exists())


if __name__=="__main__":unittest.main()
