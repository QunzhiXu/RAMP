"""Melanoma story scientific parity and create-only safety; synthetic inputs only."""
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
import melanoma_story_support as support


def claim_worker(out,comparison,mold,eng,qc,metadata,barrier,queue):
    barrier.wait(timeout=20)
    try:
        support.claim(Path(out),Path(comparison),Path(mold),Path(eng),Path(qc),Path(metadata),{})
        queue.put("won")
    except FileExistsError:queue.put("lost")


def publish_worker(stage,out,history,barrier,queue):
    with patch("melanoma_story_support.history",return_value=history):
        barrier.wait(timeout=20)
        try:
            support.publish_new(Path(stage),Path(out));queue.put("won")
        except FileExistsError:queue.put("lost")


class ProviderTest(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve()
        self.comp=self.root/"comparison";self.comp.mkdir()
        self.mold,self.eng,self.qc,self.metadata=[self.root/n for n in ("imc.tsv.gz","adt.csv","qc.tsv","metadata.csv.gz")]
        for p in (self.mold,self.eng,self.qc,self.metadata):p.write_bytes(b"synthetic")

    def test_six_frozen_sources(self):support.check_code()

    def test_seven_scientific_functions_ast_identical(self):
        def funcs(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
        old,new=funcs(HERE/"source/prepare.py"),funcs(HERE/"prepare.py")
        self.assertEqual(set(old)-set(new),{"lung","hnscc"})
        names=set(new)-{"main","module","melanoma"}
        self.assertEqual(len(names),7)
        for name in names:self.assertEqual(new[name],old[name],name)

    def test_marker_maps_literal_without_sampler_execution(self):
        path=HERE/"source/inputs/cohort_package/build_story_sample.py"
        with patch.object(recipe,"SOURCES",{}):
            with patch("builtins.exec",side_effect=AssertionError("sampler execution forbidden")):
                mod=recipe.module(path,"irrelevant")
        self.assertEqual(len(mod.MOLD_SOURCE),13)
        self.assertEqual(len(mod.ENG_SOURCE),13)
        self.assertEqual(mod.MOLD_SOURCE["CD8"],"CD8a")
        self.assertEqual(mod.ENG_SOURCE["CD8"],"CD8-AB")

    def test_melanoma_extractor_diff_is_only_four_input_paths(self):
        old=(HERE/"source/prepare.py").read_text()
        old=old.replace("mod.MOLD_ROOT/'ICI_meanIntensity.tsv.gz'","MOLD_CELLS")
        old=old.replace("ROOT/'raw_data/melanoma/moldoveanu_imc/cohort_audit/marker_inventory_and_paper_qc.tsv'","MARKER_QC")
        old=old.replace("mod.ENG_ROOT/'ADT_counts.csv'","ENG_ADT")
        old=old.replace("STORY/'inputs/cohort_package/engelbrecht_citeseq/locked_metadata.csv.gz'","ENG_METADATA")
        def body(text):
            return ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=="melanoma"),include_attributes=False)
        self.assertEqual(body(old),body((HERE/"prepare.py").read_text()))

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
        md=pd.DataFrame({"dataset":"Engelbrecht CITE-seq","cell_id":ids,"unit_key":"engelbrecht::P1"})
        if bad=="donor":donors[0]="P2"
        if bad=="id":ids=ids[::-1]
        ref=raw[:,:2].copy()
        if bad=="reference":ref[0,0]+=1
        with patch.object(recipe,"HERE",root),patch.object(recipe,"MD",md),patch.object(recipe,"REF",ref),patch.object(recipe,"SHARED",["A","B"]),patch.object(recipe,"CFG",{"case":"melanoma_story"}),patch.object(recipe,"views",[]) as views:
            recipe.save_view("Engelbrecht CITE-seq","synthetic",raw,["A","B","isotype","constant"],ids,donors,{"A":"A","B":"B"},op,{"isotype":"isotype control"})
            records=list(views)
        return root/"data/synthetic",raw,records

    def test_log1p_sample_sd_and_control_exclusion(self):
        out,raw,records=self.save_fixture()
        ready=np.log1p(raw[:,:2]);expected=(ready-ready.mean(axis=0))/ready.std(axis=0,ddof=1)
        np.testing.assert_array_equal(np.load(out/"active.npy"),expected)
        np.testing.assert_array_equal(np.load(out/"shared.npy"),expected)
        self.assertEqual(records[0]["active_features"],2)
        self.assertIn("constant",records[0]["excluded_features"])

    def test_imc_uses_asinh_cofactor_one(self):
        out,raw,_=self.save_fixture("asinh1")
        np.testing.assert_array_equal(np.load(out/"active_preprocessed.npy"),np.arcsinh(raw[:,:2]))

    def test_synthetic_provider_extraction_qc_and_requested_row_order(self):
        root=self.root/"extraction";root.mkdir();(root/"data").mkdir()
        raw=np.array([[1.,2.,4.,5.],[2.,7.,3.,2.],[4.,4.,2.,6.],[8.,9.,1.,7.]])
        table=pd.DataFrame({"x":[0]*4,"y":[0]*4,"Sample_ID":["S1"]*4,"Cell_ID":["a","b","c","d"],
                            "CD4":raw[:,0],"CD8a":raw[:,1],"CCR7":raw[:,2],"Histone.H3":raw[:,3],"Cluster":["unused"]*4})
        table.to_csv(self.mold,sep="\t",index=False)
        pd.DataFrame({"marker":["CCR7","Histone.H3"],"paper_flagged_nonspecific":[True,False],
                      "technical_or_nuclear_channel":[False,True]}).to_csv(self.qc,sep="\t",index=False)
        pd.DataFrame(raw[:,:3].T,index=["CD4-AB","CD8-AB","isotype-Ctrl-AB"],columns=["a","b","c","d"]).to_csv(self.eng)
        pd.DataFrame({"source_cell_id":["a","b","c","d"],"Sample_ID":["L1"]*4,"Patient_ID":["P2"]*4}).to_csv(self.metadata,index=False)
        ix=np.array([2,0,3,1]);jx=np.array([1,3,0,2])
        ids=np.array(["a","b","c","d"])
        md=pd.DataFrame({"dataset":["Moldoveanu IMC"]*4+["Engelbrecht CITE-seq"]*4,
                         "cell_id":["S1::"+x for x in ids[ix]]+ids[jx].tolist(),
                         "unit_key":["moldoveanu::S1"]*4+["engelbrecht::P2"]*4})
        state={"HERE":root,"MOLD_CELLS":self.mold,"ENG_ADT":self.eng,"MARKER_QC":self.qc,
               "ENG_METADATA":self.metadata,"CFG":{"case":"melanoma_story","datasets":["Moldoveanu IMC","Engelbrecht CITE-seq"]},
               "MD":md,"REF":np.vstack([raw[ix,:2],raw[jx,:2]]),"SHARED":["CD4","CD8"],"SOURCES":{},"views":[]}
        with patch.multiple(recipe,**state):recipe.melanoma()
        imc=root/"data/moldoveanu_imc";adt=root/"data/engelbrecht_adt"
        np.testing.assert_array_equal(np.load(imc/"active_preprocessed.npy"),np.arcsinh(raw[ix,:2]))
        np.testing.assert_array_equal(np.load(adt/"active_preprocessed.npy"),np.log1p(raw[jx,:2]))
        np.testing.assert_array_equal(np.load(imc/"cell_ids.npy"),md.cell_id.to_numpy()[:4])
        np.testing.assert_array_equal(np.load(adt/"cell_ids.npy"),md.cell_id.to_numpy()[4:])
        self.assertEqual(set(state["views"][0]["excluded_features"]),{"CCR7","Histone.H3"})
        self.assertEqual(set(state["views"][1]["excluded_features"]),{"isotype-Ctrl-AB"})
        self.assertEqual(len(state["SOURCES"]),5)  # two protein files, QC, identity, static helper

    def test_donor_mismatch_refused(self):
        with self.assertRaises(AssertionError):self.save_fixture(bad="donor")

    def test_cell_order_mismatch_refused(self):
        with self.assertRaises(AssertionError):self.save_fixture(bad="id")

    def test_shared_reference_mismatch_refused(self):
        with self.assertRaises(AssertionError):self.save_fixture(bad="reference")

    def test_pinned_input_not_self_reported(self):
        with patch("melanoma_story_support.input_contract",return_value={"adt":(self.eng,"0"*64)}):
            with self.assertRaises(ValueError):support.check_inputs(self.comp,self.mold,self.eng,self.qc,self.metadata)

    def test_existing_overlap_and_dangling_outputs_refused(self):
        for out,error in ((self.comp,FileExistsError),(self.comp/"nested",ValueError)):
            with self.assertRaises(error):support.claim(out,self.comp,self.mold,self.eng,self.qc,self.metadata,{})
        alias=self.root/"alias";alias.symlink_to(self.root/"missing")
        with self.assertRaises(FileExistsError):support.claim(alias,self.comp,self.mold,self.eng,self.qc,self.metadata,{})

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
        self.race(claim_worker,(self.root/"out",self.comp,self.mold,self.eng,self.qc,self.metadata))

    def stage(self):
        out=support.claim(self.root/"out",self.comp,self.mold,self.eng,self.qc,self.metadata,{})
        stage=out/"_work";stage.mkdir();data=stage/"data";data.mkdir()
        (data/"a").write_text("value")
        m={"case":"melanoma_story","output_sha256":{"a":support.sha(data/"a")},
           "source_sha256":{str(HERE/"source/inputs/cohort_package/build_story_sample.py"):support.SOURCE_HASHES["source/inputs/cohort_package/build_story_sample.py"]}}
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
        with patch("melanoma_story_support.history",return_value=m):
            support.publish_new(stage,out)
            with self.assertRaises(FileExistsError):support.publish_new(stage,out)
        self.assertNotEqual((stage/"data/a").stat().st_ino,(out/"data/a").stat().st_ino)

    def test_existing_data_preserved(self):
        out,stage,m=self.stage();(out/"data").mkdir();(out/"data/keep").write_text("old")
        with patch("melanoma_story_support.history",return_value=m):
            with self.assertRaises(FileExistsError):support.publish_new(stage,out)
        self.assertEqual((out/"data/keep").read_text(),"old")
        self.assertFalse((out/"COMPLETE.json").exists())

    def test_tampered_output_rejected(self):
        out,stage,m=self.stage();(stage/"data/a").write_text("changed")
        with patch("melanoma_story_support.history",return_value=m):
            with self.assertRaises(ValueError):support.publish_new(stage,out)
        self.assertFalse((out/"COMPLETE.json").exists())

    def test_input_changed_during_run_is_refused(self):
        out,stage,m=self.stage()
        request=json.loads((out/"REQUEST.json").read_text())
        request["inputs"]={"adt":{"path":str(self.eng),"sha256":support.sha(self.eng)}}
        (out/"REQUEST.json").write_text(json.dumps(request));self.eng.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"Input changed during run"):support.publish_new(stage,out)
        self.assertFalse((out/"data").exists());self.assertFalse((out/"COMPLETE.json").exists())

    def test_builder_changed_during_run_is_refused(self):
        out,stage,_=self.stage();real_sha=support.sha
        with patch.object(support,"sha",side_effect=lambda p:"0"*64 if Path(p)==HERE/"prepare.py" else real_sha(p)):
            with self.assertRaisesRegex(ValueError,"Builder changed during run"):support.publish_new(stage,out)
        self.assertFalse((out/"data").exists());self.assertFalse((out/"COMPLETE.json").exists())

    def test_interruption_retains_stage(self):
        out,stage,_=self.stage();support.fail(out,RuntimeError("interrupted"))
        self.assertTrue((stage/"data/a").exists());self.assertTrue((out/"FAILED.json").exists())


if __name__=="__main__":unittest.main()
