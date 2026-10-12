"""Lung story scientific parity and create-only safety; synthetic inputs only."""
import ast
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import prepare as recipe
import lung_story_support as support


def claim_worker(out,comparison,metadata,senosain,cords,zheng,rscript,barrier,queue):
    barrier.wait(timeout=20)
    try:
        support.claim(Path(out),Path(comparison),Path(metadata),Path(senosain),Path(cords),Path(zheng),{},Path(rscript))
        queue.put("won")
    except FileExistsError:queue.put("lost")


def publish_worker(stage,out,history,barrier,queue):
    with patch("lung_story_support.history",return_value=history):
        barrier.wait(timeout=20)
        try:
            support.publish_new(Path(stage),Path(out));queue.put("won")
        except FileExistsError:queue.put("lost")


class ProviderTest(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve()
        self.comp=self.root/"comparison";self.comp.mkdir()
        self.metadata,self.cords,self.zheng,self.rscript=[self.root/n for n in ("metadata.csv.gz","cords.rds","zheng.h5ad","Rscript")]
        self.senosain=self.root/"senosain";self.senosain.mkdir()
        for p in (self.metadata,self.cords,self.zheng,self.rscript):p.write_bytes(b"synthetic")

    def test_eight_frozen_sources(self):support.check_code()

    def test_seven_scientific_functions_ast_identical(self):
        def funcs(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
        old,new=funcs(HERE/"source/prepare.py"),funcs(HERE/"prepare.py")
        self.assertEqual(set(old)-set(new),{"melanoma","hnscc","module"})
        names=set(new)-{"main","lung","export_cords"}
        self.assertEqual(len(names),7)
        for name in names:self.assertEqual(new[name],old[name],name)

    def test_lung_extractor_diff_is_only_four_input_paths(self):
        old=(HERE/"source/prepare.py").read_text()
        old=old.replace("STORY/'releases/senosain140k_20260921/inputs/CELL_METADATA.csv.gz'","CELL_METADATA")
        old=old.replace("ROOT/'raw_data/lung_cancer/Senosain_2025_CyTOF_and_omics/annotation'","SENOSAIN_ROOT")
        old=old.replace("record['source'] if Path(record['source']).is_absolute() else str((ROOT/record['source']).resolve())","str(CORDS_RDS)")
        old=old.replace("ROOT/'raw_data/lung_cancer/Zheng2025/mIF_data.h5ad'","ZHENG_H5AD")
        def body(text):
            return ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=="lung"),include_attributes=False)
        self.assertEqual(body(old),body((HERE/"prepare.py").read_text()))

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

    def save_fixture(self,op="asinh5",bad=None):
        root=self.root/"fixture";root.mkdir();(root/"data").mkdir()
        raw=np.array([[1.,2.,9.,3.],[2.,7.,8.,3.],[4.,4.,5.,3.],[8.,9.,1.,3.]])
        ids=np.array(["a","b","c","d"]);donors=np.array(["P1"]*4)
        md=pd.DataFrame({"dataset":"Senosain CyTOF","cell_id":ids,"unit_key":"Senosain CyTOF|P1"})
        if bad=="donor":donors[0]="P2"
        if bad=="id":ids=ids[::-1]
        ref=raw[:,:2].copy()
        if bad=="reference":ref[0,0]+=1
        with patch.object(recipe,"HERE",root),patch.object(recipe,"MD",md),patch.object(recipe,"REF",ref),patch.object(recipe,"SHARED",["A","B"]),patch.object(recipe,"CFG",{"case":"lung_story"}),patch.object(recipe,"views",[]) as views:
            recipe.save_view("Senosain CyTOF","synthetic",raw,["A","B","isotype","constant"],ids,donors,{"A":"A","B":"B"},op,{"isotype":"isotype control"})
            records=list(views)
        return root/"data/synthetic",raw,records

    def test_asinh5_sample_sd_and_control_exclusion(self):
        out,raw,records=self.save_fixture()
        ready=np.arcsinh(raw[:,:2]/5);expected=(ready-ready.mean(axis=0))/ready.std(axis=0,ddof=1)
        np.testing.assert_array_equal(np.load(out/"active.npy"),expected)
        np.testing.assert_array_equal(np.load(out/"shared.npy"),expected)
        self.assertEqual(records[0]["active_features"],2)
        self.assertIn("constant",records[0]["excluded_features"])

    def test_imc_uses_asinh_cofactor_one(self):
        out,raw,_=self.save_fixture("asinh1")
        np.testing.assert_array_equal(np.load(out/"active_preprocessed.npy"),np.arcsinh(raw[:,:2]))

    def test_synthetic_three_platform_extraction_and_index_bases(self):
        root=self.root/"extract";root.mkdir();(root/"data").mkdir()
        shared=["CD3","CD4","CD8","CD31","KI67","VIMENTIN","EPI","CD45","HLA_DR"]
        names=["CD3","CD4","CD8","CD31","KI67","VIMENTIN","CYTOKERATIN","CD45","HLA_DR","CD90"]
        raw=np.array([[1+j*(i+1)+(i*i)%3 for j in range(10)] for i in range(4)],float)
        ix=np.array([2,0,3,1]);ids=np.array(["a","b","c","d"])
        np.save(self.senosain/"senosain_140k_raw35.npy",raw)
        pd.DataFrame({"marker":names}).to_csv(self.senosain/"senosain_140k_markers35.csv",index=False)
        pd.DataFrame({"source_cell_id":ids,"patient_id":["P1"]*4}).to_csv(self.senosain/"senosain_140k_cell_metadata.csv.gz",index=False)
        cn=["CD3","CD4","CD8a","vWF + CD31","Ki-67","Vimentin","Pan Cytokeratin + Keratin Epithelial","CD45RA + CD45R0","HLA-DR","Iridium_191"]
        folder=root/"data/cords_imc";folder.mkdir()
        (folder/"raw_feature_names.txt").write_text("\n".join(cn)+"\n")
        raw[ix].astype('<f8').tofile(folder/"counts.float64.bin")
        (folder/"EXPORT_COMPLETE.json").write_text(json.dumps({"source":str(self.cords),"source_sha256":support.sha(self.cords)}))
        zn=["CD3e","CD4","CD8","CD31","Ki67","Vimentin","Pan-Cytokeratin","CD45","HLA-DR","extra"]
        with h5py.File(self.zheng,"w") as f:
            var=f.create_group("var");var.create_dataset("_index",data=np.array(zn,dtype=h5py.string_dtype()))
            obs=f.create_group("obs")
            obs.create_dataset("CellID",data=np.array(ids,dtype=h5py.string_dtype()))
            obs.create_dataset("patient_id",data=np.array(["P3"]*4,dtype=h5py.string_dtype()))
            f.create_dataset("X",data=raw)
        datasets=["Senosain CyTOF","Cords IMC","Zheng mIF"];records=[]
        for d,patient in zip(datasets,["P1","P2","P3"]):
            for i in ix:
                records.append({"dataset":d,"cell_key":d+"|"+ids[i],"source_row_id":int(i+1 if d=="Cords IMC" else i),"source_cell_id":ids[i],"patient_id":patient})
        md=pd.DataFrame(records);md.to_csv(self.metadata,index=False)
        md.loc[md.dataset.eq("Cords IMC")].rename(columns={"cell_key":"cell_id"}).to_csv(folder/"source_identity.csv",index=False)
        fit=md.rename(columns={"cell_key":"cell_id"}).assign(unit_key=md.dataset+"|"+md.patient_id)
        state={"HERE":root,"CELL_METADATA":self.metadata,"SENOSAIN_ROOT":self.senosain,"CORDS_RDS":self.cords,"ZHENG_H5AD":self.zheng,
               "CFG":{"case":"lung_story","datasets":datasets},"MD":fit,"REF":np.tile(raw[ix,:9],(3,1)),"SHARED":shared,"SOURCES":{},"views":[]}
        with patch.multiple(recipe,**state):recipe.lung()
        for slug,factor,width in (("senosain_cytof",5,9),("cords_imc",1,9),("zheng_mif",1,10)):
            path=root/"data"/slug
            np.testing.assert_array_equal(np.load(path/"active_preprocessed.npy"),np.arcsinh(raw[ix,:width]/factor))
            np.testing.assert_array_equal(np.load(path/"shared.npy"),np.load(path/"active.npy")[:,:9])
        self.assertEqual(len(state["views"]),3)

    def r_fixture(self,returncode):
        root=self.root/"run with spaces";root.mkdir();(root/"data").mkdir()
        module=self.root/"module";(module/"source").mkdir(parents=True)
        request=pd.DataFrame({"dataset":["Cords IMC"],"cell_key":["Cords IMC|8"],"source_row_id":[8],"patient_id":["P1"],"source_cell_id":["c8"]})
        request.to_csv(self.metadata,index=False)
        expected=recipe.hashlib.sha256(request.drop(columns="dataset").to_csv(index=False).encode()).hexdigest()
        (module/"source/historical_manifest.json").write_text(json.dumps({"output_sha256":{"cords_imc/requested_cells.csv":expected}}))
        def run(cmd,**kw):
            self.assertEqual(cmd,[str(self.rscript),'--vanilla',str(module/'source/export_cords.R'),str(self.cords),str(root/'data/cords_imc')])
            self.assertNotIn(" ",kw["env"]["TMPDIR"])
            self.assertTrue(Path(kw["env"]["TMPDIR"]).is_dir())
            self.assertEqual(kw["cwd"],root)
            return SimpleNamespace(returncode=returncode)
        with patch.multiple(recipe,HERE=root,MODULE=module,CELL_METADATA=self.metadata,CORDS_RDS=self.cords,SOURCES={}):
            with patch.object(recipe.subprocess,"run",side_effect=run),patch.object(recipe,"check_export") as check:
                if returncode:
                    with self.assertRaisesRegex(RuntimeError,"R export failed"):recipe.export_cords(self.rscript)
                    check.assert_not_called()
                else:
                    recipe.export_cords(self.rscript);check.assert_called_once_with(root/'data/cords_imc',self.cords)
        self.assertTrue((root/"export_cords.stdout.log").exists())
        self.assertTrue((root/"data/cords_imc/requested_cells.csv").exists())

    def test_r_wrapper_space_free_scratch_and_fixed_arguments(self):self.r_fixture(0)

    def test_failed_r_export_retains_selected_rows_and_logs(self):self.r_fixture(1)

    def test_donor_mismatch_refused(self):
        with self.assertRaises(AssertionError):self.save_fixture(bad="donor")

    def test_cell_order_mismatch_refused(self):
        with self.assertRaises(AssertionError):self.save_fixture(bad="id")

    def test_shared_reference_mismatch_refused(self):
        with self.assertRaises(AssertionError):self.save_fixture(bad="reference")

    def test_pinned_input_not_self_reported(self):
        with patch("lung_story_support.input_contract",return_value={"adt":(self.cords,"0"*64)}):
            with self.assertRaises(ValueError):support.check_inputs(self.comp,self.metadata,self.senosain,self.cords,self.zheng)

    def test_existing_overlap_and_dangling_outputs_refused(self):
        for out,error in ((self.comp,FileExistsError),(self.comp/"nested",ValueError)):
            with self.assertRaises(error):support.claim(out,self.comp,self.metadata,self.senosain,self.cords,self.zheng,{},self.rscript)
        alias=self.root/"alias";alias.symlink_to(self.root/"missing")
        with self.assertRaises(FileExistsError):support.claim(alias,self.comp,self.metadata,self.senosain,self.cords,self.zheng,{},self.rscript)

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
        self.race(claim_worker,(self.root/"out",self.comp,self.metadata,self.senosain,self.cords,self.zheng,self.rscript))

    def stage(self):
        out=support.claim(self.root/"out",self.comp,self.metadata,self.senosain,self.cords,self.zheng,{},self.rscript)
        stage=out/"_work";stage.mkdir();data=stage/"data";data.mkdir()
        (data/"a").write_text("value")
        folder=data/"cords_imc";folder.mkdir()
        export=json.loads((HERE/"source/historical_export_complete.json").read_text())
        export["source"]=str(self.cords)
        (folder/"EXPORT_COMPLETE.json").write_text(json.dumps(export))
        for name in ("requested_cells.csv","raw_feature_names.txt","counts.float64.bin","c_counts_asinh.float64.bin","source_identity.csv"):
            (folder/name).write_bytes(b"synthetic")
        request=json.loads((out/"REQUEST.json").read_text())
        request["inputs"]={"cords_rds":{"path":str(self.cords),"sha256":support.sha(self.cords)}}
        (out/"REQUEST.json").write_text(json.dumps(request))
        m={"case":"lung_story","output_sha256":{str(p.relative_to(data)):support.sha(p) for p in data.rglob("*") if p.is_file()},
           "source_sha256":{str(self.cords):support.sha(self.cords)}}
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
        with patch("lung_story_support.history",return_value=m):
            support.publish_new(stage,out)
            with self.assertRaises(FileExistsError):support.publish_new(stage,out)
        self.assertNotEqual((stage/"data/a").stat().st_ino,(out/"data/a").stat().st_ino)

    def test_existing_data_preserved(self):
        out,stage,m=self.stage();(out/"data").mkdir();(out/"data/keep").write_text("old")
        with patch("lung_story_support.history",return_value=m):
            with self.assertRaises(FileExistsError):support.publish_new(stage,out)
        self.assertEqual((out/"data/keep").read_text(),"old")
        self.assertFalse((out/"COMPLETE.json").exists())

    def test_tampered_output_rejected(self):
        out,stage,m=self.stage();(stage/"data/a").write_text("changed")
        with patch("lung_story_support.history",return_value=m):
            with self.assertRaises(ValueError):support.publish_new(stage,out)
        self.assertFalse((out/"COMPLETE.json").exists())

    def test_export_absolute_source_is_only_allowed_record_difference(self):
        out,stage,m=self.stage()
        with patch("lung_story_support.history",return_value=m):
            support.check_export(stage/"data/cords_imc",self.cords)
            p=stage/"data/cords_imc/EXPORT_COMPLETE.json";j=json.loads(p.read_text());j["features"]=44;p.write_text(json.dumps(j))
            with self.assertRaisesRegex(ValueError,"scientific record"):support.check_export(p.parent,self.cords)

    def test_export_wrong_source_or_bin_is_refused(self):
        out,stage,m=self.stage()
        with patch("lung_story_support.history",return_value=m):
            with self.assertRaisesRegex(ValueError,"source path"):support.check_export(stage/"data/cords_imc",self.zheng)
            (stage/"data/cords_imc/counts.float64.bin").write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError,"export differs"):support.check_export(stage/"data/cords_imc",self.cords)

    def test_r_executable_changed_during_run_is_refused(self):
        out,stage,_=self.stage();self.rscript.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"R executable changed"):support.publish_new(stage,out)
        self.assertFalse((out/"COMPLETE.json").exists())

    def test_input_changed_during_run_is_refused(self):
        out,stage,m=self.stage()
        request=json.loads((out/"REQUEST.json").read_text())
        request["inputs"]={"adt":{"path":str(self.cords),"sha256":support.sha(self.cords)}}
        (out/"REQUEST.json").write_text(json.dumps(request));self.cords.write_bytes(b"changed")
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
