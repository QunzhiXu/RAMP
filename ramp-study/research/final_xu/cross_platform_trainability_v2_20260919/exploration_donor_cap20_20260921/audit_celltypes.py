"""Diagnose exact frozen per-class scores, joint support and annotation provenance."""
from run import ROOT, OUT, TASKS, ALL_TASKS, VARIANTS, save, file_sha256
import json
import numpy as np
import pandas as pd
from summarize import table

SPECS={
    'lung_story':[('story_lung_HLA_ABC/data_umap/data/CELL_METADATA.csv.gz','cell_key','provider_cell_type')],
    'melanoma_story':[
        ('figure_melanoma/inputs/cohort_package/engelbrecht_citeseq/locked_metadata.csv.gz','source_cell_id','Broad_cell.type.1'),
        ('figure_melanoma/inputs/cohort_package/moldoveanu_imc/locked_metadata.csv.gz','cell_uid','Cluster.v2')],
    'hnscc_story':[('figure_HNSCC/alignment/input/cell_metadata.csv.gz','cell_uid','provider_cell_type')],
}

def main():
    manifest=json.loads((OUT/'MANIFEST.json').read_text())
    dest=OUT/'celltype_audit';dest.mkdir(exist_ok=True)
    support=[];per_class=[];confusions=[];fit_rows=[];provider_rows=[];checked={};protected={}
    for name in ALL_TASKS:
        root=ROOT/f'data/real_inputs_v0_3_r2/{name}'
        m=json.loads((root/'manifest.json').read_text())
        md=pd.read_csv(root/'metadata.csv.gz',dtype=str,keep_default_na=False)
        common=set.intersection(*(set(g.celltype) for _,g in md.groupby('dataset')))
        assert set(m['classes'])<=common,name
        checked[name]=dict(classes=m['classes'],all_scored_classes_present_in_every_full_platform=True)
        for ds,g in md.groupby('dataset'):
            for c in m['classes']:
                x=g[g.celltype.eq(c)]
                support.append(dict(task=name,pool='original_full',dataset=ds,celltype=c,
                    cells=len(x),patients=x.patient_id.nunique()))
        protected[str(root/'metadata.csv.gz')]=file_sha256(root/'metadata.csv.gz')
        if name not in TASKS:continue
        sub=pd.read_csv(OUT/f'inputs/{name}/metadata.csv.gz',dtype=str,keep_default_na=False).set_index('cell_id')
        classes=m['classes']
        for ds,g in sub.groupby('dataset'):
            assert set(classes)<=set(g.celltype),name
            for c in classes:
                x=g[g.celltype.eq(c)]
                support.append(dict(task=name,pool='cap20',dataset=ds,celltype=c,
                    cells=len(x),patients=x.patient_id.nunique()))
        checked[name]['all_scored_classes_present_in_every_subset_platform']=True
        seen=set()
        for rel,key,col in SPECS[name]:
            path=ROOT.parent/rel
            orig=pd.read_csv(path,usecols=[key,col],dtype=str,keep_default_na=False).set_index(key)
            matching=orig.index.intersection(sub.index)
            assert not seen.intersection(matching)
            assert orig.loc[matching].index.is_unique
            seen.update(matching)
            joined=sub.loc[matching].join(orig.loc[matching].rename(columns={col:'provider_label'}))
            joined=joined[joined.celltype.isin(classes)]
            count=joined.groupby(['dataset','celltype','provider_label']).size()
            for (ds,c,label),n in count.items():
                provider_rows.append(dict(task=name,dataset=ds,celltype=c,provider_label=label,
                    cells=int(n),source_file=str(path),provider_column=col))
            protected[str(path)]=file_sha256(path)
        assert seen==set(sub.index),name
        checked[name]['subset_cells_exactly_joined_to_source_annotations']=len(seen)
        unique_fit={}
        for j in manifest['tasks'][name]['jobs']:
            for variant in VARIANTS:
                folder=OUT/'jobs'/name/variant/j['id']
                result=json.loads((folder/'RESULT.json').read_text());score=result['score']
                cm=np.asarray(score['confusion_matrix']);tp=np.diag(cm);sup=cm.sum(1);pred=cm.sum(0)
                f1=np.divide(2*tp,sup+pred,out=np.zeros(len(tp)),where=sup+pred>0)
                recall=np.divide(tp,sup,out=np.zeros(len(tp)),where=sup>0)
                precision=np.divide(tp,pred,out=np.zeros(len(tp)),where=pred>0)
                assert np.isclose(f1.mean(),score['macro_f1'],atol=1e-12)
                assert np.isclose(recall.mean(),score['balanced_accuracy'],atol=1e-12)
                key=dict(task=name,variant=variant,id=j['id'],source=j['source'],target=j['target'],
                    family=j['family'],seed=j['seed'])
                for i,c in enumerate(score['classes']):
                    per_class.append(dict(**key,celltype=c,f1=float(f1[i]),recall=float(recall[i]),
                        precision=float(precision[i]),truth_fraction=float(sup[i]),predicted_fraction=float(pred[i])))
                    for k,p in enumerate(score['classes']):
                        confusions.append(dict(**key,truth=c,prediction=p,
                            conditional_fraction=float(cm[i,k]/sup[i]) if sup[i]>0 else np.nan))
                if variant!=VARIANTS[1]:continue
                ledger=json.loads((folder/'ledger.json').read_text())
                fit=sub.loc[ledger['fit']]
                val=sub.loc[ledger['validation']]
                k=(j['source'],j['seed'])
                if k in unique_fit:
                    assert unique_fit[k]==ledger['fit'];continue
                unique_fit[k]=ledger['fit']
                for c in classes:
                    x=fit[fit.celltype.eq(c)]
                    fit_rows.append(dict(task=name,source=j['source'],seed=j['seed'],celltype=c,
                        fit_cells=len(x),fit_patients=x.patient_id.nunique(),
                        validation_cells=int(val.celltype.eq(c).sum())))
        checked[name]['all_per_class_averages_reproduce_original_scores']=True
    counts=pd.DataFrame(support);counts.to_csv(dest/'JOINT_CLASS_COUNTS.csv',index=False)
    scores=pd.DataFrame(per_class);scores.to_csv(dest/'PER_CLASS_PER_JOB.csv',index=False)
    metrics=['f1','recall','precision','truth_fraction','predicted_fraction']
    means=scores.groupby(['task','variant','celltype'])[metrics].mean().reset_index()
    means.to_csv(dest/'PER_CLASS_MEANS.csv',index=False)
    direction=scores.groupby(['task','variant','source','target','celltype'])[metrics].mean().reset_index()
    direction.to_csv(dest/'PER_CLASS_DIRECTION_MEANS.csv',index=False)
    pd.DataFrame(confusions).groupby(['task','variant','source','target','truth','prediction']).conditional_fraction.mean().reset_index().to_csv(dest/'CONFUSION_DIRECTION_MEANS.csv',index=False)
    fits=pd.DataFrame(fit_rows);fits.to_csv(dest/'SOURCE_TRAINING_SUPPORT.csv',index=False)
    prov=pd.DataFrame(provider_rows);prov.to_csv(dest/'PROVIDER_LABEL_COMPOSITION_CAP20.csv',index=False)
    check=dict(checks=checked,protected_input_hashes=protected,outputs_are_diagnostic_only=True,
        no_models_retrained=True,no_labels_changed=True)
    for path,h in protected.items():assert file_sha256(path)==h,path
    save(dest/'CHECKS.json',check)
    report='''# 最新 donor-cap 20 结果的逐类诊断

## joint 的确切含义

七组原冻结数据及本轮三个子集中，所有用于评分的类别都出现在每一个平台的完整输入池中。代码在 `src/prepare_real_inputs.py` 中取的是已经归并的 `celltype` 标签交集，而不是原始作者细标签的严格交集。因此，“joint”只保证粗类名称共同存在；它不保证原注释语义相同、亚类构成相同，也不保证每次 500-cell 源训练样本中都有这一类。

本轮按确切 cell ID 重接三个子集的来源注释，全部匹配。没有新训练、改标签、删除类别或更改正式分数。

## 当前各类实际 F1

以下是新对齐值的结果。先按原患者等权混淆矩阵计算每类 F1，再对三个模型、三个种子、所有平台方向取均值；每组逐类 F1 的均值等于上一轮汇报的宏 F1。

'''
    for name in TASKS:
        report+='### '+name+'\n\n'
        sel=means[(means.task==name)&(means.variant=='refit_cap20')].sort_values('f1')
        report+=table(sel[['celltype','f1','recall','precision']])+'\n\n'
        cc=counts[(counts.task==name)&(counts.pool=='cap20')].pivot(index='celltype',columns='dataset',values='cells').reset_index()
        report+='完整子集的数量：\n\n'+table(cc)+'\n\n'
        ff=fits[fits.task.eq(name)].pivot(index=['source','celltype'],columns='seed',values='fit_cells').reset_index()
        ff.columns=ff.columns.map(str)
        report+='每次 500 个源训练细胞中的数量：\n\n'+table(ff)+'\n\n'
    report+='''## 对低分的解释

- Melanoma story 的 Endothelial：CITE-seq 完整池仅 41 个，三个训练种子中为 2、0、0；虽属于标签交集，但其中两个种子的训练集根本没有这个类别。IMC 子集中有 1,808 个，不能把该方向的低分全部当成对齐失效。
- Melanoma story 的 Myeloid：CITE-seq 855 个中 570 个为 DC/pDC、285 个为 macrophage/monocyte；IMC 子集 8,433 个全部为 macro.mono。粗髓系类别共同存在，但亚类组成不同；CITE-seq 训练集仅 4、4、10 个，又叠加训练稀缺。
- Melanoma 的 B lineage 和 CD8 T 更接近通常意义的共同类别。CD8 T F1 为 0.811；B lineage 却只有 0.282，且 CITE-seq → IMC 为 0.081，反向为 0.483。CITE-seq 的 B 谱系包括 234 个 plasma，只占 25,496 个 B 谱系细胞的不足 1%，不能用这个小成分差异解释整个 B 类低分。它是需要单独解释的实际迁移不足，不应仅以“非 joint”为由删除。
- HNSCC 的 Epithelial / tumor：ADT 原注释为 Salivary gland，CODEX 为 tumor/tumor_CA9。映射为广义上皮大类后共有，不能直接声称两端是相同肿瘤细胞类型。
- HNSCC 的 Stromal：ADT 为 Fibroblast；CODEX 合并 CAF、stromal 和 VSMC_myofibroblast。可作为宽泛基质大类，但细粒度对应不等价。NK / gdT 双方都有相应标签，训练样本最少只有 2 个，不是完全缺失于某一平台。
- Lung 的 Fibroblast / stroma：Cords 为多种 CAF，Senosain 为 Stromal / mesenchymal，Zheng 为 Fibroblast。只有广义间质层级共同；Endothelial 也共同存在，但 Senosain 训练数量为 8、7、11，且其原标签是 Endothelial-like。

## 对评分设计的含义

应将“跨平台粗标签交集”“原始标签的可比性”和“实际训练支持”分开登记。目前低分同时包含弱语义对应、训练稀缺，以及真实共同类别仍难迁移三种情况。可以先处理数量不足和明显不等价的标签，但不能按分数删除共同类别。若改为源端分层抽样，须保留同样标注总预算并让全部方法使用相同抽样；若收紧评价本体，也应对所有方法统一，并保留原始完整结果。

所有逐类结果、平台方向、原标签组成和训练计数均在本目录的 CSV 中。
'''
    (dest/'REVIEW_ZH.md').write_text(report)
    print(means[means.variant.eq('refit_cap20')][['task','celltype','f1','recall']].to_string(index=False))

if __name__=='__main__':main()
