"""Record current A/D presentation and synchronize the bilingual caption."""
from pathlib import Path
import hashlib,json

HERE=Path(__file__).resolve().parent
FIGURE=HERE.parent
STORY=FIGURE.parent

EN='''**(D)** Spatial comparison of measured HLA-A (left) and predicted HLA-ABC state (right) in a previously selected 192 × 192 µm region of Zheng core C_30, patient P_26 (389 cell centroids). Both panels retain the same original cell positions and use the same frozen synthetic DAPI background and synthetic cell shapes. Nuclear shapes were constrained by measured nuclear geometry, with textures from an independent bone marrow DAPI image; orientations and brightness were simulated. HLA colors are cell-level numerical annotations, not acquired or reconstructed subcellular HLA fluorescence. Measured HLA-A is expressed as its rank within all available cells of the patient; predictions retain the original transferred score. The ROI and shared display range 0.25–0.55 were chosen in the earlier C30 ROI/color-range exploration; endpoint saturation is indicated. Scale bars, 50 µm. Spearman correlation from unchanged, unclipped cell values is 0.834 within this region and 0.790 across all 10,719 C30 cells. Across the full 28-patient validation cohort (119 cores, 693,693 cells), the median within-patient Spearman correlation is 0.692 and all cores have positive correlations (0.239–0.883). The synthetic background does not contribute to these metrics.'''
ZH='''**（D）** Zheng 患者 P_26、core C_30 中此前选定的 192 × 192 µm 局部区域，展示实测 HLA-A（左）与预测 HLA-ABC 状态（右），区域内有 389 个细胞质心。两图保留相同的原始细胞位置，使用同一张冻结的合成 DAPI 背景和同一套合成细胞形状。核形状受实测核几何信息约束，核纹理来自独立骨髓 DAPI 图像，朝向和亮度为模拟值。HLA 颜色表示细胞级数值注释，不是原始采集或复原的亚细胞 HLA 荧光图像。实测 HLA-A 使用患者全部可用细胞内的相对秩，预测值保持原转移分数。区域和统一 0.25–0.55 色阶沿用此前 C30 内的 ROI／色阶搜索结果，端点饱和已标注；比例尺为 50 µm。使用未截断的原始细胞值计算，局部 Spearman ρ=0.834，整个 C30 的 10,719 个细胞 ρ=0.790；全验证队列包含 28 位患者、119 个 core、693,693 个细胞，患者内相关中位数为 0.692，全部 core 为正相关（0.239–0.883）。合成背景不参与这些验证指标的计算。'''

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def replace(text,a,b,new):
    i=text.index(a);j=text.index(b,i)
    return text[:i]+new+'\n\n'+text[j:]

def main():
    synced=json.loads((HERE/'manuscript_sync/INSTALLED.json').read_text())
    assert synced['status']=='PASS'
    for path in [STORY/'FIGURE_CAPTION.md',FIGURE/'Combined ABCDEF/CAPTION_EN.md',FIGURE/'Combined ABCDEF/CAPTION_ZH.md']:
        text=path.read_text()
        if '**(D)**' in text:text=replace(text,'**(D)**','**(E)**',EN)
        if '**（D）**' in text:text=replace(text,'**（D）**','**（E）**',ZH)
        path.write_text(text)
    p=FIGURE/'Panel A/README_ZH.md';text=p.read_text()
    text=text.replace('当前为第十二版','当前为第十三版（v13_cross_platform_alignment_icon）')
    text=text.replace('两侧原有铁管残段通过图形裁剪隐藏，新双线直接连接图标边框；两侧可见直线等长。','采用新版无两侧水管图标，副标题为 CROSS-PLATFORM ALIGNMENT。新双线直接连接图标边框，沿用原连接位置；不再使用旧位图裁剪蒙版。')
    text=text.replace('原图标路径、校验值和既有 HNSCC 图使用的裁切范围。','新版图标路径、尺寸和 SHA-256；原图标保留。')
    text=text.replace('`versions/v1/` 至 `versions/v11/`：旧稿归档。','`versions/v1/` 至 `versions/v11/` 及 `versions/v12_before_alignment_icon/`：旧稿归档。')
    p.write_text(text)
    (FIGURE/'Panel D/README_ZH.md').write_text('''# Panel D：Zheng C30 局部 HLA 对照

当前版本 **v3_selected_C30_synthetic_ROI**，2026-09-14。替换为此前选定的 C30/P26 局部图：`S192_X416_Y624`，192 × 192 µm，389 个细胞，统一色阶 0.25–0.55，比例尺 50 µm。

左：实测 HLA-A 患者内秩。右：冻结的 k=30 预测 HLA-ABC 状态。两侧相同的原始位置、合成背景、合成细胞形状及绘图混合参数均沿用 `synthetic_DAPI0913/roi_threshold_search0050913`。没有重新搜索区域或阈值。

图面注明 **Synthetic DAPI background**。背景及细胞形状属于形态约束下的合成显示，使用独立骨髓 DAPI 核纹理；HLA 颜色是细胞级数值注释，不是取得了 Zheng 的原始荧光图像。

局部未截断数值 Spearman ρ=0.833798；全 C30 仍为 0.789875。患者和 core 级的总体指标保持原值。选择的局部不能替代总体结果。

- `panel_D_validation.png` / `.svg`：当前独立面板，4.90 × 2.75 英寸。
- `data/PLOT_DATA.csv.gz`：当前 389 个质心对应的原值，与此前选中表逐项核对。
- `provenance/SELECTED_ROI.json`：当前区域与色阶。
- `provenance/VERIFICATION.json`：数值、来源哈希、布局检查。
- `provenance/C30_DISPLAY.json`：旧全 core 展示的历史来源记录，当前显示以 SELECTED_ROI 为准。
- `versions/v2_before_synthetic_ROI/`：替换前全 core 质心图、脚本、文档及输入。

复现：从 final_xu 运行 `python3 -B "story_lung_HLA_ABC/Figure/Panel D/scripts/draw_panel_D.py"`。脚本仅写本 Panel D 目录，原始及冻结分析只读。
''')
    p=FIGURE/'Combined ABC/README_ZH.md';text=p.read_text().replace('当前为 v2_tighter_BC。','当前为 v3_new_alignment_icon。仅更新 A 为新版 CROSS-PLATFORM ALIGNMENT 图标，B、C 位置和内容不变。')
    text=text.replace('上一版在 `versions/v1_ABC_full_panel_C/`。','替换前版本在 `versions/v2_before_alignment_icon/`，更早版本仍保留。');p.write_text(text)
    p=FIGURE/'Combined ABCDEF/README_ZH.md';text=p.read_text().replace('**v6_survival_association_title**','**v7_alignment_icon_and_synthetic_C30_ROI**')
    text=text.replace('F 标题改为 **Survival association**，图例保留 Lower / Higher。','A 换为新版 CROSS-PLATFORM ALIGNMENT 图标；D 换为此前选定的 389 细胞 C30 局部，色阶 0.25–0.55，图面注明 Synthetic DAPI background。F 标题及 Lower / Higher 图例保持原样。')
    text=text.replace('前一版保存在 `versions/v5_all_five_immune_lineages/`。','替换前完整版本保存在 `versions/v6_before_alignment_icon_and_ROI/`。')
    text=text.replace('D：Zheng C30 的实测 HLA-A／预测 HLA-ABC 空间对照','D：Zheng C30 局部 HLA 对照，389 个细胞，合成 DAPI 背景')
    p.write_text(text)
    cap=FIGURE/'Combined ABCDEF/provenance/CAPTION_SOURCES.json';data=json.loads(cap.read_text())
    data['caption_version']='v2_selected_synthetic_C30_ROI';data['figure_version']='v7_alignment_icon_and_synthetic_C30_ROI'
    data['figure_sha256']=sha(FIGURE/'Combined ABCDEF/panels_ABCDEF_preview.png')
    for item in data['local_sources']:
        if item['supports']=='Selected spatial core and color limits':
            item['path']=str(FIGURE/'Panel D/provenance/SELECTED_ROI.json')
        item['sha256']=sha(Path(item['path']))
    for desc,p in [('Synthetic rendering provenance',STORY/'synthetic_DAPI0913/README_ZH.md'),('Unchanged ROI values',STORY/'synthetic_DAPI0913/roi_threshold_search0050913/results/render_verification.json')]:
        data['local_sources'].append({'supports':desc,'path':str(p),'sha256':sha(p)})
    data['checks']['no_figure_data_or_model_changes']=False
    data['checks'].update({'only_icon_and_display_ROI_updated':True,'HLA_values_and_models_unchanged':True,'synthetic_background_disclosed':True})
    cap.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    for folder in ['Panel A','Panel D','Combined ABC','Combined ABCDEF']:
        base=FIGURE/folder
        review={'status':'PASS','review_date':'2026-09-14','scope':'New icon and selected synthetic C30 display integrated; original numerical HLA data unchanged','manuscript_page':10}
        (base/'provenance/VISUAL_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n')
        files=[{'file':str(p.relative_to(base)),'sha256':sha(p)} for p in sorted(base.rglob('*')) if p.is_file() and p.relative_to(base).parts[0] not in {'cache','tmp','versions'} and p.name!='MANIFEST.json']
        (base/'provenance/MANIFEST.json').write_text(json.dumps(files,indent=2)+'\n')
    (HERE/'README_ZH.md').write_text('''# Lung A/D 图面更新

2026-09-14。A 更新新版无水管 RAMP 图标；D 采用已选 C30 192 µm 局部（389 细胞，0.25–0.55 色阶，50 µm 比例尺）。保留明确的 Synthetic DAPI background 标注。全图的 B、C、E、F 像素一致，统计、模型和预测保持原值。

- [整图](lung_panels_ABCDEF.png)
- [变更检查](REVISION_QA.json)
- [原图备份位置](BACKUPS.json)
- [manuscript 同步记录](manuscript_sync/INSTALLED.json)

当前正式文件仍在各 Panel 目录和 `Combined ABCDEF/panels_ABCDEF_preview.png` / `.svg`。先后运行 Panel A、Panel D 的 draw 脚本，再运行 Combined ABC 的 assemble 脚本与 Combined ABCDEF 的 assemble 脚本即可复现。所有旧图、脚本、文档保存在各目录 versions 中，根图注旧版保存在本目录 previous_docs。

已更新 manuscript 图、对应 D 图注及两份 18 页 PDF。原稿备份：

`'''+synced['backup']+'`\n')
    print('Updated current panel documentation, bilingual captions, and manifests.')

if __name__=='__main__':main()
