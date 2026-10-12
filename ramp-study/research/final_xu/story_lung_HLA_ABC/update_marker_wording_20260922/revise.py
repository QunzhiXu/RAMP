from pathlib import Path
import json,hashlib,csv

HERE=Path(__file__).resolve().parent
STORY=HERE.parent
ROOT=STORY.parent
STAGE=HERE/'manuscript_stage'

def replace(text,old,new):
    assert text.count(old)==1,(old[:80],text.count(old))
    return text.replace(old,new)

old='''reference also supplies the lung-story integration benchmark in
Figure~\ref{fig:benchmark}; the original four benchmarks and cross-platform
learning experiments retain their separately frozen reference datasets. The nine
aligned features were CD3, CD4, CD8,
CD31, Ki67, vimentin, the shared epithelial marker feature
(EPI/cytokeratin), CD45 and HLA-DR. HLA-A and HLA-ABC were excluded.'''
# Keep TeX escapes literal.
old=old.replace('\ref',r'\ref')
new=r'''reference also supplies the lung-story integration benchmark in
Figure~\ref{fig:benchmark} and its updated single-source annotation and
missing-marker experiments in Figure~\ref{fig:trainability}B,C.
Alignment used nine harmonized feature labels: CD3, CD4, CD8, CD31, Ki67,
vimentin, EPI/cytokeratin, CD45 and HLA-DR. These labels map to
platform-specific channels. In Cords IMC, the features labeled CD31, EPI and
CD45 were read from the provider's ``vWF + CD31'', ``Pan Cytokeratin + Keratin
Epithelial'' and ``CD45RA + CD45R0'' rows of the counts assay, respectively.
The corresponding Senosain channels were CD31, Cytokeratin and CD45, and the
Zheng channels were CD31, Pan-Cytokeratin and CD45. The Cords combinations
were supplied in the deposited data; we did not construct them by adding
separate channels. Thus, alignment used corresponding markers or marker
combinations rather than assuming identical antibody targets across platforms.
The complete nine-feature mapping is supplied in
\texttt{supplementary\_data/lung\_marker\_channel\_mapping.csv}.
HLA-A and HLA-ABC were excluded from alignment.'''
tex=(STAGE/'template.tex').read_text()
tex=replace(tex,old,new)
caption_old=r'\citep{zheng2025nsclc} and IMC \citep{cords2024} using nine shared proteins.'
caption_new=r'''\citep{zheng2025nsclc} and IMC \citep{cords2024} using nine corresponding
marker features, including the provider-supplied IMC marker combinations
specified in Methods.'''
tex=replace(tex,caption_old,caption_new)
boundary='''Marker holdout for the Lung configuration uses these mapped channel labels:
CD31, EPI and CD45 include the provider-supplied Cords marker combinations
specified in the lung application Methods. The regression target is the
relative expression of the corresponding measured channel in each platform.'''
anchor='The Senosain expression\nlayer is provider-normalized/denoised, rather than raw instrument output.'
tex=replace(tex,anchor,anchor+'\n'+boundary)
(STAGE/'template.tex').write_text(tex)
p=STAGE/'CROSS_PLATFORM_LEARNING_SECTION.md';p.write_text(replace(p.read_text(),anchor,anchor+'\n'+boundary))
p=STAGE/'FIGURE6_CAPTION.md';p.write_text(replace(p.read_text(),caption_old,caption_new))

en_old='Patient-wise normalization and RAMP 2.3.0 alignment used CD3, CD4, CD8, CD31, Ki-67, vimentin, the shared epithelial marker feature (EPI/cytokeratin), CD45, and HLA-DR.'
en_new='Patient-wise normalization and RAMP 2.3.0 alignment used nine harmonized feature labels: CD3, CD4, CD8, CD31, Ki-67, vimentin, EPI/cytokeratin, CD45, and HLA-DR. In Cords IMC, CD31, EPI, and CD45 denote the provider-supplied `vWF + CD31`, `Pan Cytokeratin + Keratin Epithelial`, and `CD45RA + CD45R0` channels, respectively. They were read directly from the counts assay, rather than summed by our code. These are corresponding marker features across panels, not identical antibody measurements; the full channel mapping is recorded in `MARKER_CHANNELS.md`.'
zh_old='逐患者归一化及 RAMP 2.3.0 对齐使用 CD3、CD4、CD8、CD31、Ki-67、vimentin、共同上皮标记特征 EPI/cytokeratin、CD45 和 HLA-DR。'
zh_new='逐患者归一化及 RAMP 2.3.0 对齐使用九个统一特征名：CD3、CD4、CD8、CD31、Ki-67、vimentin、EPI/cytokeratin、CD45 和 HLA-DR。Cords IMC 的 CD31、EPI、CD45 分别读取作者提供的 `vWF + CD31`、`Pan Cytokeratin + Keratin Epithelial`、`CD45RA + CD45R0` 组合通道；代码直接读取 counts assay 中已有的行，没有自行将不同通道相加。这是跨面板对应特征，不表示抗体测量完全相同；完整映射见 story 根目录的 `MARKER_CHANNELS.md`。'
p=STORY/'FIGURE_CAPTION.md';t=replace(p.read_text(),en_old,en_new);t=replace(t,zh_old,zh_new);p.write_text(t)
p=STORY/'Figure/Combined ABCDEF/CAPTION_ZH.md';p.write_text(replace(p.read_text(),zh_old,zh_new))

rows=[
 ['CD3','CD3','CD3','CD3e'],['CD4','CD4','CD4','CD4'],['CD8','CD8','CD8a','CD8'],
 ['CD31','CD31','vWF + CD31','CD31'],['KI67','KI67','Ki-67','Ki67'],
 ['VIMENTIN','VIMENTIN','Vimentin','Vimentin'],
 ['EPI','CYTOKERATIN','Pan Cytokeratin + Keratin Epithelial','Pan-Cytokeratin'],
 ['CD45','CD45','CD45RA + CD45R0','CD45'],['HLA_DR','HLA_DR','HLA-DR','HLA-DR']]
out=STAGE/'supplementary_data/lung_marker_channel_mapping.csv';out.parent.mkdir(exist_ok=True)
with out.open('w',newline='') as h:
 w=csv.writer(h);w.writerow(['harmonized_feature','Senosain_CyTOF','Cords_IMC','Zheng_mIF']);w.writerows(rows)
table=['| 特征名 | Senosain CyTOF | Cords IMC | Zheng mIF |','|---|---|---|---|']+['| '+' | '.join(row)+' |' for row in rows]
note='''# Lung story 的实际通道映射

2026-09-22。当前版本：`releases/senosain140k_20260921`。
下表区分用于计算的统一特征名与平台提供的通道名。Cords 的 CD31、EPI、CD45
是作者数据中已经存在的组合通道；并非本项目对独立通道求和，也不能描述为三个平台测量相同抗体靶标。

'''+ '\n'.join(table)+'''

Senosain 按 `senosain_140k_markers35.csv` 在 `senosain_140k_raw35.npy` 中定位列；
Cords 从作者 `sce_all_annotated.rds` 的 counts assay 读取上述原名行；
Zheng 从 `mIF_data.h5ad` 的 X 按 var 名称读取。Senosain 的 raw35 文件保存作者处理后的值，非仪器原始读数。

原始读取证据：
- `../overnight0909/scripts/export_full527_cells.R` 第 11、19 行。
- `data_umap/provenance/source_workflow_scripts/prepare_zheng_alignment.py.txt` 第 14、113–114 行。
- `releases/senosain140k_20260921/scripts/align.py` 第 24–34 行。
- `../cross_platform_trainability_v2_20260919/reports/lung_story_scope_audit_20260921/REVIEW_ZH.md` 第 56 行。

此次落实此前审计的 Methods 措辞建议，同步 bioRxiv_Ver1 Methods、Figure 6 图注、
Lung story 图注与补充通道映射表。输入、对齐、预测、统计和 figure 图片均不改变。
这些映射也适用于 Figure 2 的 Lung story benchmark 及 Figure 3 的对应任务。
旧 data_umap 的 MARKER_MAPPING.csv 使用部分统一后的中间名称，追溯原始通道应以本表和上述读取脚本为准。
'''
(STORY/'MARKER_CHANNELS.md').write_text(note)
(STAGE/'LUNG_MARKER_CHANNEL_UPDATE_20260922.md').write_text(note)
for p in [STORY/'README.md',STAGE/'README.md']:
 p.write_text(p.read_text()+'\n\n2026-09-22：已补充 Lung 的实际 marker/channel 对应关系，特别是 Cords 的 CD31、EPI、CD45 组合通道。此次仅修订说明，数据与图像不变。\n')
srcs=[ROOT/'overnight0909/scripts/export_full527_cells.R',STORY/'releases/senosain140k_20260921/scripts/align.py',ROOT/'cross_platform_trainability_v2_20260919/reports/lung_story_scope_audit_20260921/REVIEW_ZH.md']
(HERE/'AUDIT.json').write_text(json.dumps(dict(issue_confirmed=True,application_methods_tex_found=False,
 active_manuscript=str(ROOT.parent/'bioRxiv_Ver1/template.tex'),data_changed=False,
 sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in srcs}),indent=2)+'\n')
print('Methods, caption and original-channel table updated in stage; story wording synchronized.')
