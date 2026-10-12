from pathlib import Path
import hashlib,json,re,csv
import pandas as pd
import fitz
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;STAGE=HERE/'manuscript_stage';LIVE=ROOT.parent/'bioRxiv_Ver1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks={}
old=ROOT/'Main Figure/icon_update_20260915/median_shift_box_statistics.csv'
new=ROOT/'Main Figure/publication_display_20260922/median_shift_box_statistics.csv'
a=pd.read_csv(old);b=pd.read_csv(new);assert len(a)==6 and len(b)==5
pd.testing.assert_frame_equal(a[a.method.isin(b.method)].reset_index(drop=True),b)
assert b.n.eq(6173).all();checks['figure1_d_retained_statistics_identical']=True
old=ROOT/'Main Figure/Figure 2/Panel C/umap_style_review/ridge_display_audit.csv'
new=ROOT/'Main Figure/publication_display_20260922/ridge_display_audit.csv'
a=pd.read_csv(old);b=pd.read_csv(new);pd.testing.assert_frame_equal(a,b[a.columns],check_exact=False,rtol=1e-12,atol=1e-12)
assert len(b)==552;checks['figure1_melanoma_all_552_ridges_match_saved_example']=True
manifest=json.loads((ROOT/'Main Figure/publication_display_20260922/manifest.json').read_text())
assert manifest['panel_b_cells']==60000 and manifest['panel_b_shared_markers']==12 and not manifest['umap_refit']
for name,h in manifest['all_source_hashes'].items():assert sha(name)==h,name
checks['figure1_all_sources_unchanged']=True
for letter in 'BC':
 base=ROOT/'cross_platform_trainability_v2_20260919/Figure'
 for filename in ['per_model_scores.csv','per_benchmark_scores.csv']:
  pd.testing.assert_frame_equal(pd.read_csv(base/f'Panel {letter}'/filename),pd.read_csv(base/f'Combined ABC/Panel {letter}'/filename))
checks['figure3_bc_scores_unchanged']=True
supp_manifest=json.loads((ROOT/'Supplementary Figures/harmony_display_20260922/MANIFEST.json').read_text())
assert len(supp_manifest)==13 and all(x['pixels_outside_label_boxes_identical'] for x in supp_manifest)
checks['all_13_supplement_figures_only_text_changed']=True
for row in json.loads((HERE/'published_data_display_edits.json').read_text()):
 with (LIVE/row['path']).open(newline='') as f:a=list(csv.reader(f))
 with (STAGE/row['path']).open(newline='') as f:b=list(csv.reader(f))
 expected=[['Harmony' if v in ['Harmony-direct','Harmony direct'] else v for v in r] for r in a]
 assert expected==b,row['path']
checks['all_display_csv_numeric_values_unchanged']=True
for filename in ['figure2_benchmark.pdf','figure4_cd19.pdf','figure5_melanoma.png','figure6_lung_hla.png','figure7_hnscc.pdf']:
 assert sha(STAGE/'figure'/filename)==sha(LIVE/'figure'/filename)
checks['other_five_main_figures_unchanged']=True
pdfchecks={}
for stem in ['template','supplementary_material']:
 doc=fitz.open(STAGE/(stem+'.pdf'));text='\n'.join(p.get_text() for p in doc)
 assert not re.search('Harmony[- ]direct|Harmony[ -]arcsinh',text)
 log=(STAGE/(stem+'.log')).read_text(errors='replace')
 assert not re.search(r'Overfull|undefined|LaTeX Error',log)
 pdfchecks[stem]=dict(pages=len(doc),no_obsolete_display_names=True,no_overfull_or_undefined_references=True,sha256=sha(STAGE/(stem+'.pdf')))
assert pdfchecks['supplementary_material']['pages']==17
s=(STAGE/'template.tex').read_text()
for missing in [r'\ref{fig:trainability}D',r'\ref{fig:trainability}E',r'\ref{fig:trainability}F','supported smaller','fewer training labels','Annotation budgets, capacity','D--F retain']:
 assert missing not in s,missing
checks['removed_panels_and_resource_claims_absent']=True
assert s.count(r'\paperfig{')==7
sources=json.loads((STAGE/'SUPPLEMENTARY_FIGURE_SOURCES.json').read_text());assert len(sources)==13
for x in sources:assert sha(STAGE/x['destination'])==x['sha256']
checks['figure_sources_and_caption_count_match']=True
out=dict(checks=checks,pdfchecks=pdfchecks,visual_review='Reviewed main Figure 1 and Figure 3 pages, all-main-page contact sheet, and representative supplementary pages S1/S8/S10/S12. No clipping, label overlap or new blank pages.',scope='Figure 1 melanoma; artifact Harmony-arcsinh omitted; Figure 3 A-C; Harmony public label; no new supplementary figures.')
(HERE/'QA.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
