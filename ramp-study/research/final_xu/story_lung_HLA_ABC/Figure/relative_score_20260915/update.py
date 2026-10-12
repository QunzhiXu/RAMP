"""Back up and stage the Relative score terminology update without refitting."""
from pathlib import Path
import hashlib, json, shutil, difflib

HERE=Path(__file__).resolve().parent
FIG=HERE.parent
STORY=FIG.parent
DEST=STORY.parent.parent/'manuscript'
STAGE=HERE/'stage'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def replace_once(text, old, new):
    assert text.count(old)==1, (old, text.count(old))
    return text.replace(old,new)

def main():
    assert not (HERE/'original_hashes.json').exists(), 'Already staged'
    (STAGE/'figure').mkdir(parents=True)
    original={str(p):sha(p) for p in [
        *[DEST/f'{s}.{e}' for s in ['manuscript','manuscript_figure1_polished'] for e in ['tex','pdf']],
        *[p for p in (DEST/'figure').iterdir() if p.is_file()]]}
    (HERE/'original_hashes.json').write_text(json.dumps(original,indent=2)+'\n')
    for folder in ['Panel D','Combined ABCDEF']:
        base=FIG/folder
        archive=base/'versions/before_relative_score_20260915'
        files=[p for p in base.rglob('*') if p.is_file() and p.relative_to(base).parts[0] not in {'versions','cache','tmp','__pycache__'}]
        for p in files:
            out=archive/p.relative_to(base)
            out.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,out)
    cap=STORY/'FIGURE_CAPTION.md'
    shutil.copy2(cap,HERE/'FIGURE_CAPTION_before.md')
    p=FIG/'Panel D/scripts/draw_panel_D.py'
    p.write_text(replace_once(p.read_text(),"cb.set_label('Relative rank'","cb.set_label('Relative score'"))
    t=cap.read_text()
    pairs=[
        ("converted to relative ranks within each source patient's reference cells. These ranks were transferred",
         "converted to relative scores based on expression percentiles within each source patient's reference cells. These scores were transferred"),
        ('The transferred score represents relative HLA-ABC state.',
         'The predicted HLA-ABC score represents relative HLA-ABC state.'),
        ('predicted HLA-ABC state (right)', 'predicted HLA-ABC score (right)'),
        ('Measured HLA-A is expressed as its rank within all available cells of the patient; predictions retain the original transferred score.',
         'Measured HLA-A is displayed as a relative score based on its expression percentile among all available cells of the patient; predicted HLA-ABC scores retain their original transferred values. The shared colorbar is labeled Relative score.'),
        ('转化为各源患者参考细胞内的相对秩', '转化为基于各源患者参考细胞内表达百分位的相对分数'),
        ('输出表示相对 HLA-ABC 状态。', '预测 HLA-ABC 分数表示相对 HLA-ABC 状态。'),
        ('预测 HLA-ABC 状态（右）', '预测 HLA-ABC 分数（右）'),
        ('实测 HLA-A 使用患者全部可用细胞内的相对秩，预测值保持原转移分数。',
         '实测 HLA-A 显示为基于患者全部可用细胞内表达百分位的相对分数，预测 HLA-ABC 分数保持原转移值；共用色条标注为 Relative score。')]
    for old,new in pairs:t=replace_once(t,old,new)
    cap.write_text(t)
    # The root caption is the current source of truth; keep active figure copies in sync.
    combined=FIG/'Combined ABCDEF'
    (combined/'CAPTION_EN.md').write_text(t.split('## English\n\n',1)[1].split('\n---\n',1)[0].strip()+'\n')
    (combined/'CAPTION_ZH.md').write_text(t.split('## 中文对照\n\n',1)[1].strip()+'\n')
    readme=FIG/'Panel D/README_ZH.md'
    if readme.exists():
        readme.write_text(readme.read_text()+'\n\n2026-09-15：色条统一为 **Relative score**。实测侧为患者内表达百分位分数，预测侧为源参考分数经近邻加权转移的连续值。仅更新术语，ROI、色阶、细胞数和指标保持不变。\n')
    for p in (DEST/'figure').iterdir():
        if p.is_file() and p.name not in {'figure5_lung_hla.png','figure5_lung_hla.svg'}:
            (STAGE/'figure'/p.name).symlink_to(p)
    for stem in ['manuscript','manuscript_figure1_polished']:
        old=(DEST/f'{stem}.tex').read_text()
        new=replace_once(old,
            'converted to relative ranks within\neach source patient, then transferred',
            'converted to relative scores within\neach source patient, then transferred')
        new=replace_once(new, 'The transferred score therefore\nrepresents relative HLA-ABC state',
                        'The predicted HLA-ABC score\nrepresents relative HLA-ABC state')
        new=replace_once(new, r'\textbf{(D)} Measured HLA-A (left) and predicted HLA-ABC state (right)',
                        r'\textbf{(D)} Measured HLA-A (left) and predicted HLA-ABC score (right)')
        new=replace_once(new, 'Measured HLA-A is patient-ranked;\npredictions are unchanged.',
                        'The Relative score colorbar shows patient-percentile HLA-A scores\nand unchanged predicted HLA-ABC scores (Methods).')
        anchor=r'\paragraph{Lung vCAF HLA-ABC inference (Figure~\ref{fig:lunghla}).}'+'\n'
        definition=r'''Source-patient HLA-ABC relative scores equal average expression ranks divided
by $n+1$ ($n$ reference cells). Predictions are normalized distance-weighted
neighbor means of these scores. Measured HLA-A uses the same formula across
all available cells per Zheng patient; predictions are not re-ranked.
'''
        new=replace_once(new,anchor,anchor+definition)
        (STAGE/f'{stem}.tex').write_text(new)
        (HERE/f'{stem}.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(DEST/f'{stem}.tex'),tofile=str(STAGE/f'{stem}.tex'))))
    print('Backups complete; terminology and manuscript sources staged.')

if __name__=='__main__':main()
