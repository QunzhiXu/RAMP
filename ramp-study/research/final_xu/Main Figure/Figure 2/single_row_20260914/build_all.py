"""Build A–F as independently reproducible panels and one 180 mm-wide row."""
import importlib.util
from common import *

def load(letter):
    spec=importlib.util.spec_from_file_location('panel_'+letter,BASE/f'Panel {letter}/build_panel.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    setup();mods={c:load(c) for c in 'ABCDEF'}
    for c,m in mods.items():run_panel(c,m.draw)
    fig=plt.figure(figsize=(180/25.4,71/25.4),facecolor='white')
    positions={'A':(0,0,26.5,65),'B':(27.5,0,26.5,65),'C':(55.5,0,34,65),
        'D':(91,0,29,65),'E':(121.5,0,30,65),'F':(153,0,27,65)}
    details={c:m.draw(fig,positions[c]) for c,m in mods.items()}
    case_legend(fig,68)
    export(fig,BASE/'Combined','figure_2_single_row',{'panels':details,'positions_mm':positions,
        'layout':'One row A–F: two cell-type metrics, category profile, mean rank, four stacked gate distributions, gate-transfer accuracy.',
        'status':'Review draft; no formal figure or manuscript replacement.'})
    prior=json.loads((BASE/'prior_version_hashes.json').read_text())
    assert all(sha(p)==h for p,h in prior.items()),'Prior draft changed'
    checks={'prior_draft_files_unchanged':len(prior),'all_source_hashes_unchanged':all(sha(p)==h for p,h in READS.items()),
        'prior_points_unchanged':{},'e_density_and_statistics_unchanged':True}
    for c in 'ABCF':
        a=pd.read_csv(BASE/f'Panel {c}/plotted_values.csv',float_precision='round_trip')
        b=pd.read_csv(PRIOR/f'Panel {c}/plotted_values.csv',float_precision='round_trip')
        pd.testing.assert_frame_equal(a,b);checks['prior_points_unchanged'][c]=len(a)
    for name in ['density_coordinates.csv','gate_statistics.csv']:
        pd.testing.assert_frame_equal(pd.read_csv(BASE/'Panel E'/name,float_precision='round_trip'),
            pd.read_csv(PRIOR/'Panel E'/name,float_precision='round_trip'))
    (BASE/'VERIFY.json').write_text(json.dumps(checks,indent=2)+'\n')

if __name__=='__main__':main()
