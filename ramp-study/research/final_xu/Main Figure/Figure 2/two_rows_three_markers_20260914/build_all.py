"""Two-row A–F draft: mean rank above, three four-row gate examples below."""
import importlib.util
from common import *

def load(letter):
    spec=importlib.util.spec_from_file_location('panel_'+letter,BASE/f'Panel {letter}/build_panel.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    setup();mods={c:load(c) for c in 'ABCDEF'}
    for c,m in mods.items():run_panel(c,m.draw)
    fig=plt.figure(figsize=(180/25.4,140/25.4),facecolor='white')
    positions={'A':(0,0,43,58),'B':(45,0,43,58),'C':(90,0,43,58),'D':(135,0,45,58),
        'E':(0,60,133,74),'F':(135,60,45,74)}
    details={c:m.draw(fig,positions[c]) for c,m in mods.items()}
    case_legend(fig,137)
    export(fig,BASE/'Combined','figure_2_three_markers',{'panels':details,'positions_mm':positions,
        'layout':'A/B/C/D across the top, E spans three columns below with 12 distributions, F occupies the fourth column. No G.',
        'status':'Review draft, not installed into manuscript.'})
    old=json.loads((BASE/'prior_version_hashes.json').read_text())
    assert all(sha(p)==h for p,h in old.items())
    checks={}
    for letter in 'ABCF':
        a=pd.read_csv(out(letter)/'plotted_values.csv',float_precision='round_trip')
        b=pd.read_csv(PRIOR/f'Panel {letter}/plotted_values.csv',float_precision='round_trip')
        pd.testing.assert_frame_equal(a,b)
        checks[letter]=len(a)
    (BASE/'VERIFY.json').write_text(json.dumps({'old_files_unchanged':len(old),
        'unchanged_panel_values':checks,'inputs_unchanged':all(sha(p)==h for p,h in READS.items()),
        'frozen_gates_verified':True},indent=2)+'\n')

if __name__=='__main__':main()
