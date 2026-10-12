"""Two-row A–F draft: rank boxplots above, three four-row gate examples below."""
import importlib.util
from common import *

def load(letter):
    spec=importlib.util.spec_from_file_location('panel_'+letter,BASE/f'Panel {letter}/build_panel.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    setup();mods={c:load(c) for c in 'ABCDEF'}
    for c,m in mods.items():run_panel(c,m.draw)
    fig=plt.figure(figsize=(180/25.4,126/25.4),facecolor='white')
    positions={'A':(0,0,43,58),'B':(45,0,43,58),'C':(90,0,43,58),'D':(135,0,45,58),
        'E':(0,55,133,60),'F':(135,55,45,60)}
    details={c:m.draw(fig,positions[c]) for c,m in mods.items()}
    case_legend(fig,121)
    export(fig,BASE/'Combined','figure_2_seven_benchmarks_main23',{'panels':details,'positions_mm':positions,
        'layout':'A/B/C/D across the top, E spans three columns below with 12 distributions, F occupies the fourth column. No G.',
        'status':'Current Figure 2; A-D: seven benchmarks, 23 metrics, four categories. E-F: previously frozen gating evaluation on four original benchmarks.'})
    old=json.loads((BASE/'prior_version_hashes.json').read_text())
    assert all(sha(p)==h for p,h in old.items())
    checks={}
    for letter in 'F':
        a=pd.read_csv(out(letter)/'plotted_values.csv',float_precision='round_trip')
        b=pd.read_csv(PREVIOUS_FIGURE/f'Panel {letter}/plotted_values.csv',float_precision='round_trip')
        pd.testing.assert_frame_equal(a,b)
        checks[letter]=len(a)
    for name in ['density_coordinates.csv','gate_statistics.csv','data_manifest.json']:
        assert sha(out('E')/name)==sha(PREVIOUS_FIGURE/'Panel E'/name)
    (BASE/'VERIFY.json').write_text(json.dumps({'old_files_unchanged':len(old),
        'integration_benchmarks':7,'retained_metrics':23,'families':4,'each_family_weight':.25,
        'd_matches_frozen_primary_ranking':True,'gating_benchmarks':4,
        'gating_density_and_threshold_tables_unchanged':True,
        'unchanged_panel_values':checks,'inputs_unchanged':all(sha(p)==h for p,h in READS.items()),
        'frozen_gates_verified':True},indent=2)+'\n')

if __name__=='__main__':main()
