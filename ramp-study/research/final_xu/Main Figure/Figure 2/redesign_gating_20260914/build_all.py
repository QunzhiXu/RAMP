"""Build independently reproducible panels and a two-row review at 180 mm wide."""
import importlib.util
from common import *

def load(letter):
    spec=importlib.util.spec_from_file_location('panel_'+letter,BASE/f'Panel {letter}/build_panel.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    setup(); mods={c:load(c) for c in 'ABCDEFG'}
    for c,m in mods.items():run_panel(c,m.draw)
    fig=plt.figure(figsize=(180/25.4,128/25.4),facecolor='white')
    positions={'A':(0,0,43,58),'B':(45,0,43,58),'C':(90,0,43,58),'D':(135,0,45,58),
        'E':(0,60,88,61),'F':(90,60,43,61),'G':(135,60,45,61)}
    details={c:m.draw(fig,positions[c]) for c,m in mods.items()}
    case_legend(fig,125)
    export(fig,BASE/'Combined','figure_2_two_rows',{'panels':details,'positions_mm':positions,
        'layout':'Four top-row slots; E spans two bottom-row slots, F/G one each. Panel plots preserve standalone physical scale.',
        'status':'Review draft, no formal figure or manuscript replacement.'})

if __name__=='__main__':main()
