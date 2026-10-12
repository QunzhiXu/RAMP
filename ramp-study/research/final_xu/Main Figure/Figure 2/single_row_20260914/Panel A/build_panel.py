from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
def draw(fig,box):
    return draw_typical(fig,box,'A','leiden_celltype_ari','Cell-type ARI','ARI',(0,.72),[0,.2,.4,.6])
if __name__=='__main__':run_panel('A',draw)
