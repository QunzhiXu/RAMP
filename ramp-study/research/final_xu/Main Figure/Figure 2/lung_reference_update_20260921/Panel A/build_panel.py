from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
def draw(fig,box):
    return draw_typical(fig,box,'A','louvain_celltype_ari','Cell-type agreement','Louvain ARI',(0,.84),[0,.2,.4,.6,.8])
if __name__=='__main__':run_panel('A',draw)
