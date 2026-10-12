from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
def draw(fig,box):
    return draw_typical(fig,box,'B','umap_celltype_asw_macro','Cell-type separation','UMAP silhouette',(-.11,.38),[-.1,0,.1,.2,.3])
if __name__=='__main__':run_panel('B',draw)
