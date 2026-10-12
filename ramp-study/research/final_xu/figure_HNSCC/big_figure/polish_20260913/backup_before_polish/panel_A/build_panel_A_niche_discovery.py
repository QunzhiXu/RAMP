#!/usr/bin/env python3
"""Horizontal HNSCC workflow draft; all biological glyphs are schematic.

Preserves the selected bidirectional panel and its builder. No cohort data,
scores, or existing panels are changed. RNA and ADT are from the same CITE-seq
CD8 cells; the arrows describe analysis steps, not biological causality.
"""
from pathlib import Path
import os

HERE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.cache' / 'matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

INK = '#182027'
MUTED = '#65717C'
LINE = '#C9D2D9'
GREEN = '#159A72'
BLUE = '#2775B6'
CYAN = '#3BA9D4'
PINK = '#D567A3'
ORANGE = '#DB7500'
WIDTH, HEIGHT = 8.1, 3.45
LOGO = HERE.parents[2] / 'figure/introduction_concept/final_publication/RAMP_final_publication_6000x3000.png'


def box(ax, x, y, w, h, face='white', edge=LINE, lw=1.1, radius=.10):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f'round,pad=0,rounding_size={radius}',
        facecolor=face, edgecolor=edge, linewidth=lw))


def label(ax, x, y, text, size=10, color=INK, bold=False, ha='center', **kwargs):
    return ax.text(x, y, text, fontsize=size, color=color,
        weight='bold' if bold else 'normal', ha=ha, va='center',
        linespacing=1.2, **kwargs)


def arrow(ax, x0, y0, x1, y1, color=BLUE, lw=1.8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
        arrowstyle='-|>', mutation_scale=14, color=color, linewidth=lw))


def tissue(ax):
    box(ax, .36, .89, 2.22, 1.15, '#FFFFFF', '#DDE5E8', .8, .08)
    ax.add_patch(Ellipse((1.50, 1.45), 1.62, .91,
        facecolor='#ECF7F1', edgecolor=GREEN, linewidth=1.2,
        linestyle=(0, (3, 2))))
    background = [(.51,1.84),(.61,1.11),(.88,1.95),(.87,1.03),
        (1.27,1.94),(1.64,1.97),(2.05,1.92),(2.40,1.78),
        (2.43,1.10),(2.07,1.00),(1.67,1.01),(1.17,1.06),
        (.51,1.43),(2.43,1.46)]
    for x, y in background:
        ax.add_patch(Circle((x, y), .041, facecolor='#BFC9D0', edgecolor='white', lw=.4))
    for x, y, angle in [(1.03,1.47,35),(1.36,1.78,155),(1.87,1.39,55),(1.94,1.72,135)]:
        ax.add_patch(Ellipse((x, y), .20, .075, angle=angle,
            facecolor=GREEN, edgecolor='white', lw=.5))
    for x, y in [(1.07,1.71),(1.57,1.22),(2.10,1.49)]:
        ax.add_patch(Circle((x, y), .073, facecolor=PINK, edgecolor='white', lw=.6))
    for x, y in [(1.30,1.45),(1.59,1.66),(1.73,1.46),(1.17,1.24)]:
        ax.add_patch(Circle((x, y), .065, facecolor=CYAN, edgecolor='#175D7A', lw=.8))
    legend = [(.49,.63,'CD8 T',CYAN),(1.54,.63,'CAF',GREEN),
              (.49,.36,'Macrophage',PINK),(1.91,.36,'Other','#BFC9D0')]
    for x,y,text,color in legend:
        ax.add_patch(Circle((x,y),.039,facecolor=color,edgecolor='none'))
        label(ax,x+.10,y,text,8.3,ha='left')


def logo_bridge(fig, ax):
    # Identical source crop to the selected original panel; maintain its aspect.
    with Image.open(LOGO) as im:
        cropped = im.crop((2420,380,3930,1190)).convert('RGB')
    x,y,w,h = 3.035,1.37,.95,.95*810/1510
    for fraction in (.286,.45):
        yy = y+h*(1-fraction)
        for start,end in ((2.78,x),(x+w,4.25)):
            ax.plot([start,end],[yy,yy],color='#263138',lw=2.2,solid_capstyle='butt')
            ax.plot([start,end],[yy,yy],color='#778087',lw=.65,solid_capstyle='butt')
    la = fig.add_axes([x/WIDTH,y/HEIGHT,w/WIDTH,h/HEIGHT])
    la.imshow(cropped)
    la.set_axis_off()
    arrow(ax,2.95,1.08,4.07,1.08)
    label(ax,3.51,.76,'Niche score\ntransfer',9.4,bold=True)


def protein_matrix(ax):
    x,y,w,h=4.55,1.02,2.60,.65
    cols,rows=11,6
    dx,dy=.027,.035
    cw,ch=(w-dx*(cols-1))/cols,(h-dy*(rows-1))/rows
    # Illustrative ordering, not measured expression or effect estimates.
    for col in range(cols):
        xx=x+col*(cw+dx)
        ax.add_patch(Rectangle((xx,1.84),cw,.14,
            facecolor=plt.cm.Blues(.16+.80*col/(cols-1)),edgecolor='none'))
    neutral=['#EDF2F5','#DDE8EF','#CBDCE7','#B9D0DE']
    for row in range(rows):
        yy=y+(rows-row-1)*(ch+dy)
        for col in range(cols):
            color=plt.cm.Oranges(.17+.73*col/(cols-1)) if row in (1,4) else neutral[(row*3+col*5+col//2)%4]
            ax.add_patch(Rectangle((x+col*(cw+dx),yy),cw,ch,facecolor=color,edgecolor='none'))
        if row in (1,4):
            label(ax,7.28,yy+ch/2,'TIGIT' if row==1 else 'CD39',11.2,ORANGE,True,ha='left')
    label(ax,5.85,2.13,'Transferred niche score',10.2)
    label(ax,6.10,.75,'107 additional proteins',10.5,bold=True)


def rna_support_note(ax):
    # Match the protein annotation's size and weight for publication legibility.
    ax.plot([4.47,7.73],[.52,.52],color='#DCE4EB',lw=.7)
    label(ax,6.10,.35,'RNA support: exhaustion-related program',10.5,INK,True)


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(WIDTH,HEIGHT),facecolor='white')
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,WIDTH),ylim=(0,HEIGHT));ax.set_axis_off()
    label(ax,.14,3.17,'A',23,bold=True,ha='left')
    label(ax,.63,3.17,'SPATIAL NICHE-GUIDED BIOMARKER DISCOVERY',15.8,bold=True,ha='left')

    box(ax,.15,.15,2.63,2.67,'#F4FAF7',GREEN,1.5,.15)
    box(ax,.28,2.30,2.37,.39,'#DFF3EA','none',0,.09)
    label(ax,1.465,2.495,'CODEX',14,GREEN,True)
    label(ax,1.465,2.15,'CAF / macrophage niche',10,bold=True)
    tissue(ax)

    box(ax,4.25,.15,3.70,2.67,'#F4F8FC',BLUE,1.5,.15)
    box(ax,4.38,2.30,3.44,.39,'#E1ECF8','none',0,.09)
    label(ax,6.10,2.495,'CITE-seq | CD8 T cells',14,BLUE,True)
    protein_matrix(ax)
    rna_support_note(ax)
    logo_bridge(fig,ax)

    fig.canvas.draw()
    canvas=fig.bbox
    for text in ax.texts:
        bb=text.get_window_extent(fig.canvas.get_renderer())
        if not (canvas.x0<=bb.x0 and bb.x1<=canvas.x1 and canvas.y0<=bb.y0 and bb.y1<=canvas.y1):
            raise RuntimeError(f'Text exceeds canvas: {text.get_text()}')
    for suffix in ('png','pdf','svg'):
        dest=HERE/f'panel_A_niche_discovery.{suffix}'
        fig.savefig(dest,dpi=320,facecolor='white')
        print(dest)
    plt.close(fig)


if __name__=='__main__':
    main()
