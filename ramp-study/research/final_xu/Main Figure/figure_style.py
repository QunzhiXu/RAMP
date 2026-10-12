"""Shared display and provenance helpers for the compact lower row."""
from pathlib import Path
import hashlib
import json
import os

MAIN = Path(__file__).resolve().parent
ROOT = MAIN.parent
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(MAIN / '.cache/matplotlib'))
import matplotlib.pyplot as plt

DPI = 210
HEIGHT_PX = 966
WIDTHS = {'C': 1590, 'D': 1590, 'E': 806}
METHODS = ['Harmony-direct', 'cyCombine', 'CytoNorm', 'fastMNN', 'MaxFuse', 'RAMP']
CASES = ['bone_marrow', 'tnbc', 'melanoma', 'lung_cancer']
READS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def track(path):
    path = Path(path).resolve()
    READS[str(path)] = sha(path)
    return path


def setup():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.linewidth': .55, 'svg.fonttype': 'none',
                         'xtick.major.width': .55, 'ytick.major.width': .55})


def new_panel(letter, title):
    # A subpixel offset prevents float truncation from losing a raster row.
    fig = plt.figure(figsize=((WIDTHS[letter]+.001)/DPI, (HEIGHT_PX+.001)/DPI), facecolor='white')
    width, height = fig.get_size_inches()
    fig.text(.16/width, 1-.29/height, letter, fontsize=17, weight='bold')
    fig.text(.56/width, 1-.29/height, title, fontsize=13.5, weight='bold')
    return fig


def save(fig, folder, name, details):
    folder = Path(folder)
    fig.savefig(folder / f'{name}.png', dpi=DPI)
    fig.savefig(folder / f'{name}.svg', dpi=DPI)
    plt.close(fig)
    assert all(sha(p) == digest for p, digest in READS.items())
    details.update(input_sha256=READS, inputs_unchanged=True,
                   dpi=DPI, height_px=HEIGHT_PX)
    (folder / 'manifest.json').write_text(json.dumps(details, indent=2)+'\n')
    print(folder / f'{name}.png', flush=True)
