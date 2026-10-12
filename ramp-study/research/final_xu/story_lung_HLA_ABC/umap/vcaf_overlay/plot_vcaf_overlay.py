"""Show author-annotated Cords vCAF on the frozen Zheng validation UMAP."""
from pathlib import Path
import os
import sys

BASE = Path(__file__).resolve().parent
STORY = BASE.parents[1]
os.environ['MPLCONFIGDIR'] = str(BASE / 'cache' / 'matplotlib')
os.environ['XDG_CACHE_HOME'] = str(BASE / 'cache')
os.environ['TMPDIR'] = str(BASE / 'tmp')
sys.dont_write_bytecode = True


def write_guard(event, args):
    if event == 'open':
        path, mode, flags = args
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if writing and isinstance(path, (str, bytes, os.PathLike)):
            resolved = Path(os.fsdecode(path)).resolve()
            if not resolved.is_relative_to(BASE):
                raise PermissionError(f'Writes are restricted to {BASE}: {resolved}')
    elif event in {'os.mkdir', 'os.remove', 'os.rmdir'}:
        if not Path(os.fsdecode(args[0])).resolve().is_relative_to(BASE):
            raise PermissionError(f'Mutation outside {BASE}: {args[0]}')


sys.addaudithook(write_guard)
for name in ['cache/matplotlib', 'tmp', 'results']:
    (BASE / name).mkdir(parents=True, exist_ok=True)

import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

OUT = BASE / 'results'
PINK = '#e54285'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'svg.fonttype': 'none', 'axes.titleweight': 'medium'})


def sha(path):
    return hashlib.file_digest(open(path, 'rb'), 'sha256').hexdigest()


def main():
    inputs = {
        'metadata': STORY / 'data_umap/data/CELL_METADATA.csv.gz',
        'coordinates': STORY / 'umap/results/RAMP_UMAP.npy',
        'zheng_values': STORY / 'validation/results/UMAP_SAMPLE_PREDICTIONS.csv.gz',
        'original_figure': STORY / 'validation/results/figures/HLA_A_measured_vs_predicted_UMAP.png',
    }
    before = {str(p): sha(p) for p in inputs.values()}
    md = pd.read_csv(inputs['metadata'], low_memory=False)
    u = np.load(inputs['coordinates'])
    z = pd.read_csv(inputs['zheng_values'], float_precision='round_trip')
    assert len(md) == len(u) == 84000 and u.shape[1] == 2
    assert md.cell_key.is_unique and z.cell_key.is_unique
    zi = np.flatnonzero(md.dataset.eq('Zheng mIF'))
    z = z.set_index('cell_key').loc[md.iloc[zi].cell_key].reset_index()
    assert len(z) == len(zi) == 28000
    assert np.array_equal(z.patient_id, md.iloc[zi].patient_id)
    assert np.array_equal(z.source_row_id, md.iloc[zi].source_row_id)
    vcaf = md.dataset.eq('Cords IMC') & md.provider_cell_type.eq('vCAF')
    assert np.array_equal(vcaf, md.dataset.eq('Cords IMC') & md.cell_subtype.eq('vCAF'))
    vi = np.flatnonzero(vcaf)
    assert len(vi) == 646 and not np.intersect1d(zi, vi).size
    vxy = u[vi]
    order = np.random.default_rng(42).permutation(len(zi))
    zxy = u[zi][order]
    lo, hi = u.min(axis=0), u.max(axis=0)
    pad = (hi - lo) * .04
    audits = []

    def base_axis(ax, title, panel):
        ax.scatter(u[:, 0], u[:, 1], s=.7, c='#cbd0d5', alpha=.14,
                   linewidths=0, rasterized=True, zorder=1)
        ax.set(title=title, xlabel='UMAP 1', ylabel='UMAP 2', xticks=[], yticks=[],
               xlim=(lo[0]-pad[0], hi[0]+pad[0]), ylim=(lo[1]-pad[1], hi[1]+pad[1]))
        ax.set_aspect('equal', adjustable='box')
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['bottom', 'left']].set_color('#6c737d')
        ax.text(-.045, 1.045, panel, transform=ax.transAxes, fontsize=15, weight='bold')

    def rings(ax):
        # All 646 vCAF are shown at their original coordinates, without jitter.
        ax.scatter(vxy[:, 0], vxy[:, 1], s=11, facecolors='none', edgecolors='white',
                   linewidths=1.05, alpha=.9, zorder=4)
        artist = ax.scatter(vxy[:, 0], vxy[:, 1], s=11, facecolors='none', edgecolors=PINK,
                            linewidths=.55, alpha=.95, zorder=5)
        assert np.array_equal(artist.get_offsets(), vxy)
        return artist

    def expression(ax, column, title, panel):
        base_axis(ax, title, panel)
        values = z[column].to_numpy()[order]
        artist = ax.scatter(zxy[:, 0], zxy[:, 1], c=values, cmap='viridis',
                            vmin=0, vmax=1, s=2.5, linewidths=0, alpha=1,
                            rasterized=True, zorder=2)
        rings(ax)
        assert np.array_equal(artist.get_offsets(), zxy)
        assert np.array_equal(artist.get_array(), values)
        audits.append({'panel': title, 'n_zheng': len(zxy), 'n_cords_vcaf': len(vxy),
                       'coordinates_exact': True, 'values_exact': True, 'color_limits': [0, 1]})
        return artist

    handle = Line2D([], [], marker='o', markerfacecolor='none', markeredgecolor=PINK,
                    linestyle='', markersize=5.5,
                    label='Cords author-annotated vCAF (646 cells; 27 patients)')

    def finish(fig, axes, artist, stem, width):
        cb = fig.colorbar(artist, ax=axes[-2:], shrink=.74, pad=.025, fraction=.035,
                          ticks=[0, .2, .4, .6, .8, 1])
        cb.set_label('Measured HLA-A rank / predicted HLA-A proxy')
        fig.legend(handles=[handle], loc='lower center', bbox_to_anchor=(.5, .058),
                   frameon=False, fontsize=11)
        fig.text(.5, .019, 'Shared three-cohort UMAP  |  Zheng color scale: 0–1  |  All-source transfer, k = 30',
                 ha='center', fontsize=10, color='#596370')
        fig.suptitle('Cords vCAF in the Zheng HLA-A validation embedding', fontsize=17,
                     x=.5, y=.974)
        fig.savefig(OUT / f'{stem}.png', dpi=230, facecolor='white')
        fig.savefig(OUT / f'{stem}.svg', facecolor='white')
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.7))
    fig.subplots_adjust(left=.042, right=.933, top=.85, bottom=.155, wspace=.115)
    expression(axes[0], 'observed_HLA_A_sample_patient_rank', 'Zheng: measured HLA-A', 'A')
    artist = expression(axes[1], 'RAMP_k30', 'Zheng: predicted HLA-A proxy', 'B')
    finish(fig, axes, artist, 'Zheng_HLA_A_with_Cords_vCAF', 13.0)

    fig, axes = plt.subplots(1, 3, figsize=(17.8, 6.35))
    fig.subplots_adjust(left=.032, right=.948, top=.85, bottom=.165, wspace=.12)
    base_axis(axes[0], 'Cords: author-annotated vCAF', 'A')
    axes[0].scatter(zxy[:, 0], zxy[:, 1], s=1.5, c='#c3cbd0', alpha=.5,
                    linewidths=0, rasterized=True, zorder=2)
    axes[0].scatter(vxy[:, 0], vxy[:, 1], s=9, c=PINK, alpha=.85,
                    edgecolors='white', linewidths=.15, zorder=3)
    expression(axes[1], 'observed_HLA_A_sample_patient_rank', 'Zheng: measured HLA-A', 'B')
    artist = expression(axes[2], 'RAMP_k30', 'Zheng: predicted HLA-A proxy', 'C')
    finish(fig, axes, artist, 'vCAF_context_and_Zheng_validation', 17.8)

    vcells = md.iloc[vi][['cell_key', 'patient_id', 'source_row_id', 'provider_cell_type']].copy()
    vcells[['UMAP_1', 'UMAP_2']] = vxy
    vcells.to_csv(OUT / 'CORDS_VCAF_COORDINATES.csv', index=False)
    qc = {}
    for p in sorted(OUT.glob('*.png')):
        with Image.open(p) as im:
            a = np.asarray(im.convert('RGB'))
            assert a.std() > 10
            qc[p.name] = {'size': list(im.size), 'nonblank': True}
    after = {str(p): sha(p) for p in inputs.values()}
    assert before == after
    summary = {'status': 'PASS', 'n_total_reference_cells': len(md), 'n_zheng': len(z),
               'n_cords_vcaf': len(vi), 'n_vcaf_patients': int(vcells.patient_id.nunique()),
               'input_sha256': before, 'inputs_unchanged': True,
               'refit_alignment': False, 'refit_umap': False, 'refit_predictor': False,
               'spatial_coordinates_used': False, 'jitter': False,
               'overlay_annotation': 'Original Cords vCAF label; no labels assigned to Zheng',
               'panels': audits, 'image_checks': qc,
               'script_sha256': sha(Path(__file__)),
               'figure_sha256': {p.name: sha(p) for p in OUT.glob('*') if p.suffix in {'.png', '.svg'}}}
    (OUT / 'VERIFICATION.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ['status', 'n_zheng', 'n_cords_vcaf', 'n_vcaf_patients', 'inputs_unchanged']}, indent=2))


if __name__ == '__main__':
    main()
