"""Resolve every (task, arm) source. Read-only against ../final_xu.

Three storage layouts were found during implementation; see RECORD.md
amendment 1 for the MaxFuse hub rule.
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent

SRC = (ROOT.parent / 'final_xu').resolve()
D = json.load(open(ROOT / 'design.json'))

STORY_DIR = {'lung_story': 'story_lung_HLA_ABC',
             'melanoma_story': 'figure_melanoma',
             'hnscc_story': 'figure_HNSCC'}
LAYOUT = {t['name']: t['layout'] for t in D['tasks']}


def root(task):
    return SRC / ('benchmark_results/' + task if LAYOUT[task] == 'benchmark'
                  else STORY_DIR[task] + '/Comparison')


def meta_path(task):
    return root(task) / D['layouts'][LAYOUT[task]]['metadata']


def marker_path(task):
    return root(task) / D['layouts'][LAYOUT[task]]['markers']


def arm_files(task, arm):
    """Files this arm needs. [] means derived from Raw at load time."""
    spec, r, lay = D['arms'][arm], root(task), LAYOUT[task]

    if spec['kind'] == 'recipe':
        return []

    if arm == 'fastMNN-reconstructed':
        if lay == 'benchmark':
            b = r / 'fastmnn/alignment/data'
            return [b / 'fastmnn_corrected_embedding.csv.gz',
                    b / 'fastmnn_pca_rotation.csv.gz']
        b = r / 'methods/fastmnn'
        # components in values.npy; rotation and centers inside fit.rds
        return [b / 'values.npy', b / 'fit.rds']

    if arm == 'MaxFuse':
        if lay == 'benchmark':
            # maxfuse_archive is the frozen directory the manuscript evaluates;
            # maxfuse/ and maxfuse_v2/ are earlier runs on a different cell set
            hubs = sorted(r.glob('maxfuse_archive/hub_*'))
            return [h / 'alignment/data/maxfuse_embedding_float64.npy' for h in hubs] + \
                   [h / 'alignment/data/cell_ids.npy' for h in hubs]
        if task == 'lung_story':
            hubs = sorted(r.glob('methods/maxfuse_hub*'))
            out = []
            for h in hubs:
                out += sorted(h.glob('view*_gcca_embedding.npy'))
                out += sorted(h.glob('view*_gcca_cell_ids.npy'))
            return out
        b = r / 'methods/maxfuse'
        return [b / f'cca_embedding{i}{s}.npy' for i in (1, 2) for s in ('', '_cell_ids')]

    key = 'benchmark_path' if lay == 'benchmark' else 'application_path'
    return [r / spec[key]]
