"""Pack the TNBC UMAP pair using its actual aspect ratio, preserving all coordinates."""
from pathlib import Path
import os
import sys
import json

HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE.parent/'bandwidth_review/.cache/matplotlib'))
sys.path.insert(0,str(HERE.parent/'benchmark_review'))
from build_ridge_review import load_case,load_matrices,combined,sha


def main():
    sources,audit={},[]
    meta,coords=load_case('tnbc',sources)
    markers,matrices=load_matrices('tnbc',meta,sources)
    layout=combined('tnbc',meta,coords,markers,matrices,audit,sigma_bins=4,
                    raw_display='arcsinh',compact=True,
                    output_path=HERE/'tnbc_ridge_umap_sigma4_compact.png')
    assert all(sha(Path(path))==value for path,value in sources.items())
    (HERE/'compact_layout_manifest.json').write_text(json.dumps({
        'input_sha256':sources,'inputs_unchanged':True,
        'previous_figure_width_in':14,'layout':layout,
        'smoothing_sigma_bins':4,'histogram_bins':160,
        'raw_ridge_display_transform':'arcsinh(x/1)',
        'umap_coordinates_changed':False,'umap_axis_limits_changed':False,
        'umap_equal_aspect_preserved':True,'cells_removed':0,
        'note':'Physical UMAP heights preserved; axes are fit to their data aspect ratio and placed 0.10 inch apart.',
    },indent=2)+'\n')
    print(json.dumps(layout,indent=2),flush=True)


if __name__=='__main__':
    main()
