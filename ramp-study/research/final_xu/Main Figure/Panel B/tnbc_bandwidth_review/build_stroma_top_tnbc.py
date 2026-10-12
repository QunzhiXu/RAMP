"""Raise purple Stroma to the top layer, retaining the threefold-bandwidth TNBC design."""
from pathlib import Path
from functools import partial
import json
from build_refined_tnbc import (HERE,load_case,load_matrices,combined,sha,
                                refined_scatter,PALETTE,DRAW_ORDER)


def main():
    sources,audit={},[]
    meta,coords=load_case('tnbc',sources)
    markers,matrices=load_matrices('tnbc',meta,sources)
    order=[c for c in DRAW_ORDER if c!='Stroma']+['Stroma']
    draw=partial(refined_scatter,draw_order=order)
    layout=combined('tnbc',meta,coords,markers,matrices,audit,sigma_bins=6,
                    raw_display='arcsinh',compact=True,scatter_func=draw,cell_palette=PALETTE,
                    output_path=HERE/'tnbc_ridge_umap_sigma6_stroma_top.png')
    assert all(sha(Path(path))==digest for path,digest in sources.items())
    (HERE/'stroma_top_manifest.json').write_text(json.dumps({
        'input_sha256':sources,'inputs_unchanged':True,
        'change':'Move Stroma to the top drawing layer for every method',
        'celltype_draw_order_bottom_to_top':order,'celltype_palette':PALETTE,
        'previous_style_manifest':'refined_display_manifest.json',
        'other_display_parameters_unchanged':True,'ridge_sigma_bins':6,
        'all_cells_retained':True,'labels_changed':False,'umap_refit':False,
        'umap_coordinates_changed':False,'layout':layout,
    },indent=2)+'\n')


if __name__=='__main__':
    main()
