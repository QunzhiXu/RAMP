"""Reduce TNBC method-row gaps while preserving the physical size of each UMAP."""
from pathlib import Path
from functools import partial
import json
from build_refined_tnbc import (HERE,load_case,load_matrices,combined,sha,
                                refined_scatter,PALETTE)
from build_myeloid_bottom_tnbc import ORDER


def main():
    sources,audit={},[]
    meta,coords=load_case('tnbc',sources)
    markers,matrices=load_matrices('tnbc',meta,sources)
    layout=combined('tnbc',meta,coords,markers,matrices,audit,sigma_bins=6,
                    raw_display='arcsinh',compact=True,compact_rows=True,
                    scatter_func=partial(refined_scatter,draw_order=ORDER),
                    cell_palette=PALETTE,
                    output_path=HERE/'tnbc_ridge_umap_sigma6_tight_rows.png')
    assert all(sha(Path(path))==digest for path,digest in sources.items())
    (HERE/'tight_rows_manifest.json').write_text(json.dumps({
        'input_sha256':sources,'inputs_unchanged':True,
        'previous_style_manifest':'myeloid_bottom_manifest.json',
        'celltype_draw_order_bottom_to_top':ORDER,
        'ridge_sigma_bins':6,'umap_coordinates_changed':False,'all_cells_retained':True,
        'change':'Shorter row gaps and top/bottom margins; move Raw arcsinh label to row name',
        'previous_figure_height_in':11.8,'previous_row_gap_in':.3950434782608696,
        'umap_physical_size_preserved':True,'layout':layout,
    },indent=2)+'\n')
    print(json.dumps(layout,indent=2),flush=True)


if __name__=='__main__':
    main()
