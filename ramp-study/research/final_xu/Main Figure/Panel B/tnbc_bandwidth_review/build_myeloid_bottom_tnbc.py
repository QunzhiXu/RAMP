"""Place Myeloid lowest and Tumor second lowest in all TNBC cell-type UMAPs."""
from pathlib import Path
from functools import partial
import json
from build_refined_tnbc import (HERE,load_case,load_matrices,combined,sha,
                                refined_scatter,PALETTE)

ORDER=['Myeloid','Tumor / epithelial','Non-joint','Endothelial',
       'B lineage','NK','CD4 T','CD8 T','Stroma']


def main():
    sources,audit={},[]
    meta,coords=load_case('tnbc',sources)
    markers,matrices=load_matrices('tnbc',meta,sources)
    layout=combined('tnbc',meta,coords,markers,matrices,audit,sigma_bins=6,
                    raw_display='arcsinh',compact=True,
                    scatter_func=partial(refined_scatter,draw_order=ORDER),
                    cell_palette=PALETTE,
                    output_path=HERE/'tnbc_ridge_umap_sigma6_myeloid_bottom.png')
    assert all(sha(Path(path))==digest for path,digest in sources.items())
    (HERE/'myeloid_bottom_manifest.json').write_text(json.dumps({
        'input_sha256':sources,'inputs_unchanged':True,
        'change':'Myeloid lowest, Tumor second lowest, Stroma highest, for every method',
        'celltype_draw_order_bottom_to_top':ORDER,'celltype_palette':PALETTE,
        'previous_style_manifest':'stroma_top_manifest.json',
        'other_display_parameters_unchanged':True,'ridge_sigma_bins':6,
        'all_cells_retained':True,'labels_changed':False,'umap_refit':False,
        'umap_coordinates_changed':False,'layout':layout,
    },indent=2)+'\n')


if __name__=='__main__':
    main()
