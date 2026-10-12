"""Matched cell-coordinate plots and a complete 119-core gallery."""
import package_runtime as rt
from package_runtime import BASE
ROOT=BASE.parents[1]
import json,time,hashlib,html
import numpy as np
import pandas as pd
import h5py
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

OUT=BASE/'results'
FIG=OUT/'cores'
COLS=['observed_HLA_A_patient_rank','RAMP_k30']

def pair_axes(axes,g,rho,point_size=4.2):
    seed=int.from_bytes(hashlib.sha256(str(g.core_id.iloc[0]).encode()).digest()[:8],'little')
    z=g.iloc[np.random.default_rng(seed).permutation(len(g))]
    xy=z[['x_um','y_um']].to_numpy()
    low=xy.min(axis=0);high=xy.max(axis=0);center=(low+high)/2
    half=(high-low).max()/2+65
    xlim=(center[0]-half,center[0]+half);ylim=(center[1]+half,center[1]-half)
    norm=Normalize(0,1);artists=[]
    for ax,col,title in zip(axes,COLS,['Measured HLA-A','Predicted HLA-A proxy']):
        artist=ax.scatter(xy[:,0],xy[:,1],c=z[col].to_numpy(),s=point_size,cmap='viridis',norm=norm,
                          edgecolors='none',linewidths=0,alpha=1,rasterized=True)
        ax.set_xlim(xlim);ax.set_ylim(ylim);ax.set_aspect('equal');ax.axis('off')
        ax.set_title(title,fontsize=15,pad=10)
        bx=xlim[0]+30;by=ylim[0]-27
        ax.plot([bx,bx+100],[by,by],color='#29313B',lw=2.7,solid_capstyle='butt')
        ax.text(bx+50,by-12,'100 µm',ha='center',va='bottom',fontsize=9,color='#29313B')
        assert len(artist.get_offsets())==len(g)
        assert np.array_equal(np.asarray(artist.get_offsets()),xy)
        assert np.array_equal(np.asarray(artist.get_array()),z[col].to_numpy())
        artists.append(artist)
    assert np.array_equal(artists[0].get_offsets(),artists[1].get_offsets())
    assert axes[0].get_xlim()==axes[1].get_xlim() and axes[0].get_ylim()==axes[1].get_ylim()
    return artists[0],{'patient_id':g.patient_id.iloc[0],'core_id':g.core_id.iloc[0],'n_plotted_each_panel':len(g),
        'xlim':xlim,'ylim':ylim,'same_cell_ids_coordinates_order_limits':True,'color_vmin':0,'color_vmax':1,
        'predicted_color_rescaled':False,'point_size_pt2':point_size,'source_row_ids_sha256':hashlib.sha256(z.source_row_id.to_numpy(np.int64).tobytes()).hexdigest(),
        'cell_spearman':rho}

def add_colorbar(fig,artist,rect):
    cb=fig.colorbar(artist,cax=fig.add_axes(rect),orientation='horizontal',ticks=[0,.25,.5,.75,1])
    cb.set_label('Relative expression rank / predicted rank',fontsize=10)
    cb.ax.tick_params(labelsize=9);cb.outline.set_linewidth(.4)
