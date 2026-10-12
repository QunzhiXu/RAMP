"""Established fixed platform projection; no new patients enter its fit."""
import numpy as np
from sklearn.isotonic import IsotonicRegression

EPS=1e-6
LIMIT=.75


def fit_maps(b,a):
    return [(model.X_thresholds_,model.y_thresholds_) for j in range(b.shape[1])
            for model in [IsotonicRegression(out_of_bounds='clip').fit(b[:,j],a[:,j])]]


def apply_maps(b,maps):
    a=np.empty_like(b)
    for j,(x,y) in enumerate(maps):
        v=b[:,j]
        z=np.interp(v,x,y)
        z=np.where(v<x[0],v+y[0]-x[0],z)
        z=np.where(v>x[-1],v+y[-1]-x[-1],z)
        z=(1-EPS)*z+EPS*v
        a[:,j]=v+np.clip(z-v,-LIMIT,LIMIT)
    return a
