# This sets up an Arakawa C grid for 2D advection tests. Quadrilateral grid cells with doubly periodic boundary conditions. 

from numba import njit, prange
from src.jit import jitflags

import matplotlib.pyplot as plt


def grid_coordinates(config, fields): 
    xcc, ycc = (fields.xcc, fields.ycc)
    xcf, ycf = (fields.xcf, fields.ycf)
    xfc, yfc = (fields.xfc, fields.yfc)
    xffb, yffb = (fields.xffb, fields.yffb)
    xf, yf = (config.xf, config.yf)
    xc, yc = (config.xc, config.yc)
    nx, ny = (config.nx, config.ny)

    _grid_coordinates(xcc, ycc, xcf, ycf, xfc, yfc, xffb, yffb, xf, yf, xc, yc, nx, ny, config.xmax, config.ymax)


@njit(**jitflags)
def _grid_coordinates(xcc, ycc, xcf, ycf, xfc, yfc, xffb, yffb, xf, yf, xc, yc, nx, ny, xmax, ymax): 
    for jx in prange(0, nx):
        for jy in prange(0, ny):
                xcc[jx,jy] = xc[jx]
                xcf[jx,jy] = xc[jx]
                xfc[jx,jy] = xf[jx]
                xffb[jx,jy] = xf[jx]
                ycc[jx,jy] = yc[jy]
                ycf[jx,jy] = yf[jy]
                yfc[jx,jy] = yc[jy]
                yffb[jx,jy] = yf[jy]
    # Right and top boundaries for corner coordinates
    xffb[-1,:] = xmax
    yffb[:,-1] = ymax
    xffb[:-1,-1] = xfc[:,-1]
    yffb[-1,:-1] = ycf[-1,:]