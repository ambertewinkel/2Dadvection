# This sets up an Arakawa C grid for 2D advection tests. Quadrilateral grid cells with doubly periodic boundary conditions. 

from numba import njit, prange
from src.jit import jitflags

import matplotlib.pyplot as plt


def grid_coordinates(config, fields): 
    xcc, ycc = (fields.xcc, fields.ycc)
    xcf, ycf = (fields.xcf, fields.ycf)
    xfc, yfc = (fields.xfc, fields.yfc)
    xffb, yffb = (fields.xffb, fields.yffb)

    dxcc, dycc = (fields.dxcc, fields.dycc)

    xf, yf = (config.xf, config.yf)
    xc, yc = (config.xc, config.yc)
    nx, ny = (config.nx, config.ny)
    dx, dy = (config.dx, config.dy) # For uniform grid, might need adjustment for nonuniform grid

    _grid_coordinates(xcc, ycc, xcf, ycf, xfc, yfc, xffb, yffb, dxcc, dycc, xf, yf, xc, yc, nx, ny, dx, dy, config.xmax, config.ymax)


@njit(**jitflags)
def _grid_coordinates(xcc, ycc, xcf, ycf, xfc, yfc, xffb, yffb, dxcc, dycc, xf, yf, xc, yc, nx, ny, dx, dy, xmax, ymax): 
    for jx in prange(0, nx):
        for jy in prange(0, ny):
                xcc[jx,jy] = xc[jx]
                xcf[jx,jy] = xc[jx]
                xfc[jx,jy] = xf[jx]
                xffb[jx,jy] = xf[jx]
                dxcc[jx, jy] = dx # For uniform grid, needs adjustment for nonuniform grid
                dycc[jx, jy] = dy # For uniform grid, needs adjustment for nonuniform grid
                ycc[jx,jy] = yc[jy]
                ycf[jx,jy] = yf[jy]
                yfc[jx,jy] = yc[jy]
                yffb[jx,jy] = yf[jy]
    # Right and top boundaries for corner coordinates
    xffb[-1,:] = xmax
    yffb[:,-1] = ymax
    xffb[:-1,-1] = xfc[:,-1]
    yffb[-1,:-1] = ycf[-1,:]