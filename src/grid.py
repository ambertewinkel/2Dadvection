# This sets up an Arakawa C grid for 2D advection tests. Quadrilateral grid cells with doubly periodic boundary conditions. 

from numba import njit, prange
from src.jit import jitflags


def grid_coordinates(config, fields): 
    xcc, ycc = (fields.xcc, fields.ycc)
    xcf, ycf = (fields.xcf, fields.ycf)
    xfc, yfc = (fields.xfc, fields.yfc)
    xf, yf = (config.xf, config.yf)
    xc, yc = (config.xc, config.yc)
    nx, ny = (config.nx, config.ny)

    _grid_coordinates(xcc, ycc, xcf, ycf, xfc, yfc, xf, yf, xc, yc, nx, ny)


@njit(**jitflags)
def _grid_coordinates(xcc, ycc, xcf, ycf, xfc, yfc, xf, yf, xc, yc, nx, ny): 
    for jx in prange(0, nx):
        for jy in prange(0, ny):
                xcc[jx,jy] = xc[jx]
                xcf[jx,jy] = xc[jx]
                xfc[jx,jy] = xf[jx]
                ycc[jx,jy] = yc[jy]
                ycf[jx,jy] = yf[jy]
                yfc[jx,jy] = yc[jy]






