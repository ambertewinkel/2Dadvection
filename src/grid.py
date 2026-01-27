# This sets up an Arakawa C grid for 2D advection tests. Quadrilateral grid cells with doubly periodic boundary conditions. 

from numba import njit, prange
from src.jit import jitflags
import numpy as np


def grid_coordinates(config, fields): 

    globals()[f"set_grid_x_{config.grid_x}"](config, fields)
    globals()[f"set_grid_y_{config.grid_y}"](config, fields)

    xcc, ycc = (fields.xcc, fields.ycc)
    xcf, ycf = (fields.xcf, fields.ycf)
    xfc, yfc = (fields.xfc, fields.yfc)
    xffb, yffb = (fields.xffb, fields.yffb)

    dxcc, dycc = (fields.dxcc, fields.dycc)

    xf, yf = (fields.xf, fields.yf)
    xc, yc = (fields.xc, fields.yc)
    nx, ny = (config.nx, config.ny)
    dxc, dyc = (fields.dxc, fields.dyc)

    _grid_coordinates(xcc, ycc, xcf, ycf, xfc, yfc, xffb, yffb, dxcc, dycc, xf, yf, xc, yc, nx, ny, dxc, dyc, config.xmax, config.ymax)


@njit(**jitflags)
def _grid_coordinates(xcc, ycc, xcf, ycf, xfc, yfc, xffb, yffb, dxcc, dycc, xf, yf, xc, yc, nx, ny, dxc, dyc, xmax, ymax): 
    for jx in prange(0, nx):
        for jy in prange(0, ny):
                xcc[jx,jy] = xc[jx]
                xcf[jx,jy] = xc[jx]
                xfc[jx,jy] = xf[jx]
                xffb[jx,jy] = xf[jx]
                dxcc[jx, jy] = dxc[jx] 
                dycc[jx, jy] = dyc[jy]
                ycc[jx,jy] = yc[jy]
                ycf[jx,jy] = yf[jy]
                yfc[jx,jy] = yc[jy]
                yffb[jx,jy] = yf[jy]
    # Right and top boundaries for corner coordinates
    xffb[-1,:] = xmax
    yffb[:,-1] = ymax
    xffb[:-1,-1] = xfc[:,-1]
    yffb[-1,:-1] = ycf[-1,:]


def set_grid_x_uniform(config, fields):
    """Create uniform grid spacing arrays for given config"""
    fields.xf, dxc_val = np.linspace(config.xmin, config.xmax, config.nx, endpoint=False, retstep=True)
    fields.dxc = np.full(config.nx, dxc_val)
    fields.xc = fields.xf + 0.5*fields.dxc


def set_grid_y_uniform(config, fields):
    """Create uniform grid spacing arrays for given config"""
    fields.yf, dyc_val = np.linspace(config.ymin, config.ymax, config.ny, endpoint=False, retstep=True)
    fields.dyc = np.full(config.ny, dyc_val)
    fields.yc = fields.yf + 0.5*fields.dyc


def set_grid_x_cosine(config, fields):
    """Create cosine stretched grid in x direction"""

    Lx = config.xmax - config.xmin

    fields.dxc = np.cos(2.*np.pi*np.arange(config.nx)/config.nx) + 1.5
    fields.dxc = fields.dxc / np.sum(fields.dxc) * Lx

    fields.xf[0] = config.xmin
    for i in range(config.nx-1):
        fields.xf[i+1] = fields.xf[i] + fields.dxc[i]

    fields.xc = fields.xf + 0.5*fields.dxc


def set_grid_y_cosine(config, fields):
    """Create cosine stretched grid in y direction"""

    Ly = config.ymax - config.ymin

    fields.dyc = np.cos(2.*np.pi*np.arange(config.ny)/config.ny) + 1.5
    fields.dyc = fields.dyc / np.sum(fields.dyc) * Ly

    fields.yf[0] = config.ymin
    for i in range(config.ny-1):
        fields.yf[i+1] = fields.yf[i] + fields.dyc[i]

    fields.yc = fields.yf + 0.5*fields.dyc