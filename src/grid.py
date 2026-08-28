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


def set_grid_x_lowerleftGP(config, fields):
    """Create grid with geometric progression of grid spacings in x direction, with higher resolution in the lower left corner.
    Grid convergence at 0.9^-1 ratio. """

    fields.dxc = np.array([
    # Constant Boundary (56 cells)
    0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,

    # Transition Down (20 steps)
    0.0070360, 0.0063367, 0.0057068, 0.0051396, 0.0046288, 0.0041687, 0.0037543, 0.0033812, 
    0.0030451, 0.0027424, 0.0024698, 0.0022243, 0.0020032, 0.0018041, 0.0016248, 0.0014633, 
    0.0013178, 0.0011868, 0.0010689, 0.0009626,

    # Single Center Cell
    0.0008696, #0.0008670,

    # Transition Up (20 steps)
    0.0009626, 0.0010689, 0.0011868, 0.0013178, 0.0014633, 0.0016248, 0.0018041, 0.0020032, 
    0.0022243, 0.0024698, 0.0027424, 0.0030451, 0.0033812, 0.0037543, 0.0041687, 0.0046288, 
    0.0051396, 0.0057068, 0.0063367, 0.0070360,

    # Constant Boundary (56 cells)
    0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,
    0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,
    0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,
    0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,
    0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,
    0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,
    0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,
    0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,
    0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,
    0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,
    0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,
    0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,
    0.0078125, 0.0078125
    ])

    fields.dxc = fields.dxc*(config.xmax - config.xmin)
    fields.xf[0] = config.xmin
    for i in range(config.nx-1):
        fields.xf[i+1] = fields.xf[i] + fields.dxc[i]

    fields.xc = fields.xf + 0.5*fields.dxc


def set_grid_y_lowerleftGP(config, fields):
    """Create grid with geometric progression of grid spacings in y direction, with higher resolution in the lower left corner.
    Grid convergence at 0.9^-1 ratio. """

    fields.dyc = np.array([
    # Constant Boundary (56 cells)
    0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,
    0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,
    0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,

    0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,

    # Transition Down (20 steps)
    0.0070360, 0.0063367, 0.0057068, 0.0051396, 0.0046288, 0.0041687, 0.0037543, 0.0033812, 
    0.0030451, 0.0027424, 0.0024698, 0.0022243, 0.0020032, 0.0018041, 0.0016248, 0.0014633, 
    0.0013178, 0.0011868, 0.0010689, 0.0009626,

    # Single Center Cell
    0.0008696, #0.0008670,

    # Transition Up (20 steps)
    0.0009626, 0.0010689, 0.0011868, 0.0013178, 0.0014633, 0.0016248, 0.0018041, 0.0020032, 
    0.0022243, 0.0024698, 0.0027424, 0.0030451, 0.0033812, 0.0037543, 0.0041687, 0.0046288, 
    0.0051396, 0.0057068, 0.0063367, 0.0070360,

    # Constant Boundary (56 cells)
    0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,
    0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,
    0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,
    0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,
    0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,
    0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,
    0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,  0.0078125, 0.0078125,
    0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,
    0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,
        0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125
    ])

    fields.dyc = fields.dyc*(config.ymax - config.ymin)
    #Ly = config.ymax - config.ymin
    #fields.dyc = fields.dyc / np.sum(fields.dyc) * Ly
    fields.yf[0] = config.ymin
    for i in range(config.ny-1):
        fields.yf[i+1] = fields.yf[i] + fields.dyc[i]

    fields.yc = fields.yf + 0.5*fields.dyc
