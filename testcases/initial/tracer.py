# Options for initial conditions for the tracer field

import numpy as np


def initial_tracer(config, fields):
    try:
       globals()[config.initial_tracer](config, fields)
    except KeyError:
        raise ValueError(f"Unknown initial tracer: {config.initial_tracer}")


def constant(config, fields):
    fields.tracer = np.full(fields.xcc.shape, config.constant_tracer)


def sine_xy(config, fields):
    fields.tracer = np.sin(2.*np.pi*fields.xcc/config.xmax) * np.sin(2.*np.pi*fields.ycc/config.ymax) + 1.
    

def cosine_bell(config, fields): 
    """
    Create a 2D cosine bell profile centered at (x_center, y_center) with given radius.

    Parameters:
    - x, y: 2D arrays of coordinates (e.g., from np.meshgrid)
    - xmin, xmax: domain limits in x
    - ymin, ymax: domain limits in y

    Returns:
    - 2D array of cosine bell values
    """
    radius = 0.15*min(config.xmax - config.xmin, config.ymax - config.ymin)
    # Compute distance from the center
    r = np.sqrt((fields.xcc - 0.5*config.xmax)*(fields.xcc - 0.5*config.xmax) + (fields.ycc - 0.5*config.ymax)*(fields.ycc - 0.5*config.ymax))

    # Apply cosine bell formula
    fields.tracer = np.where(r < radius, 0.5 * (1 + np.cos(np.pi * r / radius)), 0)


def cosine_bell_x(config, fields):
    """
    Create a 1D cosine bell profile centered at (0.5*xmax) with given radius uniform in y.

    Parameters:
    - x, y: 2D arrays of coordinates (e.g., from np.meshgrid)
    - xmin, xmax: domain limits in x

    Returns:
    - 2D array of cosine bell values
    """
    radius = 0.15*(config.xmax - config.xmin)
    # Compute distance from the center
    r = abs(fields.xcc - 0.5*config.xmax)

    # Apply cosine bell formula
    fields.tracer = np.where(r < radius, 0.5 * (1 + np.cos(np.pi * r / radius)), 0)

    
def cosine_bell_y(config, fields):
    """
    Create a 1D cosine bell profile centered at (0.5*ymax) with given radius uniform in x.

    Parameters:
    - x, y: 2D arrays of coordinates (e.g., from np.meshgrid)
    - xmin, xmax: domain limits in x

    Returns:
    - 2D array of cosine bell values
    """
    radius = 0.15*(config.ymax - config.ymin)
    # Compute distance from the center
    r = abs(fields.ycc - 0.5*config.ymax)

    # Apply cosine bell formula
    fields.tracer = np.where(r < radius, 0.5 * (1 + np.cos(np.pi * r / radius)), 0)

#def slotted_cylinder_swift(config,):
def sine_swift(config, fields):
    Lx = config.xmax - config.xmin
    Ly = config.ymax - config.ymin
    fields.tracer = config.mref + config.mmag*np.sin(2.*np.pi*fields.xcc/Lx)*np.sin(2.*np.pi*fields.ycc/Ly)


def square_wave_x(config, fields):
    Lx = config.xmax - config.xmin
    fields.tracer[:,:] = 0.
    fields.tracer[np.where((fields.xcc >= 0.25*Lx) & (fields.xcc <= 0.75*Lx))] = 1.


def square_wave_y(config, fields):
    Ly = config.ymax - config.ymin
    fields.tracer[:,:] = 0.
    fields.tracer[np.where((fields.ycc >= 0.25*Ly) & (fields.ycc <= 0.75*Ly))] = 1.


def square_wave_xy(config, fields):
    Lx = abs(config.xmax) + abs(config.xmin)
    Ly = abs(config.ymax) + abs(config.ymin)
    center_x = 0.5 * (config.xmax + config.xmin)
    center_y = 0.5 * (config.ymax + config.ymin)
    fields.tracer[:,:] = 0.
    fields.tracer[np.where((fields.xcc >= center_x - 0.25*Lx) & (fields.xcc <= center_x + 0.25*Lx) & (fields.ycc >= center_y - 0.25*Ly) & (fields.ycc <= center_y + 0.25*Ly))] = 1.


def square_wave_xy_displaced(config, fields):
    Lx = abs(config.xmax) + abs(config.xmin)
    Ly = abs(config.ymax) + abs(config.ymin)
    center_x = 0.5 * (config.xmax + config.xmin)-0.3*Lx
    center_y = 0.5 * (config.ymax + config.ymin)+0.3*Ly
    fields.tracer[:,:] = 0.
    fields.tracer[np.where((fields.xcc >= center_x - 0.25*Lx) & (fields.xcc <= center_x + 0.25*Lx) & (fields.ycc >= center_y - 0.25*Ly) & (fields.ycc <= center_y + 0.25*Ly))] = 1.


def gaussian_chenetal2017(config, fields):
    xc = 0.5 * (config.xmax + config.xmin)
    yc = 0.5 * (config.ymax + config.ymin)
    rcphi = 2500. # m
    rphi = 500. # m
    x_phi = xc
    y_phi = yc + rcphi
    fields.tracer = np.exp(-0.5*((fields.xcc - x_phi)**2 + (fields.ycc - y_phi)**2)/(rphi*rphi)) # [i,j] at i,j


def initial_blosseydurran(config, fields):
    r_tilde = 5.*np.sqrt((fields.xcc - 0.3)**2 + (fields.ycc - 0.5)**2)
    fields.tracer = np.where(r_tilde < 1., 0.25*(1. + np.cos(np.pi*r_tilde))*(1. + np.cos(np.pi*r_tilde)), 0.)