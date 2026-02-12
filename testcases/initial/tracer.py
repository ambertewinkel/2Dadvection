# Options for initial conditions for the tracer field

import numpy as np


def initial_tracer(config, fields):
    try:
       globals()[config.initial_tracer](config, fields, it=0)
    except KeyError:
        raise ValueError(f"Unknown initial tracer: {config.initial_tracer}")


def constant(config, fields, it):
    fields.tracer[it] = np.full(fields.xcc.shape, config.constant_tracer)


def sine_xy(config, fields, it):
    fields.tracer[it] = np.sin(2.*np.pi*fields.xcc/config.xmax) * np.sin(2.*np.pi*fields.ycc/config.ymax) + 1.
    

def cosine_bell(config, fields, it): 
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
    fields.tracer[it] = np.where(r < radius, 0.5 * (1 + np.cos(np.pi * r / radius)), 0)


def cosine_bell_x(config, fields, it):
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
    fields.tracer[it] = np.where(r < radius, 0.5 * (1 + np.cos(np.pi * r / radius)), 0)

    
def cosine_bell_y(config, fields, it):
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
    fields.tracer[it] = np.where(r < radius, 0.5 * (1 + np.cos(np.pi * r / radius)), 0)

#def slotted_cylinder_swift(config,):
def sine_swift(config, fields, it):
    Lx = config.xmax - config.xmin
    Ly = config.ymax - config.ymin
    fields.tracer[it] = config.mref + config.mmag*np.sin(2.*np.pi*fields.xcc/Lx)*np.sin(2.*np.pi*fields.ycc/Ly)


def sine_x(config, fields, it):
    Lx = config.xmax - config.xmin
    fields.tracer[it] = config.mref + config.mmag*np.sin(2.*np.pi*fields.xcc/Lx)
    

def square_wave_x(config, fields, it):
    Lx = config.xmax - config.xmin
    fields.tracer[it] = 0.
    fields.tracer[it] = np.where((fields.xcc >= 0.25*Lx) & (fields.xcc <= 0.75*Lx), 1., 0.)


def square_wave_y(config, fields, it):
    Ly = config.ymax - config.ymin
    fields.tracer[it] = 0.
    fields.tracer[it] = np.where((fields.ycc >= 0.25*Ly) & (fields.ycc <= 0.75*Ly), 1., 0.)


def square_wave_xy(config, fields, it):
    Lx = abs(config.xmax) + abs(config.xmin)
    Ly = abs(config.ymax) + abs(config.ymin)
    center_x = 0.5 * (config.xmax + config.xmin)
    center_y = 0.5 * (config.ymax + config.ymin)
    fields.tracer[it,:,:] = 0.
    fields.tracer[it, np.where((fields.xcc >= center_x - 0.25*Lx) & (fields.xcc <= center_x + 0.25*Lx) & (fields.ycc >= center_y - 0.25*Ly) & (fields.ycc <= center_y + 0.25*Ly))] = 1.


def square_wave_xy_displaced(config, fields, it):
    Lx = abs(config.xmax) + abs(config.xmin)
    Ly = abs(config.ymax) + abs(config.ymin)
    center_x = 0.5 * (config.xmax + config.xmin)-0.3*Lx
    center_y = 0.5 * (config.ymax + config.ymin)+0.3*Ly
    fields.tracer[it,:,:] = 0.
    fields.tracer[np.where((fields.xcc >= center_x - 0.25*Lx) & (fields.xcc <= center_x + 0.25*Lx) & (fields.ycc >= center_y - 0.25*Ly) & (fields.ycc <= center_y + 0.25*Ly))] = 1.


def gaussian_chenetal2017(config, fields, it):
    xc = 0.5 * (config.xmax + config.xmin)
    yc = 0.5 * (config.ymax + config.ymin)
    rcphi = 2500. # m
    rphi = 500. # m
    x_phi = xc
    y_phi = yc + rcphi
    fields.tracer[it] = np.exp(-0.5*((fields.xcc - x_phi)**2 + (fields.ycc - y_phi)**2)/(rphi*rphi)) # [i,j] at i,j


def initial_blosseydurran(config, fields, it):
    r_tilde = 5.*np.sqrt((fields.xcc - 0.3)**2 + (fields.ycc - 0.5)**2)
    fields.tracer[it] = np.where(r_tilde < 1., 0.25*(1. + np.cos(np.pi*r_tilde))*(1. + np.cos(np.pi*r_tilde)), 0.)


def surface_cosine_bell(config, fields, it):
    """
    Create a 2D cosine bell profile centered at (0.5, 0.1) with given radius.

    Parameters:
    - x, y: 2D arrays of coordinates (e.g., from np.meshgrid)
    - xmin, xmax: domain limits in x
    - ymin, ymax: domain limits in y

    Returns:
    - 2D array of cosine bell values
    """
    x_center = 0.5 * (config.xmax + config.xmin)
    y_center = 0.1 * (config.ymax - config.ymin) + config.ymin
    radius = 0.1*min(config.xmax - config.xmin, config.ymax - config.ymin)
    # Compute distance from the center
    r = np.sqrt((fields.xcc - x_center)*(fields.xcc - x_center) + (fields.ycc - y_center)*(fields.ycc - y_center))

    # Apply cosine bell formula
    fields.tracer[it] = np.where(r < radius, 0.5 * (1 + np.cos(np.pi * r / radius)), 0)

def doswell_adapted(config, fields, it):
    fields.tracer[it] = 0.5*(1. - np.tanh(2.*np.pi*(fields.ycc)/(config.ymax - config.ymin)))
    #fields.tracer[it] = - np.tanh((fields.ycc - config.ymin)/(config.ymax - config.ymin)) # adapted from paper to keep positive


def hadley(config, fields, it):
    # Kent et al 2014: eq 41
    
    z1, z2 = 2000., 5000. # m, lower and upper boundary of tracer layer
    z0 = 0.5*(z1 + z2)
    fields.tracer[it] = 0.5*(1. + np.cos(2.*np.pi*(fields.ycc-z0)/(z2 - z1)))
    fields.tracer[it] = np.where((fields.ycc < z1) | (fields.ycc > z2), 0., fields.tracer[it])#, np.where((fields.ycc < z1) | (fields.ycc > z2))] = 0.


def slotted_cylinder(config, fields, it): 
    # Single one of these used in AdLImHEx paper (HW) and perhaps SWIFT.
    # Assumes Lx = Ly and centered at (0.5*Lx, 0.5*Ly)

    Lx = config.xmax - config.xmin
    Ly = config.ymax - config.ymin
    #L = 1000. # m, radius of cylinder
    rc = 160. # m
    lc = 25. # m

    r = np.sqrt((fields.xcc - 0.5*Lx - config.xmin)**2 + (fields.ycc - 0.5*Ly - config.ymin)**2)
    fields.tracer[it] = np.where(r < rc, 1., 0.)
    fields.tracer[it] = np.where((fields.ycc > 0.5*Ly + config.ymin) & (abs(fields.xcc - 0.5*Lx - config.xmin) < lc), 0., fields.tracer[it])


def slotted_cylinders(config, fields, it): 
    # Used in AdLImHEx paper (HW) and perhaps SWIFT.
    # Assumes Lx = Ly

    Lx = config.xmax - config.xmin
    Ly = config.ymax - config.ymin
    #L = 1000. # m, radius of cylinder
    rc = 160. # m
    lc = 25. # m

    xc1 = 0.25*Lx + config.xmin
    yc = 0.5*Ly + config.ymin
    xc2 = 0.75*Lx + config.xmin

    r1 = np.sqrt((fields.xcc - xc1)**2 + (fields.ycc - yc)**2)
    r2 = np.sqrt((fields.xcc - xc2)**2 + (fields.ycc - yc)**2)

    fields.tracer[it] = np.where(r1 < rc, 1., 0.)
    fields.tracer[it] = np.where(r2 < rc, 1., fields.tracer[it])
    fields.tracer[it] = np.where((fields.ycc > 0.5*Ly + config.ymin) & (abs(fields.xcc - xc1) < lc), 0., fields.tracer[it])
    fields.tracer[it] = np.where((fields.ycc > 0.5*Ly + config.ymin) & (abs(fields.xcc - xc2) < lc), 0., fields.tracer[it])
