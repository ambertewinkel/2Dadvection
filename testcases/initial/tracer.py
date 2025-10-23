# Options for initial conditions for the tracer field

import numpy as np


def initial_tracer(config, fields):
    try:
       globals()[config.initial_tracer](config, fields)
    except KeyError:
        raise ValueError(f"Unknown initial tracer: {config.initial_tracer}")


def constant(config, fields):
    fields.tracer = config.constant_tracer


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

    