# Options for initial conditions for the tracer field

import numpy as np


def initial_tracer(config, fields):
    try:
       globals()[config.initial_tracer](config, fields, it=0)
    except KeyError:
        raise ValueError(f"Unknown initial tracer: {config.initial_tracer}")


def constant(config, fields, it):
    fields.tracer[it] = np.full(fields.xcc.shape, config.constant_tracer)


def sine_swift(config, fields, it):
    Lx = config.xmax - config.xmin
    Ly = config.ymax - config.ymin
    fields.tracer[it] = config.mref + config.mmag*np.sin(2.*np.pi*fields.xcc/Lx)*np.sin(2.*np.pi*fields.ycc/Ly)


def hadley(config, fields, it):
    """Inspired by Hadley-like circulation in Kent et al. 2014."""    

    z1, z2 = 2000., 5000. # m, lower and upper boundary of tracer layer
    z0 = 0.5*(z1 + z2)
    fields.tracer[it] = 0.5*(1. + np.cos(2.*np.pi*(fields.ycc-z0)/(z2 - z1)))
    fields.tracer[it] = np.where((fields.ycc < z1) | (fields.ycc > z2), 0., fields.tracer[it])


def slotted_cylinders(config, fields, it): 
    # Assumes Lx = Ly

    Lx = config.xmax - config.xmin
    Ly = config.ymax - config.ymin
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
