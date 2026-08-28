"""This file includes functions for the FCT limiter."""


import numpy as np
import src.schemes as sch


def set_extrema_2D(config, fields, it, field_bounded):
    """This function returns the min and max values allowed for each grid cell, defined at center. Used in FCT."""

    fieldmin, fieldmax = np.zeros_like(field_bounded), np.zeros_like(field_bounded)
    use_previous = np.all(fields.Ccc[it] <= 1.) # whether to use previous time step field to set extrema (True if all C in space <= 1 else False)

    # Determine allowable minima
    if config.tracermin is not None: # global allowable min/max (nonnegative FCT)
        fieldmin = np.full_like(field_bounded, config.tracermin)
    else: # AdImEx FCT
        if use_previous:
            fieldmin = np.minimum.reduce([np.roll(field_bounded,1,0), field_bounded, np.roll(field_bounded,-1,0), np.roll(field_bounded,1,1), np.roll(field_bounded,-1,1), np.roll(fields.tracer[it],1,0), fields.tracer[it], np.roll(fields.tracer[it],-1,0), np.roll(fields.tracer[it],1,1), np.roll(fields.tracer[it],-1,1)]) # at [i,j]
        else:
            fieldmin = np.minimum.reduce([np.roll(field_bounded,1,0), field_bounded, np.roll(field_bounded,-1,0), np.roll(field_bounded,1,1), np.roll(field_bounded,-1,1)]) # at [i,j]

    # Determine allowable maxima
    if config.tracermax is not None: # global allowable min/max (nonnegative FCT)
        fieldmax = np.full_like(field_bounded, config.tracermax)
    else: # AdImEx FCT
        if use_previous:
            fieldmax = np.maximum.reduce([np.roll(field_bounded,1,0), field_bounded, np.roll(field_bounded,-1,0), np.roll(field_bounded,1,1), np.roll(field_bounded,-1,1), np.roll(fields.tracer[it],1,0), fields.tracer[it], np.roll(fields.tracer[it],-1,0), np.roll(fields.tracer[it],1,1), np.roll(fields.tracer[it],-1,1)]) # at [i,j]
        else:
            fieldmax = np.maximum.reduce([np.roll(field_bounded,1,0), field_bounded, np.roll(field_bounded,-1,0), np.roll(field_bounded,1,1), np.roll(field_bounded,-1,1)]) # at [i,j]

    return fieldmin, fieldmax


def FCT2D(config, fields, flxfc, flxfc_bounded, flxcf, flxcf_bounded, fieldmin, fieldmax, field_bounded):
    """This function implements flux-corrected transport (FCT). Either using global bounds or local bounds based on low-order bounded solution and optionally previous time step. Returns the limited field at the new time step. Assumes constant dxc and uf>0.
    flxfc : high-order flux at [i-1/2,j]
    flxfc_bounded : low-order bounded flux at [i-1/2,j]
    flxcf : high-order flux at [i,j-1/2]
    flxcf_bounded : low-order bounded flux at [i,j-1/2]
    fieldmin : minimum allowable field at [i,j]
    fieldmax : maximum allowable field at [i,j]
    field_bounded : low-order bounded field at [i,j]
    """
    # Calculate high-order correction
    corrfc = (flxfc - flxfc_bounded)*fields.dycc # at [i-1/2,j]
    corrcf = (flxcf - flxcf_bounded)*fields.dxcc # at [i,j-1/2]

    # Calculate allowable mass I/O for max rise and fall
    Qp = fields.dxcc*fields.dycc*(fieldmax - field_bounded) # at [i,j]
    Qm = fields.dxcc*fields.dycc*(field_bounded - fieldmin) # at [i,j]

    # Calculate I/O fluxes at cell centers
    Pp = config.dt*(np.maximum(0, corrfc) - np.minimum(0, np.roll(corrfc,-1,0)) + np.maximum(0, corrcf) - np.minimum(0, np.roll(corrcf,-1,1)))
    Pm = config.dt*(np.maximum(0, np.roll(corrfc,-1,0)) - np.minimum(0, corrfc) + np.maximum(0, np.roll(corrcf,-1,1)) - np.minimum(0, corrcf))

    # Calculate ratios of allowable (Q) to existing high-order (P) fluxes
    Rp = np.where(Pp > 1e-15, np.minimum(1., Qp/np.maximum(Pp,1e-15)), 0.)
    Rm = np.where(Pm > 1e-15, np.minimum(1., Qm/np.maximum(Pm,1e-15)), 0.)

    # Calculate the limiter for each face
    face_limiter_fc = np.where(corrfc >= 0., np.minimum(Rp, np.roll(Rm,1,0)), np.minimum(np.roll(Rp,1,0), Rm)) # at [i-1/2,j]
    face_limiter_cf = np.where(corrcf >= 0., np.minimum(Rp, np.roll(Rm,1,1)), np.minimum(np.roll(Rp,1,1), Rm)) # at [i,j-1/2]

    # Update the bounded flux and field
    flx_corr_fc = flxfc_bounded + face_limiter_fc*corrfc/fields.dycc # at [i-1/2,j]
    flx_corr_cf = flxcf_bounded + face_limiter_cf*corrcf/fields.dxcc # at [i,j-1/2]

    return flx_corr_fc, flx_corr_cf


def FCT(config, fields, it, flxfc_HO, flxcf_HO, **kwargs):
    """This function implements a reduced version of flux-corrected transport (FCT) that only does one pass (see Zalesak 1979 p.347-349). Returns the limited field at the new time step. Assumes constant grid spacing and u>0.
    it : time index
    flxfc_HO : high-order flux at [i-1/2,j]
    flxcf_HO : high-order flux at [i,j-1/2]
    """

    # Calculate 2D first-order solution if not using both global bounds
    sch.adimex_upwind(config, fields, it, tolerance=1e-15, kiter=200, jiter=4, **kwargs) # at [i,j] # temporarily writes in fields.tracer[it+1] but will be overwritten by the FCT solution at the end of this function

    flxfc_bounded = np.maximum(0.,fields.u[it])*(1.-fields.thetafc[it])*np.roll(fields.tracer[it],1,0) + np.minimum(0.,fields.u[it])*(1.-fields.thetafc[it])*fields.tracer[it] + np.maximum(0.,fields.u[it])*fields.thetafc[it]*np.roll(fields.tracer[it+1],1,0) + np.minimum(0.,fields.u[it])*fields.thetafc[it]*fields.tracer[it+1] # at [i-1/2,j]
    flxcf_bounded = np.maximum(0.,fields.v[it])*(1.-fields.thetacf[it])*np.roll(fields.tracer[it],1,1) + np.minimum(0.,fields.v[it])*(1.-fields.thetacf[it])*fields.tracer[it] + np.maximum(0.,fields.v[it])*fields.thetacf[it]*np.roll(fields.tracer[it+1],1,1) + np.minimum(0.,fields.v[it])*fields.thetacf[it]*fields.tracer[it+1] # at [i,j-1/2]

    # Apply 2D FCT using the limited fluxes (necessary to avoid all ripples, see Zalesak 1979 p.350)
    min2D, max2D = set_extrema_2D(config, fields, it, fields.tracer[it+1]) # at [i,j]

    flx_corr_fc, flx_corr_cf = FCT2D(config, fields, flxfc_HO, flxfc_bounded, flxcf_HO, flxcf_bounded, min2D, max2D, fields.tracer[it+1]) # at [i-1/2,j] and at [i,j-1/2]
    
    return fields.tracer[it] + config.dt*(sch.fluxdiv(fields, flx_corr_fc, flx_corr_cf, 1., 1.)) # at [i,j]

