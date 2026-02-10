"""This file includes functions for the FCT limiter."""


import numpy as np
import src.schemes as sch


def set_extrema_1D(config, fields, it, field_bounded, axis):
    """This function returns the min and max values allowed for each grid cell, defined at center. Used in FCT."""

    fieldmin, fieldmax = np.zeros_like(field_bounded), np.zeros_like(field_bounded)
    use_previous = np.all(fields.Ccc[it] <= 1.) # whether to use previous time step field to set extrema (True if all C in space <= 1 else False)

    # Determine allowable minima
    if config.tracermin is not None: # global allowable min/max (nonnegative FCT)
        fieldmin = np.full_like(field_bounded, config.tracermin)
    else: # AdImEx FCT
        if use_previous:
            fieldmin = np.minimum.reduce([np.roll(field_bounded,1,axis), field_bounded, np.roll(field_bounded,-1,axis), np.roll(fields.tracer[it],1,axis), fields.tracer[it], np.roll(fields.tracer[it],-1,axis)]) # at [i,j]
        else:
            fieldmin = np.minimum.reduce([np.roll(field_bounded,1,axis), field_bounded, np.roll(field_bounded,-1,axis)]) # at [i,j]

    # Determine allowable maxima
    if config.tracermax is not None: # global allowable min/max (nonnegative FCT)
        fieldmax = np.full_like(field_bounded, config.tracermax)
    else: # AdImEx FCT
        if use_previous:
            fieldmax = np.maximum.reduce([np.roll(field_bounded,1,axis), field_bounded, np.roll(field_bounded,-1,axis), np.roll(fields.tracer[it],1,axis), fields.tracer[it], np.roll(fields.tracer[it],-1,axis)]) # at [i,j]
        else:
            fieldmax = np.maximum.reduce([np.roll(field_bounded,1,axis), field_bounded, np.roll(field_bounded,-1,axis)]) # at [i,j]

    return fieldmin, fieldmax


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


def FCT1D(config, fields, flx_HO, flx_bounded, fieldmin, fieldmax, field_bounded, axis, d_axis):
    """!!! update docstring 
    This function implements flux-corrected transport (FCT) in 1D. Either using global bounds or local bounds based on low-order bounded solution and optionally previous time step. Returns the limited field at the new time step. Assumes constant dxc and uf>0.
    flx_HO: high-order flux at i-1/2
    flx_bounded: low-order bounded flux at i-1/2
    V : volume of cell i
    N : number of cells in the 1D direction
    dt : time step
    field_previous : field at previous time step n, at i
    use_previous : whether to use previous time step field to set extrema (True if all C in space <= 1 else False)
    ymin, ymax : allowable min and max values for the field. If None, local extrema are used.
    nondivergent : whether the winds are nondivergent (True/False)
    """
    # Calculate high-order correction
    corr = (flx_HO - flx_bounded)*d_axis # at [i-1/2,j] if axis=0, at [i,j-1/2] if axis=1

    if config.tracermin is None and config.tracermax is None:
        # Checking for rare cases where we need to set the correction to zero
        for i in range(np.shape(field_bounded)[0]):
            for j in range(np.shape(field_bounded)[1]):
                if corr[i,j]*(field_bounded[i,j] - np.roll(field_bounded,1,axis)[i,j]) <= 0. and (corr[i,j]*(np.roll(field_bounded,-1,axis)[i,j] - field_bounded[i,j]) <= 0. or corr[i,j]*(np.roll(field_bounded,1,axis)[i,j] - np.roll(field_bounded,2,axis)[i,j]) <= 0.):
                    corr[i,j] = 0. 

    # Calculate allowable mass I/O for max rise and fall
    Qp = fields.dxcc*fields.dycc*(fieldmax - field_bounded) # at [i,j]
    Qm = fields.dxcc*fields.dycc*(field_bounded - fieldmin) # at [i,j]

    # Calculate I/O fluxes at cell centers
    Pp = config.dt*(np.maximum(0, corr) - np.minimum(0, np.roll(corr,-1,axis)))
    Pm = config.dt*(np.maximum(0, np.roll(corr,-1,axis)) - np.minimum(0, corr))

    # Calculate ratios of allowable (Q) to existing high-order (P) fluxes
    Rp = np.where(Pp > 1e-12, np.minimum(1., Qp/np.maximum(Pp,1e-12)), 0.)
    Rm = np.where(Pm > 1e-12, np.minimum(1., Qm/np.maximum(Pm,1e-12)), 0.)

    # Calculate the limiter for each face
    face_limiter = np.where(corr >= 0., np.minimum(Rp, np.roll(Rm,1,axis)), np.minimum(np.roll(Rp,1,axis), Rm)) # at [i-1/2,j] if axis=0, at [i,j-1/2] if axis=1

    # Update the bounded flux and field
    flx_corr = flx_bounded + face_limiter*corr/d_axis

    return flx_corr


def FCT2D(config, fields, flxfc, flxfc_bounded, flxcf, flxcf_bounded, fieldmin, fieldmax, field_bounded):
    """2D !!!!!! update docstring
    This function implements flux-corrected transport (FCT). Either using global bounds or local bounds based on low-order bounded solution and optionally previous time step. Returns the limited field at the new time step. Assumes constant dxc and uf>0.
    flx_HO: high-order flux at i-1/2
    flx_bounded: low-order bounded flux at i-1/2
    V : volume of cell i
    N : number of cells in the 1D direction
    dt : time step
    field_previous : field at previous time step n, at i
    use_previous : whether to use previous time step field to set extrema (True if all C in space <= 1 else False)
    ymin, ymax : allowable min and max values for the field. If None, local extrema are used.
    nondivergent : whether the winds are nondivergent (True/False)
    """
    # Calculate high-order correction
    corrfc = (flxfc - flxfc_bounded)*fields.dycc # at [i-1/2,j]
    corrcf = (flxcf - flxcf_bounded)*fields.dxcc # at [i,j-1/2]

    if config.tracermin is None and config.tracermax is None:
        # Checking for rare cases where we need to set the correction to zero
        for i in range(np.shape(field_bounded)[0]):
            for j in range(np.shape(field_bounded)[1]):
                if corrfc[i,j]*(field_bounded[i,j] - np.roll(field_bounded,1,0)[i,j]) <= 0. and (corrfc[i,j]*(np.roll(field_bounded,-1,0)[i,j] - field_bounded[i,j]) <= 0. or corrfc[i,j]*(np.roll(field_bounded,1,0)[i,j] - np.roll(field_bounded,2,0)[i,j]) <= 0.):
                    corrfc[i,j] = 0. 
                if corrcf[i,j]*(field_bounded[i,j] - np.roll(field_bounded,1,1)[i,j]) <= 0. and (corrcf[i,j]*(np.roll(field_bounded,-1,1)[i,j] - field_bounded[i,j]) <= 0. or corrcf[i,j]*(np.roll(field_bounded,1,1)[i,j] - np.roll(field_bounded,2,1)[i,j]) <= 0.):
                    corrcf[i,j] = 0. 

    # Calculate allowable mass I/O for max rise and fall
    Qp = fields.dxcc*fields.dycc*(fieldmax - field_bounded) # at [i,j]
    Qm = fields.dxcc*fields.dycc*(field_bounded - fieldmin) # at [i,j]

    # Calculate I/O fluxes at cell centers
    Pp = config.dt*(np.maximum(0, corrfc) - np.minimum(0, np.roll(corrfc,-1,0)) + np.maximum(0, corrcf) - np.minimum(0, np.roll(corrcf,-1,1)))
    Pm = config.dt*(np.maximum(0, np.roll(corrfc,-1,0)) - np.minimum(0, corrfc) + np.maximum(0, np.roll(corrcf,-1,1)) - np.minimum(0, corrcf))

    # Calculate ratios of allowable (Q) to existing high-order (P) fluxes
    Rp = np.where(Pp > 1e-12, np.minimum(1., Qp/np.maximum(Pp,1e-12)), 0.)
    Rm = np.where(Pm > 1e-12, np.minimum(1., Qm/np.maximum(Pm,1e-12)), 0.)

    # Calculate the limiter for each face
    face_limiter_fc = np.where(corrfc >= 0., np.minimum(Rp, np.roll(Rm,1,0)), np.minimum(np.roll(Rp,1,0), Rm)) # at [i-1/2,j]
    face_limiter_cf = np.where(corrcf >= 0., np.minimum(Rp, np.roll(Rm,1,1)), np.minimum(np.roll(Rp,1,1), Rm)) # at [i,j-1/2]

    # Update the bounded flux and field
    flx_corr_fc = flxfc_bounded + face_limiter_fc*corrfc/fields.dycc # at [i-1/2,j]
    flx_corr_cf = flxcf_bounded + face_limiter_cf*corrcf/fields.dxcc # at [i,j-1/2]

    return flx_corr_fc, flx_corr_cf


def FCT(config, fields, it, flxfc_HO, flxcf_HO, **kwargs):
    # 2D FCT, should call the FCT1D function in x and y directions, and then call again combined to ensure monotonicity in both directions. Would need to consider how to set the extrema in 2D (probably based on the 8 surrounding cells and optionally previous time step).

    # Calculate 2D first-order solution if not using both global bounds
    sch.adimex_upwind(config, fields, it, **kwargs) # at [i,j] # temporarily writes in fields.tracer[it+1] but will be overwritten by the FCT solution at the end of this function

    flxfc_bounded = np.maximum(0.,fields.u[it])*(1.-fields.thetafc[it])*np.roll(fields.tracer[it],1,0) + np.minimum(0.,fields.u[it])*(1.-fields.thetafc[it])*fields.tracer[it] + np.maximum(0.,fields.u[it])*fields.thetafc[it]*np.roll(fields.tracer[it+1],1,0) + np.minimum(0.,fields.u[it])*fields.thetafc[it]*fields.tracer[it+1] # at [i-1/2,j]
    flxcf_bounded = np.maximum(0.,fields.v[it])*(1.-fields.thetacf[it])*np.roll(fields.tracer[it],1,1) + np.minimum(0.,fields.v[it])*(1.-fields.thetacf[it])*fields.tracer[it] + np.maximum(0.,fields.v[it])*fields.thetacf[it]*np.roll(fields.tracer[it+1],1,1) + np.minimum(0.,fields.v[it])*fields.thetacf[it]*fields.tracer[it+1] # at [i,j-1/2]
    
    # Set extrema for each cell
    minx, maxx = set_extrema_1D(config, fields, it, fields.tracer[it+1], axis=0) # at [i,j]
    miny, maxy = set_extrema_1D(config, fields, it, fields.tracer[it+1], axis=1) # at [i,j]

    # Apply 1D FCT in both x and y directions. Gives the limited fluxes in each direction
    flx_corr_x = FCT1D(config, fields, flxfc_HO, flxfc_bounded, minx, maxx, fields.tracer[it+1], axis=0, d_axis=fields.dycc) # at [i-1/2,j]
    flx_corr_y = FCT1D(config, fields, flxcf_HO, flxcf_bounded, miny, maxy, fields.tracer[it+1], axis=1, d_axis=fields.dxcc) # at [i,j-1/2]

    # Apply 2D FCT using the limited fluxes (necessary to avoid all ripples, see Zalesak 1979 p.350)
    min2D, max2D = set_extrema_2D(config, fields, it, fields.tracer[it+1]) # at [i,j]
    flx_corr_fc, flx_corr_cf = FCT2D(config, fields, flx_corr_x, flxfc_bounded, flx_corr_y, flxcf_bounded, min2D, max2D, fields.tracer[it+1]) # at [i-1/2,j] and at [i,j-1/2]

    return fields.tracer[it] + config.dt*(sch.fluxdiv(fields, flx_corr_fc, flx_corr_cf, 1., 1.)) # at [i,j]


def FCT_reduced(config, fields, it, flxfc_HO, flxcf_HO, **kwargs): # only does one pass
    """This function implements a reduced version of flux-corrected transport (FCT) that only does one pass (see Zalesak 1979 p.347-349). Returns the limited field at the new time step. Assumes constant dxc and uf>0.
    """

    # Calculate 2D first-order solution if not using both global bounds
    sch.adimex_upwind(config, fields, it, **kwargs) # at [i,j] # temporarily writes in fields.tracer[it+1] but will be overwritten by the FCT solution at the end of this function

    flxfc_bounded = np.maximum(0.,fields.u[it])*(1.-fields.thetafc[it])*np.roll(fields.tracer[it],1,0) + np.minimum(0.,fields.u[it])*(1.-fields.thetafc[it])*fields.tracer[it] + np.maximum(0.,fields.u[it])*fields.thetafc[it]*np.roll(fields.tracer[it+1],1,0) + np.minimum(0.,fields.u[it])*fields.thetafc[it]*fields.tracer[it+1] # at [i-1/2,j]
    flxcf_bounded = np.maximum(0.,fields.v[it])*(1.-fields.thetacf[it])*np.roll(fields.tracer[it],1,1) + np.minimum(0.,fields.v[it])*(1.-fields.thetacf[it])*fields.tracer[it] + np.maximum(0.,fields.v[it])*fields.thetacf[it]*np.roll(fields.tracer[it+1],1,1) + np.minimum(0.,fields.v[it])*fields.thetacf[it]*fields.tracer[it+1] # at [i,j-1/2]

    # Apply 2D FCT using the limited fluxes (necessary to avoid all ripples, see Zalesak 1979 p.350)
    min2D, max2D = set_extrema_2D(config, fields, it, fields.tracer[it+1]) # at [i,j]
    flx_corr_fc, flx_corr_cf = FCT2D(config, fields, flxfc_HO, flxfc_bounded, flxcf_HO, flxcf_bounded, min2D, max2D, fields.tracer[it+1]) # at [i-1/2,j] and at [i,j-1/2]
    
    return fields.tracer[it] + config.dt*(sch.fluxdiv(fields, flx_corr_fc, flx_corr_cf, 1., 1.)) # at [i,j]

