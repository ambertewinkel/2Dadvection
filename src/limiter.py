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

import matplotlib.pyplot as plt
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

    # Calculate allowable mass I/O for max rise and fall
    Qp = fields.dxcc*fields.dycc*(fieldmax - field_bounded) # at [i,j]
    Qm = fields.dxcc*fields.dycc*(field_bounded - fieldmin) # at [i,j]

    # Calculate I/O fluxes at cell centers
    Pp = config.dt*(np.maximum(0, corr) - np.minimum(0, np.roll(corr,-1,axis)))
    Pm = config.dt*(np.maximum(0, np.roll(corr,-1,axis)) - np.minimum(0, corr))

    # Calculate ratios of allowable (Q) to existing high-order (P) fluxes
    Rp = np.where(Pp > 1e-12, np.minimum(1., Qp/np.maximum(Pp,1e-12)), 0.)
    Rm = np.where(Pm > 1e-12, np.minimum(1., Qm/np.maximum(Pm,1e-12)), 0.)

    #plt.contourf(fields.xcc, fields.ycc, Qp)
    #plt.colorbar()
    #plt.title('Qp FCT1D')
    #plt.show()
    plt.contourf(fields.xcc, fields.ycc, Qm)
    plt.colorbar()
    plt.title('Qm FCT1D')
    plt.show()
    #plt.contourf(fields.xcc, fields.ycc, Pp)
    #plt.colorbar()
    #plt.title('Pp FCT1D')
    #plt.show()
    plt.contourf(fields.xcc, fields.ycc, Pm)
    plt.colorbar()
    plt.title('Pm FCT1D')
    plt.show()
    #plt.contourf(fields.xcc, fields.ycc, Rp, levels=100)
    #plt.colorbar()
    #plt.title('Rp FCT1D axis='+str(axis))
    #plt.show()
    plt.contourf(fields.xcc, fields.ycc, Rm, levels=100)
    plt.colorbar()
    plt.title('Rm FCT1D axis='+str(axis))
    plt.show()

    # Calculate the limiter for each face
    face_limiter = np.where(corr >= 0., np.minimum(Rp, np.roll(Rm,1,axis)), np.minimum(np.roll(Rp,1,axis), Rm)) # at [i-1/2,j] if axis=0, at [i,j-1/2] if axis=1

    plt.contourf(fields.xcc, fields.ycc, face_limiter)
    plt.colorbar()
    plt.title('face_limiter FCT1D axis='+str(axis))
    plt.show()
    
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

    ###plt.contourf(fields.xfc, fields.yfc, corrfc)
    ###plt.colorbar()
    ###plt.title('corrfc FCT2D')
    ###plt.show()
    ###plt.contourf(fields.xfc, fields.yfc, flxfc)
    ###plt.colorbar()
    ###plt.title('flxfc FCT2D')
    ###plt.show()
    ###plt.contourf(fields.xfc, fields.yfc, flxfc_bounded)
    ###plt.colorbar()
    ###plt.title('flxfc_bounded FCT2D')
    ###plt.show()
###
###
###
###
    ###exit()
    ###plt.contourf(fields.xcf, fields.ycf, corrcf)
    ###plt.colorbar()
    ###plt.title('corrcf FCT2D')
    ###plt.show()
###
    # Calculate allowable mass I/O for max rise and fall
    Qp = fields.dxcc*fields.dycc*(fieldmax - field_bounded) # at [i,j]
    Qm = fields.dxcc*fields.dycc*(field_bounded - fieldmin) # at [i,j]

    # Calculate I/O fluxes at cell centers
    Pp = config.dt*(np.maximum(0, corrfc) - np.minimum(0, np.roll(corrfc,-1,0)) + np.maximum(0, corrcf) - np.minimum(0, np.roll(corrcf,-1,1)))
    Pm = config.dt*(np.maximum(0, np.roll(corrfc,-1,0)) - np.minimum(0, corrfc) + np.maximum(0, np.roll(corrcf,-1,1)) - np.minimum(0, corrcf))

    # Calculate ratios of allowable (Q) to existing high-order (P) fluxes
    Rp = np.where(Pp > 1e-12, np.minimum(1., Qp/np.maximum(Pp,1e-12)), 0.)
    Rm = np.where(Pm > 1e-12, np.minimum(1., Qm/np.maximum(Pm,1e-12)), 0.)


    ###plt.contourf(fields.xcc, fields.ycc, Qp)
    ###plt.colorbar()
    ###plt.title('Qp FCT2D')
    ###plt.show()
    ###plt.contourf(fields.xcc, fields.ycc, Qm)
    ###plt.colorbar()
    ###plt.title('Qm FCT2D')
    ###plt.show()
    ###plt.contourf(fields.xcc, fields.ycc, Pp)
    ###plt.colorbar()
    ###plt.title('Pp FCT2D')
    ###plt.show()
    ###plt.contourf(fields.xcc, fields.ycc, Pm)
    ###plt.colorbar()
    ###plt.title('Pm FCT2D')
    ###plt.show()
    ###plt.contourf(fields.xcc, fields.ycc, Rp, levels=100)
    ###plt.colorbar()
    ###plt.title('Rp FCT2D')
    ###plt.show()
    ###plt.contourf(fields.xcc, fields.ycc, Rm, levels=100)
    ###plt.colorbar()
    ###plt.title('Rm FCT2D')
    ###plt.show()


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

    #print('min bounded field after adimex upwind=', np.min(fields.tracer[it+1]))
    #print('max bounded field after adimex upwind=', np.max(fields.tracer[it+1]))
    
    # Set extrema for each cell
    minx, maxx = set_extrema_1D(config, fields, it, fields.tracer[it+1], axis=0) # at [i,j]
    miny, maxy = set_extrema_1D(config, fields, it, fields.tracer[it+1], axis=1) # at [i,j]

    # Check whether I am taking the min max correctly:

    #plt.contourf(fields.xcc, fields.ycc, fields.tracer[0])
    #plt.colorbar()
    #plt.title('initial field')
    #plt.show()
#
    #plt.contourf(fields.xcc, fields.ycc, maxx - fields.tracer[0])
    #plt.colorbar()
    #plt.title('maxx - initial field')
    #plt.show()
#
    #plt.contourf(fields.xcc, fields.ycc, maxy - fields.tracer[0])
    #plt.colorbar()
    #plt.title('maxy - initial field')
    #plt.show()

    #plt.contourf(fields.xcc, fields.ycc, fields.tracer[it+1] - fields.tracer[0])
    #plt.colorbar()
    #plt.title('difference between final and initial field')
    #plt.show()
#
    #exit()



    # TEST - this could be it. - don't think it makes much, if any impact compared to FCTred, and it is not what is described in the Zalesak paper
    #minx, maxx = set_extrema_2D(config, fields, it, fields.tracer[it+1]) # at [i,j]
    #miny, maxy = set_extrema_2D(config, fields, it, fields.tracer[it+1]) # at [i,j]
#

    # Apply 1D FCT in both x and y directions. Gives the limited fluxes in each direction
    flx_corr_x = FCT1D(config, fields, flxfc_HO, flxfc_bounded, minx, maxx, fields.tracer[it+1], axis=0, d_axis=fields.dycc) # at [i-1/2,j]
    flx_corr_y = FCT1D(config, fields, flxcf_HO, flxcf_bounded, miny, maxy, fields.tracer[it+1], axis=1, d_axis=fields.dxcc) # at [i,j-1/2]

    tempfield = fields.tracer[it] + config.dt*(sch.fluxdiv(fields, flx_corr_x, flx_corr_y, 1., 1.)) # at [i,j]

    #listwrong = []
    #for i in range(config.nx):
    #    for j in range(config.ny):
    #        if tempfield[i,j] < minx[i,j] - 1e-12 or tempfield[i,j] < miny[i,j] - 1e-12:
    #            print('FCT failed: temp field is outside bounds at point (',i,',',j,')')
    #            print('temp field value=', tempfield[i,j], 'minx value=', minx[i,j], 'miny value=', miny[i,j])   
    #            listwrong.append((i,j))             
    #        if tempfield[i,j] > maxx[i,j] + 1e-12 or tempfield[i,j] > maxy[i,j] + 1e-12:
    #            print('FCT failed: temp field is outside bounds at point (',i,',',j,')')
    #            print('temp field value=', tempfield[i,j], 'maxx value=', maxx[i,j], 'maxy value=', maxy[i,j])
    #            listwrong.append((i,j))
#
    #plt.plot([p[0] for p in listwrong], [p[1] for p in listwrong], 'ro')
    #plt.title('temp field after FCT1D is outside bounds')
    #plt.show()

    ####!plt.contourf(fields.xfc, fields.yfc, abs(flx_corr_x)-abs(flxfc_HO))
    ####!plt.colorbar()
    ####!plt.title('flx_corr_x - flxfc_HO after FCT1D abs')
    ####!plt.show()
    ####!plt.contourf(fields.xcf, fields.ycf, flx_corr_y-flxcf_HO)
    ####!plt.colorbar()
    ####!plt.title('flx_corr_y - flxcf_HO after FCT1D')
    ####!plt.show()
####!
####!
    ####!plt.contourf(fields.xcc, fields.ycc, tempfield-fields.tracer[it])
    ####!plt.colorbar()
    ####!plt.title('tempfield - bounded after FCT1D before FCT2D')
    ####!plt.show()

    # Apply 2D FCT using the limited fluxes (necessary to avoid all ripples, see Zalesak 1979 p.350)
    min2D, max2D = set_extrema_2D(config, fields, it, fields.tracer[it+1]) # at [i,j]
    flx_corr_fc, flx_corr_cf = FCT2D(config, fields, flx_corr_x, flxfc_bounded, flx_corr_y, flxcf_bounded, min2D, max2D, fields.tracer[it+1]) # at [i-1/2,j] and at [i,j-1/2]


    #if it%10 == 0:
    #    plt.contourf(fields.xfc, fields.yfc, flx_corr_fc-flxfc_bounded)
    #    plt.colorbar()
    #    plt.title('flx_corr_fc - flxfc_bounded after FCT2D')
    #    plt.show()
    #plt.contourf(fields.xfc, fields.yfc, flx_corr_fc-flx_corr_x)
    #plt.colorbar()
    #plt.title('flx_corr_fc - flx_corr_x after FCT2D')
    #plt.show()

    #if it%10 == 0:
    #    plt.contourf(fields.xcf, fields.ycf, flx_corr_cf-flxcf_bounded)
    #    plt.colorbar()
    #    plt.title('flx_corr_cf - flxcf_bounded after FCT2D')
    #    plt.show()
    finalfield = fields.tracer[it] + config.dt*(sch.fluxdiv(fields, flx_corr_fc, flx_corr_cf, 1., 1.)) # at [i,j]

    #plt.contourf(fields.xcc, fields.ycc, finalfield-fields.tracer[it])
    #plt.colorbar()
    #plt.title('final field - bounded after both FCT1D and FCT2D')
    #plt.show()`
    ## 
    #if finalfield.min() < minx.min() - 1e-12 or finalfield.min() < miny.min() - 1e-12 or finalfield.max() > maxx.max() + 1e-12 or finalfield.max() > maxy.max() + 1e-12:
#        print('FCT failed: final field is outside bounds')
#        print('final field min=', finalfield.min(), 'minx min=', minx.min(), 'miny min=', miny.min())
#        print('final field max=', finalfield.max(), 'maxx max=', maxx.max(), 'maxy max=', maxy.max())
    
    ###listwrong = []
    ###for i in range(config.nx):
    ###    for j in range(config.ny):
    ###        if finalfield[i,j] < minx[i,j] - 1e-12 or finalfield[i,j] < miny[i,j] - 1e-12:
    ###            print('FCT failed: final field is outside bounds at point (',i,',',j,')')
    ###            print('final field value=', finalfield[i,j], 'minx value=', minx[i,j], 'miny value=', miny[i,j])   
    ###            listwrong.append((i,j))             
    ###        if finalfield[i,j] > maxx[i,j] + 1e-12 or finalfield[i,j] > maxy[i,j] + 1e-12:
    ###            print('FCT failed: final field is outside bounds at point (',i,',',j,')')
    ###            print('final field value=', finalfield[i,j], 'maxx value=', maxx[i,j], 'maxy value=', maxy[i,j])
    ###            listwrong.append((i,j))

    #plt.plot([p[0] for p in listwrong], [p[1] for p in listwrong], 'ro')
    #plt.show()

    #listwrong2D = []
    #for i in range(config.nx):
    #    for j in range(config.ny):
    #        if finalfield[i,j] < min2D[i,j] - 1e-12:
    #            print('FCT failed: final field is outside bounds at point (',i,',',j,')')
    #            print('final field value=', finalfield[i,j], 'min2D value=', min2D[i,j])   
    #            listwrong2D.append((i,j))             
    #        if finalfield[i,j] > max2D[i,j] + 1e-12:
    #            print('FCT failed: final field is outside bounds at point (',i,',',j,')')
    #            print('final field value=', finalfield[i,j], 'max2D value=', max2D[i,j])
    #            listwrong2D.append((i,j))
#
    ##plt.plot([p[0] for p in listwrong2D], [p[1] for p in listwrong2D], 'ro')
    ##plt.show()
    ##
    #exit()

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

