import numpy as np
from functools import partial
import src.solvers as sv
import src.limiter as lim

################# UPWIND SCHEME #################

def fluxdiv_first(config, fields, it, phi, factor_fc, factor_cf):
    """Calculating the first-order flux divergence for a certain phi, can be applied for bot the explicit and implicit parts (using different factors)"""

    phi_BS_fc = np.roll(phi,1,0) # backward in space (upwind if u>0) flux at face in x direction
    phi_FS_fc = phi # forward in space (upwind if u<0) flux at face in x direction
    phi_BS_cf = np.roll(phi,1,1) # backward in space (upwind if v>0) flux at face in y direction
    phi_FS_cf = phi # forward in space (upwind if v<0) flux at face in y direction
    flxx = factor_fc*(np.maximum(0., fields.u[config.store_all_timesteps*it]) * phi_BS_fc + np.minimum(0., fields.u[config.store_all_timesteps*it]) * phi_FS_fc) # at [i-1/2,j]
    flxy = factor_cf*(np.maximum(0., fields.v[config.store_all_timesteps*it]) * phi_BS_cf + np.minimum(0., fields.v[config.store_all_timesteps*it]) * phi_FS_cf) # at [i,j-1/2]

    return (flxx - np.roll(flxx,-1,0))/fields.dxcc + (flxy - np.roll(flxy,-1,1))/fields.dycc # at [i,j]


def upwind(config, fields, it, **kwargs):
    """Implement the upwind scheme for the given time step"""

    it_prev = config.store_all_timesteps*it + 1 - config.store_all_timesteps
    field_prev = fields.tracer[it_prev].copy()

    fields.tracer[config.store_all_timesteps*it+1] = field_prev + config.dt*fluxdiv_first(config, fields, it, field_prev, 1., 1.) # at [i,j]


################# ADIMEX UPWIND SCHEME #################

def implicitness_adimex_upwind(config, fields, it, **kwargs):
    """Calculate Courant numbers at cell centers and implicitness at cell centers and faces for AdImEx upwind scheme"""
    # Assumes nondivergent winds

    # Calculate Courant numbers at cell faces
    #Cfc = config.dt/fields.dxcc * fields.u[config.store_all_timesteps*it] # at [i-1/2,j] # not sure if this is quite valid to do when using a nonuniform grid
    #Ccf = config.dt/fields.dycc * fields.v[config.store_all_timesteps*it] # at [i,j-1/2]
    
    # Calculate Courant numbers at cell centers
    #C_in_cc = np.maximum(0.,Cfc) - np.minimum(0.,np.roll(Cfc,-1,0)) + np.maximum(0.,Ccf) - np.minimum(0.,np.roll(Ccf,-1,1)) # at [i,j]
    #C_out_cc = - np.minimum(0.,Cfc) + np.maximum(0.,np.roll(Cfc,-1,0)) - np.minimum(0.,Ccf) + np.maximum(0.,np.roll(Ccf,-1,1)) # at [i,j]

    # assumes dy is constant in the x direction (should be defined at each face to multiply with the velocity to get the flux, but this is the same value as dycc at the cell center, so using that for simplicity) -- the same thing applies for dx in the y direction.
    #Cincc =  0.5*config.dt/(fields.dxcc*fields.dycc)*((np.maximum(0.,fields.u[config.store_all_timesteps*it]) - np.minimum(0.,np.roll(fields.u[config.store_all_timesteps*it],-1,0)))*fields.dycc + (np.maximum(0.,fields.v[config.store_all_timesteps*it]) - np.minimum(0.,np.roll(fields.v[config.store_all_timesteps*it],-1,1)))*fields.dxcc) # at [i,j]
    #Coutcc = 0.5*config.dt/(fields.dxcc*fields.dycc)*((-np.minimum(0.,fields.u[config.store_all_timesteps*it]) + np.maximum(0.,np.roll(fields.u[config.store_all_timesteps*it],-1,0)))*fields.dycc + (-np.minimum(0.,fields.v[config.store_all_timesteps*it]) + np.maximum(0.,np.roll(fields.v[config.store_all_timesteps*it],-1,1)))*fields.dxcc) # at [i,j]
    #fields.Ccc[config.store_all_timesteps*it] = 0.5*(Cincc + Coutcc) # at [i,j] (always nonnegative) 
    
    # assumes dy is constant in the x direction (should be defined at each face to multiply with the velocity to get the flux, but this is the same value as dycc at the cell center, so using that for simplicity) -- the same thing applies for dx in the y direction.
    sum_abs_velarea = (abs(fields.u[config.store_all_timesteps*it]) + abs(np.roll(fields.u[config.store_all_timesteps*it],-1,0)))*fields.dycc + (abs(fields.v[config.store_all_timesteps*it]) + abs(np.roll(fields.v[config.store_all_timesteps*it],-1,1)))*fields.dxcc # at [i,j]
    fields.Ccc[config.store_all_timesteps*it] = 0.5*config.dt*sum_abs_velarea/(fields.dxcc*fields.dycc) # at [i,j] (always nonnegative) # see Weller et al 2023 for definition

    if config.theta_anisotropic: # anisotropic implicitness (independent in x and y directions)
        Cfc = config.dt*(0.5*(fields.u[config.store_all_timesteps*it] + abs(fields.u[config.store_all_timesteps*it]))*fields.dycc/(np.roll(fields.dxcc,1,0)*fields.dycc) + 0.5*(-fields.u[config.store_all_timesteps*it] + abs(fields.u[config.store_all_timesteps*it]))*fields.dycc/(fields.dxcc*fields.dycc)) # assumes an orthogonal grid (dyfc = dycc etc). #  always positive # at [i-1/2,j] 
        Ccf = config.dt*(0.5*(fields.v[config.store_all_timesteps*it] + abs(fields.v[config.store_all_timesteps*it]))*fields.dxcc/(fields.dxcc*np.roll(fields.dycc,1,1)) + 0.5*(-fields.v[config.store_all_timesteps*it] + abs(fields.v[config.store_all_timesteps*it]))*fields.dxcc/(fields.dxcc*fields.dycc)) # assumes an orthogonal grid (dyfc = dycc etc). #  always positive # at [i,j-1/2]
        plot=False
        if plot:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
            contour = axes[0].contourf(fields.xcc, fields.ycc, fields.Ccc[config.store_all_timesteps*it])
            axes[0].set_title('Ccc')
            fig.colorbar(contour, ax=axes[0])
            contour = axes[1].contourf(fields.xfc, fields.yfc, Cfc)
            axes[1].set_title('Cfc')
            fig.colorbar(contour, ax=axes[1])
            contour = axes[2].contourf(fields.xcf, fields.ycf, Ccf)
            axes[2].set_title('Ccf')
            fig.colorbar(contour, ax=axes[2])
            plt.show()
        fields.thetafc[config.store_all_timesteps*it] = np.maximum(0., 1. - config.factordiv/(2.*Cfc + 1e-12)) # at [i-1/2,j] 
        fields.thetacf[config.store_all_timesteps*it] = np.maximum(0., 1. - config.factordiv/(2.*Ccf + 1e-12)) # at [i,j-1/2]
        fields.thetacc[config.store_all_timesteps*it] = np.maximum(0., 1. - config.factordiv/(fields.Ccc[config.store_all_timesteps*it] + 1e-12)) # at [i,j] # Needed to activate the matrix solve


        # For comparison: previous settings:
        if plot:
            thetacc2 = np.maximum(0., 1. - config.factordiv/(fields.Ccc[config.store_all_timesteps*it] + 1e-12))
            thetafc2 = np.maximum(thetacc2, np.roll(thetacc2,1,0)) # at [i-1/2,j]
            thetacf2 = np.maximum(thetacc2, np.roll(thetacc2,1,1)) # at [i,j-1/2]

            fig, axes = plt.subplots(1, 6, figsize=(25, 5), constrained_layout=True)
            contour = axes[0].contourf(fields.xfc, fields.yfc, fields.thetafc[config.store_all_timesteps*it])
            axes[0].set_title('thetafc1')
            fig.colorbar(contour, ax=axes[0])
            contour = axes[1].contourf(fields.xfc, fields.yfc, thetafc2)
            axes[1].set_title('thetafc2')
            fig.colorbar(contour, ax=axes[1])
            contour = axes[2].contourf(fields.xfc, fields.yfc, fields.thetafc[config.store_all_timesteps*it] - thetafc2)
            axes[2].set_title('thetafc1 - thetafc2')
            fig.colorbar(contour, ax=axes[2])
            contour = axes[3].contourf(fields.xcf, fields.ycf, fields.thetacf[config.store_all_timesteps*it])
            axes[3].set_title('thetacf1')
            fig.colorbar(contour, ax=axes[3])
            contour = axes[4].contourf(fields.xcf, fields.ycf, thetacf2)
            axes[4].set_title('thetacf2')
            fig.colorbar(contour, ax=axes[4])        
            contour = axes[5].contourf(fields.xcf, fields.ycf, fields.thetacf[config.store_all_timesteps*it] - thetacf2)
            axes[5].set_title('thetacf1 - thetacf2')
            fig.colorbar(contour, ax=axes[5])
            plt.show()

    else:
        # Calculate implicitness at cell centers and faces
        fields.thetacc[config.store_all_timesteps*it] = np.maximum(0., 1. - config.factordiv/(fields.Ccc[config.store_all_timesteps*it] + 1e-12)) # at [i,j] # for nondivergent winds: factordiv = 1.; for divergent winds: factordiv = 0.5; preserves positivity in all cases for c_in and c_out
        fields.thetafc[config.store_all_timesteps*it] = np.maximum(fields.thetacc[config.store_all_timesteps*it], np.roll(fields.thetacc[config.store_all_timesteps*it],1,0)) # at [i-1/2,j]
        fields.thetacf[config.store_all_timesteps*it] = np.maximum(fields.thetacc[config.store_all_timesteps*it], np.roll(fields.thetacc[config.store_all_timesteps*it],1,1)) # at [i,j-1/2]

def adimex_upwind_matrix_func(phi, config, fields, it, thetafc, thetacf):
    """Matrix function for the implicit part of the AdImEx upwind scheme"""

    return phi - config.dt*fluxdiv_first(config, fields, it, phi, thetafc, thetacf) # at [i,j]


def adimex_upwind(config, fields, it, tolerance=1e-6, kiter=10, jiter=4, **kwargs):
    """Implement the AdImEx upwind scheme for the given time step"""

    # Calculate the implicitness (1-1/(2C)) at each cell face
    implicitness_adimex_upwind(config, fields, it, **kwargs)

    it_prev = config.store_all_timesteps*it + 1 - config.store_all_timesteps
    field_prev = fields.tracer[it_prev].copy()

    # Calculate RHS (explicit) at cell faces
    rhs = field_prev + config.dt*fluxdiv_first(config, fields, it, field_prev, 1.-fields.thetafc[config.store_all_timesteps*it], 1.-fields.thetacf[config.store_all_timesteps*it]) # at [i,j] 
    
    # Calculate LHS (implicit) upwind fluxes at cell faces    
    solver = config.solver # numpy, gcrk_matrix, gcrk_matrixfree # not sure if numpy is possible with a 4D matrix. # 17-11-2025: only gcrk_matrixfree implemented

    if np.any(fields.thetacc[config.store_all_timesteps*it]): # avoids gmresm breaking down, only running the solver when there is a nonunit matrix
        matrix = partial(adimex_upwind_matrix_func, config=config, fields=fields, it=it, thetafc=fields.thetafc[config.store_all_timesteps*it], thetacf=fields.thetacf[config.store_all_timesteps*it]) # at [i,j]
        solver = getattr(sv, config.solver)
        fields.tracer[config.store_all_timesteps*it+1] = solver(matrix, rhs, field_prev, kiter=kiter, jiter=jiter, tolerance=tolerance) # kiter, jiter, and tolerance can be set differently, e.g. for limiting
    else:
        fields.tracer[config.store_all_timesteps*it+1] = rhs.copy()


################# ADHIMEX SCHEME #################

import matplotlib.pyplot as plt
def implicitness_adhimex(config, fields, it, **kwargs):
    """Calculate Courant numbers at cell centers and implicitness at cell centers and faces for AdHImEx scheme. Assumes nondivergent winds, and that dy is constant in x and dx is constant in y"""

    # assumes dy is constant in the x direction (should be defined at each face to multiply with the velocity to get the flux, but this is the same value as dycc at the cell center, so using that for simplicity) -- the same thing applies for dx in the y direction.
    # sum_abs_velarea like this is still fine in the nonperiodic BCs as the u and v at the boundaries are assumed zero (e.g. for Hadley circulation)
    sum_abs_velarea = (abs(fields.u[config.store_all_timesteps*it]) + abs(np.roll(fields.u[config.store_all_timesteps*it],-1,0)))*fields.dycc + (abs(fields.v[config.store_all_timesteps*it]) + abs(np.roll(fields.v[config.store_all_timesteps*it],-1,1)))*fields.dxcc # at [i,j]
    fields.Ccc[config.store_all_timesteps*it] = 0.5*config.dt*sum_abs_velarea/(fields.dxcc*fields.dycc) # at [i,j] (always nonnegative) # see Weller et al 2023 for definition
    
    if config.theta_anisotropic: # anisotropic implicitness (different in x and y directions)
        Cfc = config.dt*(0.5*(fields.u[config.store_all_timesteps*it] + abs(fields.u[config.store_all_timesteps*it]))*fields.dycc/(np.roll(fields.dxcc,1,0)*fields.dycc) + 0.5*(-fields.u[config.store_all_timesteps*it] + abs(fields.u[config.store_all_timesteps*it]))*fields.dycc/(fields.dxcc*fields.dycc)) # assumes an orthogonal grid (dyfc = dycc etc). #  always positive # at [i-1/2,j] 
        Ccf = config.dt*(0.5*(fields.v[config.store_all_timesteps*it] + abs(fields.v[config.store_all_timesteps*it]))*fields.dxcc/(fields.dxcc*np.roll(fields.dycc,1,1)) + 0.5*(-fields.v[config.store_all_timesteps*it] + abs(fields.v[config.store_all_timesteps*it]))*fields.dxcc/(fields.dxcc*fields.dycc)) # assumes an orthogonal grid (dyfc = dycc etc). #  always positive # at [i,j-1/2]
        plot=False
        if plot:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
            contour = axes[0].contourf(fields.xcc, fields.ycc, fields.Ccc[config.store_all_timesteps*it])
            axes[0].set_title('Ccc')
            fig.colorbar(contour, ax=axes[0])
            contour = axes[1].contourf(fields.xfc, fields.yfc, Cfc)
            axes[1].set_title('Cfc')
            fig.colorbar(contour, ax=axes[1])
            contour = axes[2].contourf(fields.xcf, fields.ycf, Ccf)
            axes[2].set_title('Ccf')
            fig.colorbar(contour, ax=axes[2])
            plt.show()
        fields.thetafc[config.store_all_timesteps*it] = 1. - 1./(1. + 0.7*np.maximum(0., 2.*Cfc - 1.4)) # at [i-1/2,j] 
        fields.thetacf[config.store_all_timesteps*it] = 1. - 1./(1. + 0.7*np.maximum(0., 2.*Ccf - 1.4)) # at [i,j-1/2]
        #fields.thetacc[config.store_all_timesteps*it] = np.maximum.reduce([fields.thetafc[config.store_all_timesteps*it], np.roll(fields.thetafc[config.store_all_timesteps*it],-1,0), fields.thetacf[config.store_all_timesteps*it], np.roll(fields.thetacf[config.store_all_timesteps*it],-1,1)]) # at [i,j] # Needed to activate the matrix solve + for the fEx_c and fIm_c parts needed for constancy. This works, it seems to be stable for uniform uv and hadley and swift nondiv cases. 
        fields.thetacc[config.store_all_timesteps*it] = 1. - 1./(1. + 0.7*np.maximum(0., fields.Ccc[config.store_all_timesteps*it] - 1.4)) # at [i,j] # Needed to activate the matrix solve + for the fEx_c and fIm_c parts needed for constancy. This works, it seems to be stable for uniform uv and hadley and swift nondiv cases. 

        # For comparison: previous settings:
        if plot:
            thetacc2 = 1. - 1./(1. + 0.7*np.maximum(0., fields.Ccc[config.store_all_timesteps*it] - 1.4)) # at [i,j]
            thetafc2 = np.maximum(thetacc2, np.roll(thetacc2,1,0)) # at [i-1/2,j]
            thetacf2 = np.maximum(thetacc2, np.roll(thetacc2,1,1)) # at [i,j-1/2]

            fig, axes = plt.subplots(1, 6, figsize=(25, 5), constrained_layout=True)
            contour = axes[0].contourf(fields.xfc, fields.yfc, fields.thetafc[config.store_all_timesteps*it])
            axes[0].set_title('thetafc1')
            fig.colorbar(contour, ax=axes[0])
            contour = axes[1].contourf(fields.xfc, fields.yfc, thetafc2)
            axes[1].set_title('thetafc2')
            fig.colorbar(contour, ax=axes[1])
            contour = axes[2].contourf(fields.xfc, fields.yfc, fields.thetafc[config.store_all_timesteps*it] - thetafc2)
            axes[2].set_title('thetafc1 - thetafc2')
            fig.colorbar(contour, ax=axes[2])
            contour = axes[3].contourf(fields.xcf, fields.ycf, fields.thetacf[config.store_all_timesteps*it])
            axes[3].set_title('thetacf1')
            fig.colorbar(contour, ax=axes[3])
            contour = axes[4].contourf(fields.xcf, fields.ycf, thetacf2)
            axes[4].set_title('thetacf2')
            fig.colorbar(contour, ax=axes[4])        
            contour = axes[5].contourf(fields.xcf, fields.ycf, fields.thetacf[config.store_all_timesteps*it] - thetacf2)
            axes[5].set_title('thetacf1 - thetacf2')
            fig.colorbar(contour, ax=axes[5])
            plt.show()
    else:
        fields.thetacc[config.store_all_timesteps*it] = 1. - 1./(1. + 0.7*np.maximum(0., fields.Ccc[config.store_all_timesteps*it] - 1.4)) # at [i,j]
        fields.thetafc[config.store_all_timesteps*it] = np.maximum(fields.thetacc[config.store_all_timesteps*it], np.roll(fields.thetacc[config.store_all_timesteps*it],1,0)) # at [i-1/2,j]
        fields.thetacf[config.store_all_timesteps*it] = np.maximum(fields.thetacc[config.store_all_timesteps*it], np.roll(fields.thetacc[config.store_all_timesteps*it],1,1)) # at [i,j-1/2]
    
    if config.BC_x == 'periodic':
        fields.dthetafc[config.store_all_timesteps*it] = np.roll(fields.thetafc[config.store_all_timesteps*it],-1,0) - fields.thetafc[config.store_all_timesteps*it] # at [i,j]
    else: # doesn't work properly for nonperiodic BC
        fields.dthetafc[config.store_all_timesteps*it] = np.full(np.shape(fields.dthetafc[config.store_all_timesteps*it]), -1.)
    if config.BC_y == 'periodic':
        fields.dthetacf[config.store_all_timesteps*it] = np.roll(fields.thetacf[config.store_all_timesteps*it],-1,1) - fields.thetacf[config.store_all_timesteps*it] # at [i,j]
    else: # doesn't work properly for nonperiodic BC
        fields.dthetacf[config.store_all_timesteps*it] = np.full(np.shape(fields.dthetacf[config.store_all_timesteps*it]), -1.)


def adhimex_butcher():
    """Sets up AdHImEx Butcher tableau"""

    # Butcher tableau for explicit part (left tableau from Ullrich and Jablonowski 2012. See the Weller Lock and Wood (2013) UJ3(1+e,3,2) scheme)   
    AEx = np.array([[0., 0., 0., 0., 0.],[0., 0., 0., 0., 0.],[0., 1., 0., 0., 0.],[0., 0.25, 0.25, 0., 0.],[0., 1/6, 1/6, 2/3, 0.]])

    # Butcher tableau for implicit part (right tableau from Ullrich and Jablonowski 2012. See the Weller Lock and Wood (2013) UJ3(1+e,3,2) scheme)   
    AIm = np.array([[0., 0., 0., 0., 0.],[0.5, 0., 0., 0., 0.],[0.5, 0., 0., 0., 0.],[0.5, 0., 0., 0., 0.],[0.5, 0., 0., 0., 0.5]])

    nstages = np.shape(AIm)[1]

    return AEx, AIm, nstages


def fifth_order(config, fields, it, phi):
    """Fifth-order spatial discretisation"""
    if config.BC_x == 'periodic' and config.BC_y == 'periodic':
        phi_BS_fc = -1./20*np.roll(phi,-1,0) + 9./20.*phi + 47./60.*np.roll(phi,1,0) - 13./60.*np.roll(phi,2,0) + 1./30.*np.roll(phi,3,0) # backward in space (upwind if u>0) flux at face in x direction
        phi_FS_fc = -1./20*np.roll(phi,2,0) + 9./20.*np.roll(phi,1,0) + 47./60.*phi - 13./60.*np.roll(phi,-1,0) + 1./30.*np.roll(phi,-2,0) # forward in space (upwind if u<0) flux at face in x direction
        phi_BS_cf = -1./20*np.roll(phi,-1,1) + 9./20.*phi + 47./60.*np.roll(phi,1,1) - 13./60.*np.roll(phi,2,1) + 1./30.*np.roll(phi,3,1) # backward in space (upwind if v>0) flux at face in y direction
        phi_FS_cf = -1./20*np.roll(phi,2,1) + 9./20.*np.roll(phi,1,1) + 47./60.*phi - 13./60.*np.roll(phi,-1,1) + 1./30.*np.roll(phi,-2,1) # forward in space (upwind if v<0) flux at face in y direction
    elif config.BC_x == 'noflux' and config.BC_y == 'noflux':
        phi = phi
        phi_temp = np.append(phi, np.array([phi[-1,:],phi[-1,:],phi[0,:],phi[0,:]]), axis=0)
        phi_temp = np.append(phi_temp, np.array([phi_temp[:,-1],phi_temp[:,-1],phi_temp[:,0],phi_temp[:,0]]).T, axis=1)
        phi_temp_BS_fc = -1./20*np.roll(phi_temp,-1,0) + 9./20.*phi_temp + 47./60.*np.roll(phi_temp,1,0) - 13./60.*np.roll(phi_temp,2,0) + 1./30.*np.roll(phi_temp,3,0) # backward in space (upwind if u>0) flux at face in x direction
        phi_temp_FS_fc = -1./20*np.roll(phi_temp,2,0) + 9./20.*np.roll(phi_temp,1,0) + 47./60.*phi_temp - 13./60.*np.roll(phi_temp,-1,0) + 1./30.*np.roll(phi_temp,-2,0) # forward in space (upwind if u<0) flux at face in x direction
        phi_temp_BS_cf = -1./20*np.roll(phi_temp,-1,1) + 9./20.*phi_temp + 47./60.*np.roll(phi_temp,1,1) - 13./60.*np.roll(phi_temp,2,1) + 1./30.*np.roll(phi_temp,3,1) # backward in space (upwind if v>0) flux at face in y direction
        phi_temp_FS_cf = -1./20*np.roll(phi_temp,2,1) + 9./20.*np.roll(phi_temp,1,1) + 47./60.*phi_temp - 13./60.*np.roll(phi_temp,-1,1) + 1./30.*np.roll(phi_temp,-2,1) # forward in space (upwind if v<0) flux at face in y direction
        phi_BS_fc = phi_temp_BS_fc[:-4,:-4]
        phi_FS_fc = phi_temp_FS_fc[:-4,:-4]
        phi_BS_cf = phi_temp_BS_cf[:-4,:-4]
        phi_FS_cf = phi_temp_FS_cf[:-4,:-4]
    else:
        raise ValueError('Fifth-order SD is set to use either both BCs periodic or both nonperiodic (noflux).')

    flxfc = np.maximum(0., fields.u[config.store_all_timesteps*it]) * phi_BS_fc + np.minimum(0., fields.u[config.store_all_timesteps*it]) * phi_FS_fc # at [i-1/2,j]
    flxcf = np.maximum(0., fields.v[config.store_all_timesteps*it]) * phi_BS_cf + np.minimum(0., fields.v[config.store_all_timesteps*it]) * phi_FS_cf # at [i,j-1/2]

    return flxfc, flxcf


def third_order(config, fields, it, phi):
    """Third-order spatial discretisation (potentially used for the matrix)"""
    phi_BS_fc = 1./3.*phi + 5./6.*np.roll(phi,1,0) - 1./6.*np.roll(phi,2,0) # backward in space (upwind if u>0) flux at face in x direction
    phi_FS_fc = 1./3.*np.roll(phi,1,0) + 5./6.*phi - 1./6.*np.roll(phi,-1,0) # forward in space (upwind if u<0) flux at face in x direction
    phi_BS_cf = 1./3.*phi + 5./6.*np.roll(phi,1,1) - 1./6.*np.roll(phi,2,1) # backward in space (upwind if v>0) flux at face in y direction
    phi_FS_cf = 1./3.*np.roll(phi,1,1) + 5./6.*phi - 1./6.*np.roll(phi,-1,1) # forward in space (upwind if v<0) flux at face in y direction

    flxfc = np.maximum(0., fields.u[config.store_all_timesteps*it]) * phi_BS_fc + np.minimum(0., fields.u[config.store_all_timesteps*it]) * phi_FS_fc # at [i-1/2,j]

    flxcf = np.maximum(0., fields.v[config.store_all_timesteps*it]) * phi_BS_cf + np.minimum(0., fields.v[config.store_all_timesteps*it]) * phi_FS_cf # at [i,j-1/2]

    return flxfc, flxcf


def fluxdiv_fifth(config, fields, it,  phi, factor_fc, factor_cf): # old code for adhimex_12 etc. Should do the same as fluxdiv() with fifth_order() # assumes periodic BC
    """Calculating the fifth-order flux divergence for a certain phi, can be applied for bot the explicit and implicit parts (using different factors)"""

    phi_BS_fc = -1./20*np.roll(phi,-1,0) + 9./20.*phi + 47./60.*np.roll(phi,1,0) - 13./60.*np.roll(phi,2,0) + 1./30.*np.roll(phi,3,0) # backward in space (upwind if u>0) flux at face in x direction
    phi_FS_fc = -1./20*np.roll(phi,2,0) + 9./20.*np.roll(phi,1,0) + 47./60.*phi - 13./60.*np.roll(phi,-1,0) + 1./30.*np.roll(phi,-2,0) # forward in space (upwind if u<0) flux at face in x direction
    phi_BS_cf = -1./20*np.roll(phi,-1,1) + 9./20.*phi + 47./60.*np.roll(phi,1,1) - 13./60.*np.roll(phi,2,1) + 1./30.*np.roll(phi,3,1) # backward in space (upwind if v>0) flux at face in y direction
    phi_FS_cf = -1./20*np.roll(phi,2,1) + 9./20.*np.roll(phi,1,1) + 47./60.*phi - 13./60.*np.roll(phi,-1,1) + 1./30.*np.roll(phi,-2,1) # forward in space (upwind if v<0) flux at face in y direction
    flxx = factor_fc*(np.maximum(0., fields.u[config.store_all_timesteps*it]) * phi_BS_fc + np.minimum(0., fields.u[config.store_all_timesteps*it]) * phi_FS_fc) # at [i-1/2,j]
    flxy = factor_cf*(np.maximum(0., fields.v[config.store_all_timesteps*it]) * phi_BS_cf + np.minimum(0., fields.v[config.store_all_timesteps*it]) * phi_FS_cf) # at [i,j-1/2]

    return (flxx - np.roll(flxx,-1,0))/fields.dxcc + (flxy - np.roll(flxy,-1,1))/fields.dycc # at [i,j] # would need adapting for arbitrary grid (and dx varying in y and dy varying in x)


def fluxdiv(fields, flxfc, flxcf, factorfc, factorcf):
    """Calculating the flux divergence for a certain flx, can be applied for both the explicit and implicit parts (using different factors)"""

    flxfactorfc = factorfc*flxfc # at [i-1/2,j]
    flxfactorcf = factorcf*flxcf # at [i,j-1/2]

    return (flxfactorfc - np.roll(flxfactorfc,-1,0))/fields.dxcc + (flxfactorcf - np.roll(flxfactorcf,-1,1))/fields.dycc # at [i,j] # would need adapting for arbitrary grid (and dx varying in y and dy varying in x)


def adhimex_matrix_func(phi, config, fields, it, thetafc, thetacf, alpha):
    """Matrix function for the implicit part of the AdHImEx scheme"""

    if config.thirdordermatrix: # Matrix with third-order spatial discretisation (unclear whether stable as of 01-03-2026)
        if config.BC_x != 'periodic' or config.BC_y != 'periodic':
            raise ValueError('Third-order SD is not implemented with nonperiodic BCs.')
        return phi - config.dt*alpha*fluxdiv(fields, *third_order(config, fields, it, phi), thetafc, thetacf) # at [i,j]
    else: # Default fifth-order spatial discretisation
        return phi - config.dt*alpha*fluxdiv(fields, *fifth_order(config, fields, it, phi), thetafc, thetacf) # at [i,j]


def adhimex_ncp(config, fields, it, **kwargs):
    """Implement the AdHImEx scheme for the given time step - non-constancy-preserving version"""

    # Set up Butcher tableau
    AEx, AIm, nstages = adhimex_butcher()

    # Calculate the implicitness at each cell face
    implicitness_adhimex(config, fields, it, **kwargs)

    # Time step
    fEx, fIm = np.zeros((nstages+1, *np.shape(fields.tracer)[1:])), np.zeros((nstages+1, *np.shape(fields.tracer)[1:]))
    
    it_prev = config.store_all_timesteps*it + 1 - config.store_all_timesteps
    field_prev = fields.tracer[it_prev].copy()

    field_k = field_prev.copy()
    flxfc_HO, flxcf_HO = np.zeros_like(field_k), np.zeros_like(field_k)

    for ik in range(nstages):
        # Calculate the field at stage k          
        rhs_k = field_prev + config.dt*(np.dot(np.rollaxis(fEx[:ik,:],0,3), AEx[ik,:ik]) + np.dot(np.rollaxis(fIm[:ik,:],0,3), AIm[ik,:ik])) # at [i,j]

        if ik == 4 and np.any(fields.thetacc[config.store_all_timesteps*it]): # 22-12-2025: I think this is necessary for GMRES not breaking down because of existing convergence (when the matrix is full of zeros)
            matrix = partial(adhimex_matrix_func, config=config, fields=fields, it=it, thetafc=fields.thetafc[config.store_all_timesteps*it], thetacf=fields.thetacf[config.store_all_timesteps*it], alpha=AIm[ik,ik]) # at [i,j]
            solver = getattr(sv, config.solver)
            #field_k = solver(matrix, rhs_k, field_k, kiter=200, jiter=4, tolerance=1e-6)
            guess = field_prev.copy()
            field_k = solver(matrix, rhs_k, guess, kiter=200, jiter=4, tolerance=1e-6, it=it)
        else:
            field_k = rhs_k.copy()

        # Calculate the velocity times the fifth-order field approximation at faces
        flxkfc, flxkcf = fifth_order(config, fields, it, field_k) # at [i-1/2,j] and [i,j-1/2]

        fEx[ik,:] = fluxdiv(fields, flxkfc, flxkcf, 1.-fields.thetafc[config.store_all_timesteps*it], 1.-fields.thetacf[config.store_all_timesteps*it])
        fIm[ik,:] = fluxdiv(fields, flxkfc, flxkcf, fields.thetafc[config.store_all_timesteps*it], fields.thetacf[config.store_all_timesteps*it])

        # Accumulate the flux contributions from the stages (needed for FCT)
        flxfc_HO += AEx[-1,ik]*(1 - fields.thetafc[config.store_all_timesteps*it])*flxkfc + AIm[-1,ik]*fields.thetafc[config.store_all_timesteps*it]*flxkfc # at [i-1/2,j]
        flxcf_HO += AEx[-1,ik]*(1 - fields.thetacf[config.store_all_timesteps*it])*flxkcf + AIm[-1,ik]*fields.thetacf[config.store_all_timesteps*it]*flxkcf # at [i,j-1/2]

    # Implement FCT if required
    if config.FCT_2pass:
        fields.tracer[config.store_all_timesteps*it+1] = lim.FCT_2pass(config, fields, it, flxfc_HO, flxcf_HO)     
    elif config.FCT:
        fields.tracer[config.store_all_timesteps*it+1] = lim.FCT(config, fields, it, flxfc_HO, flxcf_HO)
    else:     
        fields.tracer[config.store_all_timesteps*it+1] = field_k.copy()


def adhimex(config, fields, it, iterations_convergence=np.zeros(10), **kwargs):
    """Implement the AdHImEx scheme for the given time step - constancy-preserving version"""

    it_prev = config.store_all_timesteps*it + 1 - config.store_all_timesteps
    field_prev = fields.tracer[it_prev].copy()

    # Set up Butcher tableau
    AEx, AIm, nstages = adhimex_butcher()

    # Calculate the implicitness at each cell face
    implicitness_adhimex(config, fields, it, **kwargs)
    
    # Time step
    fEx_c, fIm_c = np.zeros((nstages+1, *np.shape(fields.tracer)[1:])), np.zeros((nstages+1, *np.shape(fields.tracer)[1:]))
    fEx_f, fIm_f = np.zeros((nstages+1, *np.shape(fields.tracer)[1:])), np.zeros((nstages+1, *np.shape(fields.tracer)[1:]))
    field_k = field_prev.copy()
    flxfc_HO, flxcf_HO = np.zeros_like(field_k), np.zeros_like(field_k)

    for ik in range(nstages):
        # Calculate the field at stage k      
        if ik == 1 or ik == 2:
            rhs_k = field_prev + config.dt*(np.dot(np.rollaxis(fEx_c[:ik,:],0,3), AEx[ik,:ik]) + np.dot(np.rollaxis(fIm_c[:ik,:],0,3), AIm[ik,:ik])) # at [i,j]
        else: 
            rhs_k = field_prev + config.dt*(np.dot(np.rollaxis(fEx_f[:ik,:],0,3), AEx[ik,:ik]) + np.dot(np.rollaxis(fIm_f[:ik,:],0,3), AIm[ik,:ik])) # at [i,j]

        if ik == 4 and np.any(fields.thetacc[config.store_all_timesteps*it]): # 22-12-2025: I think this is necessary for GMRES not breaking down because of existing convergence (when the matrix is full of zeros)
            matrix = partial(adhimex_matrix_func, config=config, fields=fields, it=it, thetafc=fields.thetafc[config.store_all_timesteps*it], thetacf=fields.thetacf[config.store_all_timesteps*it], alpha=AIm[ik,ik]) # at [i,j]
            solver = getattr(sv, config.solver)
            guess = field_prev.copy()
            field_k = solver(matrix, rhs_k, guess, kiter=200, jiter=4, tolerance=1e-6, iterations_convergence=iterations_convergence, it=it)
        else:
            field_k = rhs_k.copy()

        # Calculate the velocity times the fifth-order field approximation at faces
        flxkfc, flxkcf = fifth_order(config, fields, it, field_k) # at [i-1/2,j] and [i,j-1/2]

        fEx_c[ik,:] = (1.-fields.thetacc[config.store_all_timesteps*it])*fluxdiv(fields, flxkfc, flxkcf, 1., 1.)
        fIm_c[ik,:] = fields.thetacc[config.store_all_timesteps*it]*fluxdiv(fields, flxkfc, flxkcf, 1., 1.)
        fEx_f[ik,:] = fluxdiv(fields, flxkfc, flxkcf, 1.-fields.thetafc[config.store_all_timesteps*it], 1.-fields.thetacf[config.store_all_timesteps*it])
        fIm_f[ik,:] = fluxdiv(fields, flxkfc, flxkcf, fields.thetafc[config.store_all_timesteps*it], fields.thetacf[config.store_all_timesteps*it])    
        
        # Accumulate the flux contributions from the stages (needed for FCT)
        flxfc_HO += AEx[-1,ik]*(1 - fields.thetafc[config.store_all_timesteps*it])*flxkfc + AIm[-1,ik]*fields.thetafc[config.store_all_timesteps*it]*flxkfc # at [i-1/2,j]
        flxcf_HO += AEx[-1,ik]*(1 - fields.thetacf[config.store_all_timesteps*it])*flxkcf + AIm[-1,ik]*fields.thetacf[config.store_all_timesteps*it]*flxkcf # at [i,j-1/2]

    # Implement FCT if required
    if config.BC_x == 'periodic' and config.BC_y == 'periodic': # to avoid having to implement boundaries for Hadley nonperiodic BC. Not planning to implement FCT results for this for now (22-03-2026)
        if config.FCT_2pass:
            fields.tracer[config.store_all_timesteps*it+1] = lim.FCT_2pass(config, fields, it, flxfc_HO, flxcf_HO)     
        elif config.FCT:
            fields.tracer[config.store_all_timesteps*it+1] = lim.FCT(config, fields, it, flxfc_HO, flxcf_HO)
        else:
            fields.tracer[config.store_all_timesteps*it+1] = field_k.copy() 
    else:     
        if config.FCT_2pass or config.FCT:
            raise ValueError('FCT is not implemented for periodic BCs.')
        else:
            fields.tracer[config.store_all_timesteps*it+1] = field_k.copy() 
