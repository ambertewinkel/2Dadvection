import numpy as np
from functools import partial
from src.solvers import gcrk_matrixfree


################# UPWIND SCHEME #################

def fluxdiv_first(config, fields, it, phi, factor_fc, factor_cf):
    """Calculating the first-order flux divergence for a certain phi, can be applied for bot the explicit and implicit parts (using different factors)"""

    phi_BS_fc = np.roll(phi,1,0) # backward in space (upwind if u>0) flux at face in x direction
    phi_FS_fc = phi # forward in space (upwind if u<0) flux at face in x direction
    phi_BS_cf = np.roll(phi,1,1) # backward in space (upwind if v>0) flux at face in y direction
    phi_FS_cf = phi # forward in space (upwind if v<0) flux at face in y direction
    flxx = factor_fc*(np.maximum(0., fields.u[it]) * phi_BS_fc + np.minimum(0., fields.u[it]) * phi_FS_fc) # at [i-1/2,j]
    flxy = factor_cf*(np.maximum(0., fields.v[it]) * phi_BS_cf + np.minimum(0., fields.v[it]) * phi_FS_cf) # at [i,j-1/2]

    return (flxx - np.roll(flxx,-1,0))/config.dx + (flxy - np.roll(flxy,-1,1))/config.dy # at [i,j]


def upwind(config, fields, it, **kwargs):
    """Implement the upwind scheme for the given time step"""

    fields.tracer[it+1] = fields.tracer[it] + config.dt*fluxdiv_first(config, fields, it, fields.tracer[it], 1., 1.) # at [i,j]


################# ADIMEX UPWIND SCHEME #################

def implicitness_adimex_upwind(config, fields, it, **kwargs):
    """Calculate Courant numbers at cell centers and implicitness at cell centers and faces for AdImEx upwind scheme"""

    # Calculate Courant numbers at cell faces
    Cfc = config.dt/config.dx * fields.u[it] # at [i-1/2,j]
    Ccf = config.dt/config.dy * fields.v[it] # at [i,j-1/2]

    # Calculate Courant numbers at cell centers
    C_in_cc = np.maximum(0.,Cfc) - np.minimum(0.,np.roll(Cfc,-1,0)) + np.maximum(0.,Ccf) - np.minimum(0.,np.roll(Ccf,-1,1)) # at [i,j]
    C_out_cc = - np.minimum(0.,Cfc) + np.maximum(0.,np.roll(Cfc,-1,0)) - np.minimum(0.,Ccf) + np.maximum(0.,np.roll(Ccf,-1,1)) # at [i,j]
    fields.Ccc[it] = 0.5*(C_in_cc + C_out_cc) # at [i,j] (always nonnegative) 

    # Calculate implicitness at cell centers and faces
    fields.thetacc[it] = np.maximum(0., 1. - 0.5/fields.Ccc[it]) # at [i,j] # using 2C here instead of C preserves monotonicity better (17-11-2025: check if guaranteed monotonicity?)
    fields.thetafc[it] = np.maximum(fields.thetacc[it], np.roll(fields.thetacc[it],1,0)) # at [i-1/2,j]
    fields.thetacf[it] = np.maximum(fields.thetacc[it], np.roll(fields.thetacc[it],1,1)) # at [i,j-1/2]


def adimex_upwind_matrix_func(phi, config, fields, it, thetafc, thetacf):
    """Matrix function for the implicit part of the AdImEx upwind scheme"""

    return phi - config.dt*fluxdiv_first(config, fields, it, phi, thetafc, thetacf) # at [i,j]


def adimex_upwind(config, fields, it, **kwargs):
    """Implement the AdImEx upwind scheme for the given time step"""

    # Calculate the implicitness (1-1/(2C)) at each cell face
    implicitness_adimex_upwind(config, fields, it, **kwargs)

    # Calculate RHS (explicit) at cell faces
    rhs = fields.tracer[it] + config.dt*fluxdiv_first(config, fields, it, fields.tracer[it], 1.-fields.thetafc[it], 1.-fields.thetacf[it]) # at [i,j] 
    
    # Calculate LHS (implicit) upwind fluxes at cell faces    
    solver = config.solver # numpy, gcrk_matrix, gcrk_matrixfree # not sure if numpy is possible with a 4D matrix. # 17-11-2025: only gcrk_matrixfree implemented

    if solver == 'gcrk_matrixfree':
        matrix = partial(adimex_upwind_matrix_func, config=config, fields=fields, it=it, thetafc=fields.thetafc[it], thetacf=fields.thetacf[it])
        fields.tracer[it+1] = gcrk_matrixfree(config, matrix, rhs, fields.tracer[it], kiter=5, jiter=5)
    else:
        raise ValueError(f"Unknown solver {solver}")


################# ADHIMEX SCHEME #################

def implicitness_adhimex(config, fields, it, **kwargs):
    """Calculate Courant numbers at cell centers and implicitness at cell centers and faces for AdHImEx scheme"""

    sum_abs_velarea = (abs(fields.u[it]) + abs(np.roll(fields.u[it],-1,0)))*config.dy + (abs(fields.v[it]) + abs(np.roll(fields.v[it],-1,1)))*config.dx # at [i,j]
    fields.Ccc[it] = 0.5*config.dt*sum_abs_velarea/(config.dx*config.dy) # at [i,j] (always nonnegative) # see Weller et al 2023 for definition
    fields.thetacc[it] = 1. - 1./(1. + 0.7*np.maximum(0., fields.Ccc[it] - 1.4)) # at [i,j]
    fields.thetafc[it] = np.maximum(fields.thetacc[it], np.roll(fields.thetacc[it],1,0)) # at [i-1/2,j]
    fields.thetacf[it] = np.maximum(fields.thetacc[it], np.roll(fields.thetacc[it],1,1)) # at [i,j-1/2]


def adhimex_butcher():
    """Sets up AdHImEx Butcher tableau"""

    # Butcher tableau for explicit part (left tableau from Ullrich and Jablonowski 2012. See the Weller Lock and Wood (2013) UJ3(1+e,3,2) scheme)   
    AEx = np.array([[0., 0., 0., 0., 0.],[0., 0., 0., 0., 0.],[0., 1., 0., 0., 0.],[0., 0.25, 0.25, 0., 0.],[0., 1/6, 1/6, 2/3, 0.]])

    # Butcher tableau for implicit part (right tableau from Ullrich and Jablonowski 2012. See the Weller Lock and Wood (2013) UJ3(1+e,3,2) scheme)   
    AIm = np.array([[0., 0., 0., 0., 0.],[0.5, 0., 0., 0., 0.],[0.5, 0., 0., 0., 0.],[0.5, 0., 0., 0., 0.],[0.5, 0., 0., 0., 0.5]])

    nstages = np.shape(AIm)[1]

    return AEx, AIm, nstages


def fluxdiv_fifth(config, fields, it,  phi, factor_fc, factor_cf):
    """Calculating the fifth-order flux divergence for a certain phi, can be applied for bot the explicit and implicit parts (using different factors)"""

    phi_BS_fc = -1./20*np.roll(phi,-1,0) + 9./20.*phi + 47./60.*np.roll(phi,1,0) - 13./60.*np.roll(phi,2,0) + 1./30.*np.roll(phi,3,0) # backward in space (upwind if u>0) flux at face in x direction
    phi_FS_fc = -1./20*np.roll(phi,2,0) + 9./20.*np.roll(phi,1,0) + 47./60.*phi - 13./60.*np.roll(phi,-1,0) + 1./30.*np.roll(phi,-2,0) # forward in space (upwind if u<0) flux at face in x direction
    phi_BS_cf = -1./20*np.roll(phi,-1,1) + 9./20.*phi + 47./60.*np.roll(phi,1,1) - 13./60.*np.roll(phi,2,1) + 1./30.*np.roll(phi,3,1) # backward in space (upwind if v>0) flux at face in y direction
    phi_FS_cf = -1./20*np.roll(phi,2,1) + 9./20.*np.roll(phi,1,1) + 47./60.*phi - 13./60.*np.roll(phi,-1,1) + 1./30.*np.roll(phi,-2,1) # forward in space (upwind if v<0) flux at face in y direction
    flxx = factor_fc*(np.maximum(0., fields.u[it]) * phi_BS_fc + np.minimum(0., fields.u[it]) * phi_FS_fc) # at [i-1/2,j]
    flxy = factor_cf*(np.maximum(0., fields.v[it]) * phi_BS_cf + np.minimum(0., fields.v[it]) * phi_FS_cf) # at [i,j-1/2]

    return (flxx - np.roll(flxx,-1,0))/config.dx + (flxy - np.roll(flxy,-1,1))/config.dy # at [i,j]


def adhimex_matrix_func(phi, config, fields, it, thetafc, thetacf, alpha):
    """Matrix function for the implicit part of the AdHImEx scheme"""
    
    return phi - config.dt*alpha*fluxdiv_fifth(config, fields, it, phi, thetafc, thetacf) # at [i,j]


def adhimex(config, fields, it, **kwargs):
    """Implement the AdHImEx scheme for the given time step"""

    # Set up Butcher tableau
    AEx, AIm, nstages = adhimex_butcher()

    # Calculate the implicitness at each cell face
    implicitness_adhimex(config, fields, it, **kwargs)

    # Time step
    fEx, fIm = np.zeros((nstages+1, *np.shape(fields.tracer)[1:])), np.zeros((nstages+1, *np.shape(fields.tracer)[1:]))
    field_k = fields.tracer[it].copy()
    for ik in range(nstages):
        # Calculate the field at stage k          
        rhs_k = fields.tracer[it] + config.dt*(np.dot(np.rollaxis(fEx[:ik,:],0,3), AEx[ik,:ik]) + np.dot(np.rollaxis(fIm[:ik,:],0,3), AIm[ik,:ik])) # at [i,j]
        matrix = partial(adhimex_matrix_func, config=config, fields=fields, it=it, thetafc=fields.thetafc[it], thetacf=fields.thetacf[it], alpha=AIm[ik,ik]) # at [i,j]
        field_k = gcrk_matrixfree(config, matrix, rhs_k, field_k, kiter=10, jiter=10)

        fEx[ik,:] = fluxdiv_fifth(config, fields, it, field_k, 1.-fields.thetafc[it], 1.-fields.thetacf[it])
        fIm[ik,:] = fluxdiv_fifth(config, fields, it, field_k, fields.thetafc[it], fields.thetacf[it])

        # Calculate the flux based on the field at stage k (legacy code, but kept for reference for later FCT implementation)
        #flx_k[ik,:] = uf[it]*fluxfn(field_k) # [i] at i-1/2
        #fEx[ik,:] = -ddx((1 - beta[it])*flx_k[ik,:], np.roll((1 - beta[it])*flx_k[ik,:],-1), dxc)
        #fIm[ik,:] = -ddx(beta[it]*flx_k[ik,:], np.roll(beta[it]*flx_k[ik,:],-1), dxc)   
        #flx_contribution_from_stage_k[ik,:] = AEx[-1,ik]*(1 - beta[it])*flx_k[ik,:] + AIm[-1,ik]*beta[it]*flx_k[ik,:]

    fields.tracer[it+1] = field_k.copy()