import numpy as np
from functools import partial
from src.solvers import gcrk_matrixfree


def upwind(config, fields, **kwargs):
    # Implement the upwind scheme for the given time step
    
    # Calculate upwind fluxes at cell faces
    fields.flxx = np.maximum(0., fields.u) * np.roll(fields.tracer,1,0) + np.minimum(0.0, fields.u) * fields.tracer # defined at [i-1/2,j]
    fields.flxy = np.maximum(0., fields.v) * np.roll(fields.tracer,1,1) + np.minimum(0.0, fields.v) * fields.tracer # defined at [i,j-1/2]

    # Calculate the new tracer field
    fields.tracer = fields.tracer + config.dt/config.dx*(fields.flxx - np.roll(fields.flxx,-1,0)) + config.dt/config.dy*(fields.flxy - np.roll(fields.flxy,-1,1)) # defined at [i,j]
    

################# ADIMEX UPWIND SCHEME #################

def implicitness_adimex_upwind(config, fields, **kwargs):

    # Calculate Courant numbers at cell faces
    Cfc = config.dt/config.dx * fields.u # defined at [i-1/2,j]
    Ccf = config.dt/config.dy * fields.v # defined at [i,j-1/2]

    # Calculate Courant numbers at cell centers
    C_in_cc = np.maximum(0.,Cfc) - np.minimum(0.,np.roll(Cfc,-1,0)) + np.maximum(0.,Ccf) - np.minimum(0.,np.roll(Ccf,-1,1)) # defined at [i,j]
    C_out_cc = - np.minimum(0.,Cfc) + np.maximum(0.,np.roll(Cfc,-1,0)) - np.minimum(0.,Ccf) + np.maximum(0.,np.roll(Ccf,-1,1)) # defined at [i,j]
    C_total = C_in_cc + C_out_cc # defined at [i,j]

    # Calculate implicitness at cell centers and faces
    thetacc = np.maximum(0., 1. - 1./C_total) # defined at [i,j]
    thetafc = np.maximum(thetacc, np.roll(thetacc,1,0)) # defined at [i-1/2,j]
    thetacf = np.maximum(thetacc, np.roll(thetacc,1,1)) # defined at [i,j-1/2]

    return thetafc, thetacf


def adimex_upwind(config, fields, **kwargs):

    # Calculate the implicitness (1-1/C_total) at each cell face
    
    thetafc, thetacf = implicitness_adimex_upwind(config, fields, **kwargs)

    # Calculate RHS (explicit) upwind fluxes at cell faces
    fields.flxx = (1. - thetafc)*(np.maximum(0., fields.u) * np.roll(fields.tracer,1,0) + np.minimum(0.0, fields.u) * fields.tracer) # defined at [i-1/2,j]
    fields.flxy = (1. - thetacf)*(np.maximum(0., fields.v) * np.roll(fields.tracer,1,1) + np.minimum(0.0, fields.v) * fields.tracer) # defined at [i,j-1/2]

    # Calculate the new tracer field
    rhs = fields.tracer + config.dt/config.dx*(fields.flxx - np.roll(fields.flxx,-1,0)) + config.dt/config.dy*(fields.flxy - np.roll(fields.flxy,-1,1)) # defined at [i,j]
    
    # Calculate LHS (implicit) upwind fluxes at cell faces    
    solver = config.solver # numpy, gcrk_matrix, gcrk_matrixfree # not sure if numpy is possible with a 4D matrix.
 
    #if solver == 'numpy':
    #    M = 
    #    fields.tracer = np.linalg.solve(M, rhs)#(config, fields, thetafc, thetacf, rhs)
    #elif solver == 'gcrk_matrix':
#
    if solver == 'gcrk_matrixfree':
        matrix = partial(adimex_upwind_matrix_func, config=config, fields=fields, thetafc=thetafc, thetacf=thetacf)
        fields.tracer = gcrk_matrixfree(matrix, rhs, fields.tracer, kiter=5, jiter=5)
    else:
        raise ValueError(f"Unknown solver {solver}")


def adimex_upwind_matrix_func(psi, config, fields, thetafc, thetacf):
    
    # Overwriting the explicit fluxes used just now with the implicit part
    fields.flxx = thetafc*(np.maximum(0., fields.u) * np.roll(psi,1,0) + np.minimum(0., fields.u) * psi) # defined at [i-1/2,j] # !!! very similar to explicit flux calc. Perhaps put in one function and call twice?
    fields.flxy = thetacf*(np.maximum(0., fields.v) * np.roll(psi,1,1) + np.minimum(0., fields.v) * psi) # defined at [i,j-1/2]

    return psi - config.dt/config.dx*(fields.flxx - np.roll(fields.flxx,-1,0)) - config.dt/config.dy*(fields.flxy - np.roll(fields.flxy,-1,1)) # defined at [i,j]


################# ADHIMEX SCHEME #################

def implicitness_adhimex(config, fields, **kwargs):

    # Calculate Courant numbers at cell faces
    #Cfc = config.dt/config.dx*fields.u # defined at [i-1/2,j]
    #Ccf = config.dt/config.dy*fields.v # defined at [i,j-1/2]

    # Calculate implicitness at cell centers and faces
    #thetafc = 1. - 1./(1. + 0.7*np.maximum(0., np.abs(Cfc) - 1.4)) # defined at [i-1/2,j]
    #thetacf = 1. - 1./(1. + 0.7*np.maximum(0., np.abs(Ccf) - 1.4)) # defined at [i,j-1/2]

    # Calculate Courant numbers at cell centers
    sum_abs_vel = abs(fields.u) + abs(np.roll(fields.u,-1,0)) + abs(fields.v) + abs(np.roll(fields.v,-1,1)) # defined at [i,j]
    Ccc = 0.5*config.dt*sum_abs_vel/(config.dx*config.dy) # defined at [i,j] (always nonnegative) # see Weller et al 2023 for definition

    thetacc = 1. - 1./(1. + 0.7*np.maximum(0., Ccc - 1.4)) # defined at [i,j]
    thetafc = np.maximum(thetacc, np.roll(thetacc,1,0)) # defined at [i-1/2,j]
    thetacf = np.maximum(thetacc, np.roll(thetacc,1,1)) # defined at [i,j-1/2]

    return thetafc, thetacf


def adhimex_butcher():
    """Set up Butcher tableau"""
    # Butcher tableau for explicit part (left tableau from Ullrich and Jablonowski 2012. See the Weller Lock and Wood (2013) UJ3(1+e,3,2) scheme)   
    AEx = np.array([[0., 0., 0., 0., 0.],[0., 0., 0., 0., 0.],[0., 1., 0., 0., 0.],[0., 0.25, 0.25, 0., 0.],[0., 1/6, 1/6, 2/3, 0.]])

    # Butcher tableau for implicit part (right tableau from Ullrich and Jablonowski 2012. See the Weller Lock and Wood (2013) UJ3(1+e,3,2) scheme)   
    AIm = np.array([[0., 0., 0., 0., 0.],[0.5, 0., 0., 0., 0.],[0.5, 0., 0., 0., 0.],[0.5, 0., 0., 0., 0.],[0.5, 0., 0., 0., 0.5]])

    nstages = np.shape(AIm)[1]

    return AEx, AIm, nstages


def flux_divergence(config, fields, psi, factor_fc, factor_cf):
    # Overwriting the explicit fluxes used just now with the implicit part
    psi_BS_fc = -1./20*np.roll(psi,-1,0) + 9./20.*psi + 47./60.*np.roll(psi,1,0) - 13./60.*np.roll(psi,2,0) + 1./30.*np.roll(psi,3,0) # backward in space (upwind if u>0) flux at face in x direction
    psi_FS_fc = -1./20*np.roll(psi,2,0) + 9./20.*np.roll(psi,1,0) + 47./60.*psi - 13./60.*np.roll(psi,-1,0) + 1./30.*np.roll(psi,-2,0) # forward in space (upwind if u<0) flux at face in x direction # !!! need to fix indexing
    psi_BS_cf = -1./20*np.roll(psi,-1,1) + 9./20.*psi + 47./60.*np.roll(psi,1,1) - 13./60.*np.roll(psi,2,1) + 1./30.*np.roll(psi,3,1) # backward in space (upwind if v>0) flux at face in y direction
    psi_FS_cf = -1./20*np.roll(psi,2,1) + 9./20.*np.roll(psi,1,1) + 47./60.*psi - 13./60.*np.roll(psi,-1,1) + 1./30.*np.roll(psi,-2,1) # forward in space (upwind if v<0) flux at face in y direction
    flxx = factor_fc*(np.maximum(0., fields.u) * psi_BS_fc + np.minimum(0., fields.u) * psi_FS_fc) # defined at [i-1/2,j]
    flxy = factor_cf*(np.maximum(0., fields.v) * psi_BS_cf + np.minimum(0., fields.v) * psi_FS_cf) # defined at [i,j-1/2]

    return (flxx - np.roll(flxx,-1,0))/config.dx + (flxy - np.roll(flxy,-1,1))/config.dy # defined at [i,j]



def adhimex(config, fields, **kwargs):

    # Set up Butcher tableau
    AEx, AIm, nstages = adhimex_butcher()

    # Calculate the implicitness at each cell face
    thetafc, thetacf = implicitness_adhimex(config, fields, **kwargs)

    # Time step
    fEx, fIm = np.zeros((nstages+1, *np.shape(fields.tracer))), np.zeros((nstages+1, *np.shape(fields.tracer)))
    field_k = fields.tracer.copy()
    for ik in range(nstages):#+1):
        # Calculate the field at stage k          
        rhs_k = fields.tracer + config.dt*(np.dot(np.rollaxis(fEx[:ik,:],0,3), AEx[ik,:ik]) + np.dot(np.rollaxis(fIm[:ik,:],0,3), AIm[ik,:ik])) # defined at [i,j]
        matrix = partial(adhimex_matrix_func, config=config, fields=fields, thetafc=thetafc, thetacf=thetacf, alpha=AIm[ik,ik])
        field_k = gcrk_matrixfree(matrix, rhs_k, field_k, kiter=10, jiter=10) # I need to store the intermediate RK stages

        fEx[ik,:] = flux_divergence(config, fields, field_k, 1.-thetafc, 1.-thetacf)
        fIm[ik,:] = flux_divergence(config, fields, field_k, thetafc, thetacf)
        
        # Calculate the flux based on the field at stage k
        #flx_k[ik,:] = uf[it]*fluxfn(field_k) # [i] at i-1/2
        #fEx[ik,:] = -ddx((1 - beta[it])*flx_k[ik,:], np.roll((1 - beta[it])*flx_k[ik,:],-1), dxc)
        #fIm[ik,:] = -ddx(beta[it]*flx_k[ik,:], np.roll(beta[it]*flx_k[ik,:],-1), dxc)   
        #flx_contribution_from_stage_k[ik,:] = AEx[-1,ik]*(1 - beta[it])*flx_k[ik,:] + AIm[-1,ik]*beta[it]*flx_k[ik,:]
    fields.tracer = field_k.copy()

    ################



    # Overwriting the explicit fluxes used just now with the implicit part
    #psi_BS_fc = -1./20.*np.roll(fields.tracer,-1,0) + 9./20.*fields.tracer + 47./60.*np.roll(fields.tracer,1,0) - 13./60.*np.roll(fields.tracer,2,0) + 1./30.*np.roll(fields.tracer,3,0) # backward in space (upwind if u>0) flux at face in x direction
    #psi_FS_fc = -1./20.*np.roll(fields.tracer,2,0) + 9./20.*np.roll(fields.tracer,1,0) + 47./60.*fields.tracer - 13./60.*np.roll(fields.tracer,-1,0) + 1./30.*np.roll(fields.tracer,-2,0) # forward in space (upwind if u<0) flux at face in x direction
    #psi_BS_cf = -1./20.*np.roll(fields.tracer,-1,1) + 9./20.*fields.tracer + 47./60.*np.roll(fields.tracer,1,1) - 13./60.*np.roll(fields.tracer,2,1) + 1./30.*np.roll(fields.tracer,3,1) # backward in space (upwind if v>0) flux at face in y direction
    #psi_FS_cf = -1./20.*np.roll(fields.tracer,2,1) + 9./20.*np.roll(fields.tracer,1,1) + 47./60.*fields.tracer - 13./60.*np.roll(fields.tracer,-1,1) + 1./30.*np.roll(fields.tracer,-2,1) # forward in space (upwind if v<0) flux at face in y direction
    #fields.flxx = (1. - thetafc)*(np.maximum(0., fields.u) * psi_BS_fc + np.minimum(0., fields.u) * psi_FS_fc) # defined at [i-1/2,j] # !!! very similar to explicit flux calc. Perhaps put in one function and call twice?
    #fields.flxy = (1. - thetacf)*(np.maximum(0., fields.v) * psi_BS_cf + np.minimum(0., fields.v) * psi_FS_cf) # defined at [i,j-1/2]

    # Calculate RHS (explicit) upwind fluxes at cell faces
    #rhs = fields.tracer + config.dt/config.dx*(fields.flxx - np.roll(fields.flxx,-1,0)) + config.dt/config.dy*(fields.flxy - np.roll(fields.flxy,-1,1)) # defined at [i,j]
    #rhs = fields.tracer + config.dt*AEx*flux_divergence(config, fields, fields.tracer, 1.-thetafc, 1.-thetacf) # defined at [i,j]
    #rhs = fields.tracer + config.dt*AEx*flux_divergence(config, fields, fields.tracer, 1.-thetafc, 1.-thetacf) # defined at [i,j]
#
    ## Calculate LHS (implicit) upwind fluxes at cell faces    
    #solver = config.solver # numpy, gcrk_matrix, gcrk_matrixfree # not sure if numpy is possible with a 4D matrix.
 #
    #if solver == 'gcrk_matrixfree':
    #    matrix = partial(adhimex_matrix_func, config=config, fields=fields, thetafc=thetafc, thetacf=thetacf)
    #    fields.tracer = gcrk_matrixfree(matrix, rhs, fields.tracer, kiter=10, jiter=10) # !!! field_k and implement the intermediate RK stages
    #else:
    #    raise ValueError(f"Unknown solver {solver}")


def adhimex_matrix_func(psi, config, fields, thetafc, thetacf, alpha):
    
    ## Overwriting the explicit fluxes used just now with the implicit part
    #psi_BS_fc = -1./20*np.roll(psi,-1,0) + 9./20.*psi + 47./60.*np.roll(psi,1,0) - 13./60.*np.roll(psi,2,0) + 1./30.*np.roll(psi,3,0) # backward in space (upwind if u>0) flux at face in x direction
    #psi_FS_fc = -1./20*np.roll(psi,1,0) + 9./20.*psi + 47./60.*np.roll(psi,-1,0) - 13./60.*np.roll(psi,-2,0) + 1./30.*np.roll(psi,-3,0) # forward in space (upwind if u<0) flux at face in x direction # !!! need to fix indexing
    #psi_BS_cf = -1./20*np.roll(psi,-1,1) + 9./20.*psi + 47./60.*np.roll(psi,1,1) - 13./60.*np.roll(psi,2,1) + 1./30.*np.roll(psi,3,1) # backward in space (upwind if v>0) flux at face in y direction
    #psi_FS_cf = -1./20*np.roll(psi,1,1) + 9./20.*psi + 47./60.*np.roll(psi,-1,1) - 13./60.*np.roll(psi,-2,1) + 1./30.*np.roll(psi,-3,1) # forward in space (upwind if v<0) flux at face in y direction
    #fields.flxx = thetafc*(np.maximum(0., fields.u) * psi_BS_fc + np.minimum(0., fields.u) * psi_FS_fc) # defined at [i-1/2,j] # !!! very similar to explicit flux calc. Perhaps put in one function and call twice?
    #fields.flxy = thetacf*(np.maximum(0., fields.v) * psi_BS_cf + np.minimum(0., fields.v) * psi_FS_cf) # defined at [i,j-1/2]
    #return psi - config.dt/config.dx*(fields.flxx - np.roll(fields.flxx,-1,0)) - config.dt/config.dy*(fields.flxy - np.roll(fields.flxy,-1,1)) # defined at [i,j]

    fluxdiv = flux_divergence(config, fields, psi, thetafc, thetacf)

    return psi - config.dt*alpha*fluxdiv # defined at [i,j]


def adhimex_bkp(config, fields, **kwargs):

    # Calculate the implicitness at each cell face
    thetafc, thetacf = implicitness_adhimex(config, fields, **kwargs)

    # Overwriting the explicit fluxes used just now with the implicit part
    psi_BS_fc = -1./20.*np.roll(fields.tracer,-1,0) + 9./20.*fields.tracer + 47./60.*np.roll(fields.tracer,1,0) - 13./60.*np.roll(fields.tracer,2,0) + 1./30.*np.roll(fields.tracer,3,0) # backward in space (upwind if u>0) flux at face in x direction
    psi_FS_fc = -1./20.*np.roll(fields.tracer,2,0) + 9./20.*np.roll(fields.tracer,1,0) + 47./60.*fields.tracer - 13./60.*np.roll(fields.tracer,-1,0) + 1./30.*np.roll(fields.tracer,-2,0) # forward in space (upwind if u<0) flux at face in x direction
    psi_BS_cf = -1./20.*np.roll(fields.tracer,-1,1) + 9./20.*fields.tracer + 47./60.*np.roll(fields.tracer,1,1) - 13./60.*np.roll(fields.tracer,2,1) + 1./30.*np.roll(fields.tracer,3,1) # backward in space (upwind if v>0) flux at face in y direction
    psi_FS_cf = -1./20.*np.roll(fields.tracer,2,1) + 9./20.*np.roll(fields.tracer,1,1) + 47./60.*fields.tracer - 13./60.*np.roll(fields.tracer,-1,1) + 1./30.*np.roll(fields.tracer,-2,1) # forward in space (upwind if v<0) flux at face in y direction
    fields.flxx = (1. - thetafc)*(np.maximum(0., fields.u) * psi_BS_fc + np.minimum(0., fields.u) * psi_FS_fc) # defined at [i-1/2,j] # !!! very similar to explicit flux calc. Perhaps put in one function and call twice?
    fields.flxy = (1. - thetacf)*(np.maximum(0., fields.v) * psi_BS_cf + np.minimum(0., fields.v) * psi_FS_cf) # defined at [i,j-1/2]

    # Calculate RHS (explicit) upwind fluxes at cell faces
    rhs = fields.tracer + config.dt/config.dx*(fields.flxx - np.roll(fields.flxx,-1,0)) + config.dt/config.dy*(fields.flxy - np.roll(fields.flxy,-1,1)) # defined at [i,j]

    # Calculate LHS (implicit) upwind fluxes at cell faces    
    solver = config.solver # numpy, gcrk_matrix, gcrk_matrixfree # not sure if numpy is possible with a 4D matrix.
 
    if solver == 'gcrk_matrixfree':
        matrix = partial(adhimex_matrix_func, config=config, fields=fields, thetafc=thetafc, thetacf=thetacf)
        fields.tracer = gcrk_matrixfree(matrix, rhs, fields.tracer, kiter=10, jiter=10) # !!! field_k and implement the intermediate RK stages
    else:
        raise ValueError(f"Unknown solver {solver}")