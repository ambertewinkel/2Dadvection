import numpy as np
from functools import partial
import src.solvers as sv
import src.limiter as lim


################# ADIMEX UPWIND SCHEME #################

def fluxdiv_first(config, fields, it, phi, factor_fc, factor_cf):
    """Calculating the first-order flux divergence for a certain phi, can be applied for bot the explicit and implicit parts (using different factors)"""

    phi_BS_fc = np.roll(phi,1,0) # backward in space (upwind if u>0) flux at face in x direction
    phi_FS_fc = phi # forward in space (upwind if u<0) flux at face in x direction
    phi_BS_cf = np.roll(phi,1,1) # backward in space (upwind if v>0) flux at face in y direction
    phi_FS_cf = phi # forward in space (upwind if v<0) flux at face in y direction
    flxx = factor_fc*(np.maximum(0., fields.u[it]) * phi_BS_fc + np.minimum(0., fields.u[it]) * phi_FS_fc) # at [i-1/2,j]
    flxy = factor_cf*(np.maximum(0., fields.v[it]) * phi_BS_cf + np.minimum(0., fields.v[it]) * phi_FS_cf) # at [i,j-1/2]

    return (flxx - np.roll(flxx,-1,0))/fields.dxcc + (flxy - np.roll(flxy,-1,1))/fields.dycc # at [i,j]


def implicitness_adimex_upwind(config, fields, it, **kwargs):
    """Calculate Courant numbers at cell centers and implicitness at cell centers and faces for AdImEx upwind scheme. Assumes nondivergent winds, and that dy is constant in x and dx is constant in y"""

    sum_abs_velarea = (abs(fields.u[it]) + abs(np.roll(fields.u[it],-1,0)))*fields.dycc + (abs(fields.v[it]) + abs(np.roll(fields.v[it],-1,1)))*fields.dxcc # at [i,j]
    fields.Ccc[it] = 0.5*config.dt*sum_abs_velarea/(fields.dxcc*fields.dycc) # at [i,j] (always nonnegative) # see Weller et al 2023 for definition

    # Calculate implicitness at cell centers and faces
    fields.thetacc[it] = np.maximum(0., 1. - config.factordiv/fields.Ccc[it]) # at [i,j]
    fields.thetafc[it] = np.maximum(fields.thetacc[it], np.roll(fields.thetacc[it],1,0)) # at [i-1/2,j]
    fields.thetacf[it] = np.maximum(fields.thetacc[it], np.roll(fields.thetacc[it],1,1)) # at [i,j-1/2]


def adimex_upwind_matrix_func(phi, config, fields, it, thetafc, thetacf):
    """Matrix function for the implicit part of the AdImEx upwind scheme"""

    return phi - config.dt*fluxdiv_first(config, fields, it, phi, thetafc, thetacf) # at [i,j]


def adimex_upwind(config, fields, it, tolerance=1e-6, kiter=10, jiter=4, **kwargs):
    """Implement the AdImEx upwind scheme for the given time step"""

    # Calculate the implicitness (1-1/(2C)) at each cell face
    implicitness_adimex_upwind(config, fields, it, **kwargs)

    # Calculate RHS (explicit) at cell faces
    rhs = fields.tracer[it] + config.dt*fluxdiv_first(config, fields, it, fields.tracer[it], 1.-fields.thetafc[it], 1.-fields.thetacf[it]) # at [i,j] 
    
    # Calculate LHS (implicit) upwind fluxes at cell faces    
    solver = config.solver # numpy, gcrk_matrix, gcrk_matrixfree # not sure if numpy is possible with a 4D matrix. # 17-11-2025: only gcrk_matrixfree implemented

    if np.any(fields.thetacc[it]): # avoids gmresm breaking down, only running the solver when there is a nonunit matrix
        matrix = partial(adimex_upwind_matrix_func, config=config, fields=fields, it=it, thetafc=fields.thetafc[it], thetacf=fields.thetacf[it]) # at [i,j]
        solver = getattr(sv, config.solver)
        fields.tracer[it+1] = solver(matrix, rhs, fields.tracer[it], kiter=kiter, jiter=jiter, tolerance=tolerance) # kiter, jiter, and tolerance can be set differently, e.g. for limiting
    else:
        fields.tracer[it+1] = rhs.copy()


################# ADHIMEX SCHEME #################

def implicitness_adhimex(config, fields, it, **kwargs):
    """Calculate Courant numbers at cell centers and implicitness at cell centers and faces for AdHImEx scheme. Assumes nondivergent winds, and that dy is constant in x and dx is constant in y"""

    sum_abs_velarea = (abs(fields.u[it]) + abs(np.roll(fields.u[it],-1,0)))*fields.dycc + (abs(fields.v[it]) + abs(np.roll(fields.v[it],-1,1)))*fields.dxcc # at [i,j]
    fields.Ccc[it] = 0.5*config.dt*sum_abs_velarea/(fields.dxcc*fields.dycc) # at [i,j] (always nonnegative) # see Weller et al 2023 for definition
    
    fields.thetacc[it] = 1. - 1./(1. + 0.7*np.maximum(0., fields.Ccc[it] - 1.4)) # at [i,j]
    #fields.thetacc[it] = np.zeros(np.shape(fields.thetacc[it]))
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

    flxfc = np.maximum(0., fields.u[it]) * phi_BS_fc + np.minimum(0., fields.u[it]) * phi_FS_fc # at [i-1/2,j]
    flxcf = np.maximum(0., fields.v[it]) * phi_BS_cf + np.minimum(0., fields.v[it]) * phi_FS_cf # at [i,j-1/2]

    return flxfc, flxcf


def fluxdiv(fields, flxfc, flxcf, factorfc, factorcf):
    """Calculating the flux divergence for a certain flx, can be applied for both the explicit and implicit parts (using different factors)"""

    flxfactorfc = factorfc*flxfc # at [i-1/2,j]
    flxfactorcf = factorcf*flxcf # at [i,j-1/2]

    return (flxfactorfc - np.roll(flxfactorfc,-1,0))/fields.dxcc + (flxfactorcf - np.roll(flxfactorcf,-1,1))/fields.dycc # at [i,j]


def adhimex_matrix_func(phi, config, fields, it, thetafc, thetacf, alpha):
    """Matrix function for the implicit part of the AdHImEx scheme"""

    return phi - config.dt*alpha*fluxdiv(fields, *fifth_order(config, fields, it, phi), thetafc, thetacf) # at [i,j]


def adhimex_ncp(config, fields, it, **kwargs):
    """Implement the AdHImEx scheme for the given time step - non-constancy-preserving version"""

    # Set up Butcher tableau
    AEx, AIm, nstages = adhimex_butcher()

    # Calculate the implicitness at each cell face
    implicitness_adhimex(config, fields, it, **kwargs)

    # Time step
    fEx, fIm = np.zeros((nstages+1, *np.shape(fields.tracer)[1:])), np.zeros((nstages+1, *np.shape(fields.tracer)[1:]))
    field_k = fields.tracer[it].copy()
    flxfc_HO, flxcf_HO = np.zeros_like(field_k), np.zeros_like(field_k)

    for ik in range(nstages):
        # Calculate the field at stage k          
        rhs_k = fields.tracer[it] + config.dt*(np.dot(np.rollaxis(fEx[:ik,:],0,3), AEx[ik,:ik]) + np.dot(np.rollaxis(fIm[:ik,:],0,3), AIm[ik,:ik])) # at [i,j]

        if ik == 4 and np.any(fields.thetacc[it]): # necessary for GMRES not breaking down because of existing convergence (when the matrix is full of zeros)
            matrix = partial(adhimex_matrix_func, config=config, fields=fields, it=it, thetafc=fields.thetafc[it], thetacf=fields.thetacf[it], alpha=AIm[ik,ik]) # at [i,j]
            solver = getattr(sv, config.solver)
            #field_k = solver(matrix, rhs_k, field_k, kiter=200, jiter=4, tolerance=1e-6)
            guess = fields.tracer[it].copy()
            field_k = solver(matrix, rhs_k, guess, kiter=200, jiter=4, tolerance=1e-6, it=it)
        else:
            field_k = rhs_k.copy()

        # Calculate the velocity times the fifth-order field approximation at faces
        flxkfc, flxkcf = fifth_order(config, fields, it, field_k) # at [i-1/2,j] and [i,j-1/2]

        fEx[ik,:] = fluxdiv(fields, flxkfc, flxkcf, 1.-fields.thetafc[it], 1.-fields.thetacf[it])
        fIm[ik,:] = fluxdiv(fields, flxkfc, flxkcf, fields.thetafc[it], fields.thetacf[it])

        # Accumulate the flux contributions from the stages (needed for FCT)
        flxfc_HO += AEx[-1,ik]*(1 - fields.thetafc[it])*flxkfc + AIm[-1,ik]*fields.thetafc[it]*flxkfc # at [i-1/2,j]
        flxcf_HO += AEx[-1,ik]*(1 - fields.thetacf[it])*flxkcf + AIm[-1,ik]*fields.thetacf[it]*flxkcf # at [i,j-1/2]

    # Implement FCT if required
    if config.FCT:
        fields.tracer[it+1] = lim.FCT(config, fields, it, flxfc_HO, flxcf_HO)
    else:     
        fields.tracer[it+1] = field_k.copy()


def adhimex(config, fields, it, iterations_convergence=np.zeros(10), **kwargs):
    """Implement the AdHImEx scheme for the given time step - constancy-preserving version"""

    # Set up Butcher tableau
    AEx, AIm, nstages = adhimex_butcher()

    # Calculate the implicitness at each cell face
    implicitness_adhimex(config, fields, it, **kwargs)
    
    # Time step
    fEx_c, fIm_c = np.zeros((nstages+1, *np.shape(fields.tracer)[1:])), np.zeros((nstages+1, *np.shape(fields.tracer)[1:]))
    fEx_f, fIm_f = np.zeros((nstages+1, *np.shape(fields.tracer)[1:])), np.zeros((nstages+1, *np.shape(fields.tracer)[1:]))
    field_k = fields.tracer[it].copy()
    flxfc_HO, flxcf_HO = np.zeros_like(field_k), np.zeros_like(field_k)

    for ik in range(nstages):
        # Calculate the field at stage k      
        if ik == 1 or ik == 2:
            rhs_k = fields.tracer[it] + config.dt*(np.dot(np.rollaxis(fEx_c[:ik,:],0,3), AEx[ik,:ik]) + np.dot(np.rollaxis(fIm_c[:ik,:],0,3), AIm[ik,:ik])) # at [i,j]
        else: 
            rhs_k = fields.tracer[it] + config.dt*(np.dot(np.rollaxis(fEx_f[:ik,:],0,3), AEx[ik,:ik]) + np.dot(np.rollaxis(fIm_f[:ik,:],0,3), AIm[ik,:ik])) # at [i,j]

        if ik == 4 and np.any(fields.thetacc[it]): # necessary for GMRES not breaking down because of existing convergence (when the matrix is full of zeros)
            matrix = partial(adhimex_matrix_func, config=config, fields=fields, it=it, thetafc=fields.thetafc[it], thetacf=fields.thetacf[it], alpha=AIm[ik,ik]) # at [i,j]
            solver = getattr(sv, config.solver)
            guess = fields.tracer[it].copy()
            field_k = solver(matrix, rhs_k, guess, kiter=200, jiter=4, tolerance=1e-6, iterations_convergence=iterations_convergence, it=it)
        else:
            field_k = rhs_k.copy()

        # Calculate the velocity times the fifth-order field approximation at faces
        flxkfc, flxkcf = fifth_order(config, fields, it, field_k) # at [i-1/2,j] and [i,j-1/2]

        fEx_c[ik,:] = (1.-fields.thetacc[it])*fluxdiv(fields, flxkfc, flxkcf, 1., 1.)
        fIm_c[ik,:] = fields.thetacc[it]*fluxdiv(fields, flxkfc, flxkcf, 1., 1.)
        fEx_f[ik,:] = fluxdiv(fields, flxkfc, flxkcf, 1.-fields.thetafc[it], 1.-fields.thetacf[it])
        fIm_f[ik,:] = fluxdiv(fields, flxkfc, flxkcf, fields.thetafc[it], fields.thetacf[it])    
        
        # Accumulate the flux contributions from the stages (needed for FCT)
        flxfc_HO += AEx[-1,ik]*(1 - fields.thetafc[it])*flxkfc + AIm[-1,ik]*fields.thetafc[it]*flxkfc # at [i-1/2,j]
        flxcf_HO += AEx[-1,ik]*(1 - fields.thetacf[it])*flxkcf + AIm[-1,ik]*fields.thetacf[it]*flxkcf # at [i,j-1/2]

    # Implement FCT if required
    if config.BC_x == 'periodic' and config.BC_y == 'periodic':
        if config.FCT:
            fields.tracer[it+1] = lim.FCT(config, fields, it, flxfc_HO, flxcf_HO)
        else:
            fields.tracer[it+1] = field_k.copy() 
    else:     
        if config.FCT:
            raise ValueError('FCT is not implemented for nonperiodic BCs.')
        else:
            fields.tracer[it+1] = field_k.copy() 
