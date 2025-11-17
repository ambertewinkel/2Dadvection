import numpy as np
import matplotlib.pyplot as plt

def velocity(config, fields, it):
    # Define velocity fields at time step it (actually velocities taken at it + 0.5 for second-order accuracy)
    globals()[config.velocity_setting](config, fields, it)


def constant_u(config, fields, it):
    fields.u[:,:] = config.constant_u
    fields.v[:,:] = 0.


def constant_v(config, fields, it):
    fields.u[:,:] = 0.
    fields.v[:,:] = config.constant_v


def constant_uv(config, fields, it):
    fields.u[:,:] = config.constant_u
    fields.v[:,:] = config.constant_v


def swift_nondiv(config, fields, it):
    # Non-divergent velocity field at half time levels

    Lx = config.xmax - config.xmin
    Ly = config.ymax - config.ymin
    
    ut = config.u0*(it + 0.5)*config.dt

    fields.u = config.u0*np.sin(np.pi*(fields.xfc + 0.5*Lx - ut)/Lx)*np.sin(np.pi*(fields.xfc + 0.5*Lx - ut)/Lx)*np.sin(2.*np.pi*(fields.yfc + 0.5*Ly - ut)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.u0
    
    fields.v = -config.u0*np.sin(2.*np.pi*(fields.xcf + 0.5*Lx - ut)/Lx)*np.sin(np.pi*(fields.ycf + 0.5*Ly - ut)/Ly)*np.sin(np.pi*(fields.ycf + 0.5*Ly - ut)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.u0


def swift_nondiv_streamfunction(config, fields, it):
    # Non-divergent velocity field at half time levels using the streamfunction

    Lx = config.xmax - config.xmin
    Ly = config.ymax - config.ymin

    if Lx != Ly:
        raise ValueError("SWIFT nondiv velocity is only nondivergent for square domains (Lx=Ly).")
    
    coeff = 0.5*Lx - config.u0*(it + 0.5)*config.dt # using one coeff assumes Lx=Ly

    fields.psi = - config.u0*Lx/np.pi*np.sin(np.pi*(fields.xffb + coeff)/Lx)*np.sin(np.pi*(fields.xffb + coeff)/Lx)*np.sin(np.pi*(fields.yffb + coeff)/Ly)*np.sin(np.pi*(fields.yffb + coeff)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) - config.u0*fields.yffb + config.u0*fields.xffb # assumes Lx=Ly

    velocities_from_streamfunction(config, fields)


def swift_nondiv_double_streamfunction(config, fields, it):

    Lx = config.xmax - config.xmin
    Ly = config.ymax - config.ymin

    coeff = 0.5*Lx - config.u0*(it + 0.5)*config.dt # using one coeff assumes Lx=Ly
    
    fields.psi = - 2.*config.u0*Lx/np.pi*np.sin(np.pi*(fields.xffb + coeff)/Lx)*np.sin(np.pi*(fields.xffb + coeff)/Lx)*np.sin(np.pi*(fields.yffb + coeff)/Ly)*np.sin(np.pi*(fields.yffb + coeff)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) - config.u0*fields.yffb + config.u0*fields.xffb # assumes Lx=Ly

    velocities_from_streamfunction(config, fields)


def swift_nondiv_double(config, fields, it):
    # Non-divergent velocity field at half time levels

    Lx = config.xmax - config.xmin
    Ly = config.ymax - config.ymin
    
    ut = config.u0*(it + 0.5)*config.dt

    fields.u = 2.*config.u0*np.sin(np.pi*(fields.xfc + 0.5*Lx - ut)/Lx)*np.sin(np.pi*(fields.xfc + 0.5*Lx - ut)/Lx)*np.sin(2.*np.pi*(fields.yfc + 0.5*Ly - ut)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.u0
    
    fields.v = -2.*config.u0*np.sin(2.*np.pi*(fields.xcf + 0.5*Lx - ut)/Lx)*np.sin(np.pi*(fields.ycf + 0.5*Ly - ut)/Ly)*np.sin(np.pi*(fields.ycf + 0.5*Ly - ut)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.u0


def swift_nondiv_try(config, fields, it):
    # Non-divergent velocity field at half time levels

    Lx = config.xmax - config.xmin
    Ly = config.ymax - config.ymin
    
    ut = config.u0*(it + 0.5)*config.dt

    #fields.u = config.u0*np.sin(np.pi*(fields.xfc + 0.5*Lx - ut)/Lx)**4*np.sin(2.*np.pi*(fields.yfc + 0.5*Ly - ut)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.u0
    
    #fields.v = -config.u0*np.sin(2.*np.pi*(fields.xcf + 0.5*Lx - ut)/Lx)*np.sin(np.pi*(fields.ycf + 0.5*Ly - ut)/Ly)**4*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.u0




    fields.u = 5.*config.u0*np.sin(np.pi*(fields.xfc + 0.5*Lx - ut)/Lx)**4*np.sin(np.pi*(fields.yfc + 0.5*Ly - ut)/Ly)**3*np.cos(np.pi*(fields.yfc + 0.5*Ly - ut)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.u0
    
    fields.v = -5.*config.u0*np.sin(np.pi*(fields.xcf + 0.5*Lx - ut)/Lx)**3*np.cos(np.pi*(fields.xcf + 0.5*Lx - ut)/Lx)*np.sin(np.pi*(fields.ycf + 0.5*Ly - ut)/Ly)**4*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.u0

    #fields.v[:,:] = 0.
    
    ##fields.u = config.u0*np.sin(np.pi*(fields.xfc/Lx + 0.5))*np.sin(np.pi*(fields.xfc/Lx + 0.5))*np.sin(2.*np.pi*(fields.yfc/Ly + 0.5))*np.cos(np.pi*(it+0.5)*config.dt/config.T) + config.u0
    ##fields.v = -config.u0*np.sin(2.*np.pi*(fields.xfc/Lx + 0.5))*np.sin(np.pi*(fields.yfc/Ly + 0.5))*np.sin(np.pi*(fields.yfc/Ly + 0.5))*np.cos(np.pi*(it+0.5)*config.dt/config.T) + config.u0


def solid_body_rotation(config, fields, it): # Chen, Weller et al 2017
    xc = 0.5 * (config.xmax + config.xmin)
    yc = 0.5 * (config.ymax + config.ymin)
    A = 5.*np.pi/3000. # s (angular velocity = 2A)

    fields.psi = A*((fields.xffb - xc)**2 + (fields.yffb - yc)**2)

    velocities_from_streamfunction(config, fields)


def blossey_durran(config, fields, it): # blossey durran 2008
    xc = 0.5 * (config.xmax + config.xmin)
    yc = 0.5 * (config.ymax + config.ymin)

    r = np.sqrt((fields.xffb - xc)**2 + (fields.yffb - yc)**2)
    t = (it + 0.5)*config.dt

    fields.psi = 4.*np.pi/config.T*(0.5*r*r + np.cos(2.*np.pi*t/config.T)*(0.5*r*r + np.log(1. - 16.*r*r + 256.*r*r*r*r)/96. - np.log(1. + 16.*r*r)/48. - np.sqrt(3.)/48.*np.arctan((-1. + 32.*r*r)/np.sqrt(3.))))

    velocities_from_streamfunction(config, fields)


def new_blossey_durran(config, fields, it): # blossey durran 2008
    #xc = 0.5 * (config.xmax + config.xmin)
    #yc = 0.5 * (config.ymax + config.ymin)
#
    #r = np.sqrt((fields.xffb - xc)**2 + (fields.yffb - yc)**2)
    #t = (it + 0.5)*config.dt
#
    #psi = 4.*np.pi/config.T*np.cos(2.*np.pi*0.5*config.T/config.T)*(0.5*r*r + np.log(1. - 16.*r*r + 256.*r*r*r*r)/96. - np.log(1. + 16.*r*r)/48. - np.sqrt(3.)/48.*np.arctan((-1. + 32.*r*r)/np.sqrt(3.)))
#
    #velocities_from_streamfunction(config, fields, psi)
    
    xc = 0.5 * (config.xmax + config.xmin)
    yc = 0.5 * (config.ymax + config.ymin)

    r = np.sqrt((fields.xffb - xc)**2 + (fields.yffb - yc)**2)
    t = (it + 0.5)*config.dt

    fields.psi = np.cos(2.*np.pi*t/config.T)*(-4.*np.pi/config.T*(np.log(1. - 16.*r*r + 256.*r*r*r*r)/96. - np.log(1. + 16.*r*r)/48. - np.sqrt(3.)/48.*np.arctan((-1. + 32.*r*r)/np.sqrt(3.))) - np.pi*np.pi/(8.*np.sqrt(3)*config.T))

    velocities_from_streamfunction(config, fields)


def new_blossey_durran_plusmean(config, fields, it): # blossey durran 2008
    #xc = 0.5 * (config.xmax + config.xmin)
    #yc = 0.5 * (config.ymax + config.ymin)
#
    #r = np.sqrt((fields.xffb - xc)**2 + (fields.yffb - yc)**2)
    #t = (it + 0.5)*config.dt
#
    #psi = 4.*np.pi/config.T*np.cos(2.*np.pi*0.5*config.T/config.T)*(0.5*r*r + np.log(1. - 16.*r*r + 256.*r*r*r*r)/96. - np.log(1. + 16.*r*r)/48. - np.sqrt(3.)/48.*np.arctan((-1. + 32.*r*r)/np.sqrt(3.)))
#
    #velocities_from_streamfunction(config, fields, psi)
    u0 = (config.xmax - config.xmin)/config.T # mean flow velocity

    xc = 0.5 * (config.xmax + config.xmin)
    yc = 0.5 * (config.ymax + config.ymin)

    t = (it + 0.5)*config.dt
    r = np.sqrt((fields.xffb - xc - u0*t)**2 + (fields.yffb - yc - u0*t)**2)

    fields.psi = np.cos(2.*np.pi*t/config.T)*(-4.*np.pi/config.T*(np.log(1. - 16.*r*r + 256.*r*r*r*r)/96. - np.log(1. + 16.*r*r)/48. - np.sqrt(3.)/48.*np.arctan((-1. + 32.*r*r)/np.sqrt(3.))) - np.pi*np.pi/(8.*np.sqrt(3)*config.T)) - u0*fields.yffb + u0*fields.xffb

    velocities_from_streamfunction(config, fields)


def rotationalflow_20251112(config, fields, it):

    L = config.xmax - config.xmin # domain size in x direction (assumed square domain)
    t = (it + 0.5)*config.dt
    xc = 0.5
    yc = 0.5
    xdist = np.minimum((fields.xffb - xc - config.u0*t)%L, -(fields.xffb - xc - config.u0*t)) # could this lead to x and y's that aren't associated to the correct point, that shouldn't be looked at together? !!!
    ydist = np.minimum((fields.yffb - yc - config.u0*t)%L, -(fields.yffb - yc - config.u0*t))
    r = np.sqrt((xdist)**2 + (ydist)**2)
    R = L/4.

    fields.psi = np.where(r < R, 0.5*config.u0*L/np.pi*np.cos(2.*np.pi*r/L)*np.cos(2.*np.pi*r/L), 0.) - config.u0*fields.yffb + config.u0*fields.xffb

    velocities_from_streamfunction(config, fields)


def rotationalflowplusmeaninstrfn_20251112(config, fields, it):

    L = config.xmax - config.xmin # domain size in x direction (assumed square domain)
    t = (it + 0.5)*config.dt
    xc, yc = 0.5, 0.5
    xdist = fields.xffb - xc - config.u0*t
    xdist = np.minimum.reduce([np.abs(xdist), np.abs(xdist%L), np.abs(-xdist), np.abs((-xdist)%L)])
    ydist = fields.yffb - yc - config.u0*t
    ydist = np.minimum.reduce([np.abs(ydist), np.abs(ydist%L), np.abs(-ydist), np.abs((-ydist)%L)])
    r = np.sqrt((xdist)**2 + (ydist)**2)
    R = L/4.

    fields.psi = np.where(r < R, 0.5*config.u0*L/np.pi*np.cos(2.*np.pi*r/L)*np.cos(2.*np.pi*r/L), 0.) - config.u0*fields.yffb + config.u0*fields.xffb
    
    velocities_from_streamfunction(config, fields)


def velocities_from_streamfunction(config, fields):
    """Deriving the u and v velocity components from the streamfunction psi.
    psi[i,j] is defined at i-1/2, j-1/2 -> bottom left cell corner"""

    fields.u = - (fields.psi[:-1,1:] - fields.psi[:-1,:-1])/config.dy
    fields.v = (fields.psi[1:,:-1] - fields.psi[:-1,:-1])/config.dx

    div = (np.roll(fields.u, -1, axis=0) - fields.u)/config.dx + (np.roll(fields.v, -1, axis=1) - fields.v)/config.dy
    print('Max divergence in space =', np.max(np.abs(div)))