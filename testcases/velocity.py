import numpy as np
import matplotlib.pyplot as plt

def velocity(config, fields, it):
    # Define velocity fields at time step it (actually velocities taken at it + 0.5 for second-order accuracy)
    globals()[config.velocity_setting](config, fields, it)


def constant_u(config, fields, it):
    fields.u[it,:,:] = config.constant_u
    fields.v[it,:,:] = 0.


def constant_v(config, fields, it):
    fields.u[it,:,:] = 0.
    fields.v[it,:,:] = config.constant_v


def constant_uv(config, fields, it):
    fields.u[it,:,:] = config.constant_u
    fields.v[it,:,:] = config.constant_v


def swift_nondiv(config, fields, it):
    # Non-divergent velocity field at half time levels

    Lx = config.xmax - config.xmin
    Ly = config.ymax - config.ymin
    ut = config.u0*(it + 0.5)*config.dt
    fields.u[it] = config.u0*np.sin(np.pi*(fields.xfc + 0.5*Lx - ut)/Lx)*np.sin(np.pi*(fields.xfc + 0.5*Lx - ut)/Lx)*np.sin(2.*np.pi*(fields.yfc + 0.5*Ly - ut)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.u0
    fields.v[it] = -config.u0*np.sin(2.*np.pi*(fields.xcf + 0.5*Lx - ut)/Lx)*np.sin(np.pi*(fields.ycf + 0.5*Ly - ut)/Ly)*np.sin(np.pi*(fields.ycf + 0.5*Ly - ut)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.u0


def swift_nondiv_streamfunction(config, fields, it):
    # Non-divergent velocity field at half time levels using the streamfunction

    Lx = config.xmax - config.xmin
    Ly = config.ymax - config.ymin
    if Lx != Ly:
        raise ValueError("SWIFT nondiv velocity is only nondivergent for square domains (Lx=Ly).")
    coeff = 0.5*Lx - config.u0*(it + 0.5)*config.dt # using one coeff assumes Lx=Ly
    fields.psi[it] = - config.u0*Lx/np.pi*np.sin(np.pi*(fields.xffb + coeff)/Lx)*np.sin(np.pi*(fields.xffb + coeff)/Lx)*np.sin(np.pi*(fields.yffb + coeff)/Ly)*np.sin(np.pi*(fields.yffb + coeff)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) - config.u0*fields.yffb + config.u0*fields.xffb # assumes Lx=Ly
    velocities_from_streamfunction(config, fields, it)


def swift_nondiv_double_streamfunction(config, fields, it):
    Lx = config.xmax - config.xmin
    Ly = config.ymax - config.ymin
    coeff = 0.5*Lx - config.u0*(it + 0.5)*config.dt # using one coeff assumes Lx=Ly
    fields.psi[it] = - 2.*config.u0*Lx/np.pi*np.sin(np.pi*(fields.xffb + coeff)/Lx)*np.sin(np.pi*(fields.xffb + coeff)/Lx)*np.sin(np.pi*(fields.yffb + coeff)/Ly)*np.sin(np.pi*(fields.yffb + coeff)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) - config.u0*fields.yffb + config.u0*fields.xffb # assumes Lx=Ly
    velocities_from_streamfunction(config, fields, it)


def swift_nondiv_double(config, fields, it):
    # Non-divergent velocity field at half time levels

    Lx = config.xmax - config.xmin
    Ly = config.ymax - config.ymin
    ut = config.u0*(it + 0.5)*config.dt
    fields.u[it] = 2.*config.u0*np.sin(np.pi*(fields.xfc + 0.5*Lx - ut)/Lx)*np.sin(np.pi*(fields.xfc + 0.5*Lx - ut)/Lx)*np.sin(2.*np.pi*(fields.yfc + 0.5*Ly - ut)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.u0
    fields.v[it] = -2.*config.u0*np.sin(2.*np.pi*(fields.xcf + 0.5*Lx - ut)/Lx)*np.sin(np.pi*(fields.ycf + 0.5*Ly - ut)/Ly)*np.sin(np.pi*(fields.ycf + 0.5*Ly - ut)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.u0


def swift_nondiv_try(config, fields, it):
    # Non-divergent velocity field at half time levels

    Lx = config.xmax - config.xmin
    Ly = config.ymax - config.ymin
    ut = config.u0*(it + 0.5)*config.dt

    fields.u[it] = 5.*config.u0*np.sin(np.pi*(fields.xfc + 0.5*Lx - ut)/Lx)**4*np.sin(np.pi*(fields.yfc + 0.5*Ly - ut)/Ly)**3*np.cos(np.pi*(fields.yfc + 0.5*Ly - ut)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.u0
    fields.v[it] = -5.*config.u0*np.sin(np.pi*(fields.xcf + 0.5*Lx - ut)/Lx)**3*np.cos(np.pi*(fields.xcf + 0.5*Lx - ut)/Lx)*np.sin(np.pi*(fields.ycf + 0.5*Ly - ut)/Ly)**4*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.u0


def solid_body_rotation(config, fields, it): # Chen, Weller et al 2017
    xc = 0.5 * (config.xmax + config.xmin)
    yc = 0.5 * (config.ymax + config.ymin)
    A = 5.*np.pi/3000. # s (angular velocity = 2A)
    fields.psi[it] = A*((fields.xffb - xc)**2 + (fields.yffb - yc)**2)
    velocities_from_streamfunction(config, fields, it)


def blossey_durran(config, fields, it): # blossey durran 2008
    xc = 0.5 * (config.xmax + config.xmin)
    yc = 0.5 * (config.ymax + config.ymin)
    r = np.sqrt((fields.xffb - xc)**2 + (fields.yffb - yc)**2)
    t = (it + 0.5)*config.dt
    fields.psi[it] = 4.*np.pi/config.T*(0.5*r*r + np.cos(2.*np.pi*t/config.T)*(0.5*r*r + np.log(1. - 16.*r*r + 256.*r*r*r*r)/96. - np.log(1. + 16.*r*r)/48. - np.sqrt(3.)/48.*np.arctan((-1. + 32.*r*r)/np.sqrt(3.))))
    velocities_from_streamfunction(config, fields, it)


def new_blossey_durran(config, fields, it): # blossey durran 2008
    xc = 0.5 * (config.xmax + config.xmin)
    yc = 0.5 * (config.ymax + config.ymin)
    r = np.sqrt((fields.xffb - xc)**2 + (fields.yffb - yc)**2)
    t = (it + 0.5)*config.dt
    fields.psi[it] = np.cos(2.*np.pi*t/config.T)*(-4.*np.pi/config.T*(np.log(1. - 16.*r*r + 256.*r*r*r*r)/96. - np.log(1. + 16.*r*r)/48. - np.sqrt(3.)/48.*np.arctan((-1. + 32.*r*r)/np.sqrt(3.))) - np.pi*np.pi/(8.*np.sqrt(3)*config.T))
    velocities_from_streamfunction(config, fields, it)


def new_blossey_durran_plusmean(config, fields, it): # blossey durran 2008
    u0 = (config.xmax - config.xmin)/config.T # mean flow velocity
    xc = 0.5 * (config.xmax + config.xmin)
    yc = 0.5 * (config.ymax + config.ymin)
    t = (it + 0.5)*config.dt
    r = np.sqrt((fields.xffb - xc - u0*t)**2 + (fields.yffb - yc - u0*t)**2)
    fields.psi[it] = np.cos(2.*np.pi*t/config.T)*(-4.*np.pi/config.T*(np.log(1. - 16.*r*r + 256.*r*r*r*r)/96. - np.log(1. + 16.*r*r)/48. - np.sqrt(3.)/48.*np.arctan((-1. + 32.*r*r)/np.sqrt(3.))) - np.pi*np.pi/(8.*np.sqrt(3)*config.T)) - u0*fields.yffb + u0*fields.xffb
    velocities_from_streamfunction(config, fields, it)


def rotationalflow_20251112(config, fields, it):
    L = config.xmax - config.xmin # domain size in x direction (assumed square domain)
    t = (it + 0.5)*config.dt
    xc = 0.5
    yc = 0.5
    xdist = np.minimum((fields.xffb - xc - config.u0*t)%L, -(fields.xffb - xc - config.u0*t)) # could this lead to x and y's that aren't associated to the correct point, that shouldn't be looked at together? !!!
    ydist = np.minimum((fields.yffb - yc - config.u0*t)%L, -(fields.yffb - yc - config.u0*t))
    r = np.sqrt((xdist)**2 + (ydist)**2)
    R = L/4.
    fields.psi[it] = np.where(r < R, 0.5*config.u0*L/np.pi*np.cos(2.*np.pi*r/L)*np.cos(2.*np.pi*r/L), 0.) - config.u0*fields.yffb + config.u0*fields.xffb
    velocities_from_streamfunction(config, fields, it)


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
    fields.psi[it] = np.where(r < R, 0.5*config.u0*L/np.pi*np.cos(2.*np.pi*r/L)*np.cos(2.*np.pi*r/L), 0.) - config.u0*fields.yffb + config.u0*fields.xffb
    velocities_from_streamfunction(config, fields, it)


def ujet(config, fields, it):
    """Zonal jet to create a zonal velocity field of:
    u = 0 when y < a
    u = 0.5(1-cos(2pi(y-a)/(b-a))) when a <= y <= b
    u = 0 when y > b"""

    a = config.ujet_a
    b = config.ujet_b

    for j in range(config.ny+1):
        y = fields.yffb[0,j]
        if y <= a:
            fields.psi[it,:,j] = 0.
        elif a < y < b:
            fields.psi[it,:,j] = 0.5*config.ujet_max*(0.5*(b-a)/np.pi*np.sin(2.*np.pi*(y-a)/(b-a)) + a - y)
        else: # y >= b
            fields.psi[it,:,j] = 0.5*config.ujet_max*(a-b)
    
    velocities_from_streamfunction(config, fields, it)


def ujet_reversal(config, fields, it):
    """Zonal jet to create a zonal velocity field of:
    u = 0 when y < a
    u = 0.5(1-cos(2pi(y-a)/(b-a))) when a <= y <= b
    u = 0 when y > b"""

    a = config.ujet_a
    b = config.ujet_b

    for j in range(config.ny+1):
        y = fields.yffb[0,j]
        if y <= a:
            fields.psi[it,:,j] = 0.
        elif a < y < b:
            fields.psi[it,:,j] = 0.5*config.ujet_max*(0.5*(b-a)/np.pi*np.sin(2.*np.pi*(y-a)/(b-a)) + a - y)*np.cos(np.pi*(it + 0.5)*config.dt/config.T)
        else: # y >= b
            fields.psi[it,:,j] = 0.5*config.ujet_max*(a-b)*np.cos(np.pi*(it + 0.5)*config.dt/config.T)
    
    velocities_from_streamfunction(config, fields, it)


def ujet_45deg_reversal(config, fields, it):
    """Zonal jet to create a zonal velocity field of:
    u = 0 when y < a
    u = 0.5(1-cos(2pi(y-a)/(b-a))) when a <= y <= b
    u = 0 when y > b"""

    a = config.ujet_a - config.ymin
    b = config.ujet_b - config.ymin

    L = config.ymax - config.ymin
    for j in range(config.ny+1):
        y = fields.yffb[0,j] - config.ymin
        for i in range(config.nx+1):
            x = fields.xffb[i,0] -  config.xmin
            ya = a + x
            yb = b + x
            if ya > L: ya = ya%L
            if yb > L: yb = yb%L

            if y <= ya <= a and x > 0.:
                fields.psi[it,i,j] = -0.5*config.ujet_max*(a - b)*np.cos(np.pi*(it + 0.5)*config.dt/config.T)
            elif (ya < y < yb and ya <= a and yb <= b and x > 0.) or (yb < ya and y < yb):
                fields.psi[it,i,j] = 0.5*config.ujet_max*(0.5*(b - a)/np.pi*np.sin(2.*np.pi*(y - a - x)/(b - a)) + a + x - y)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.ujet_max*(a - b)*np.cos(np.pi*(it + 0.5)*config.dt/config.T)
            elif (y >= yb and a < yb <= b and x > 0.) or (yb <= y <= ya) or (y <= ya and ya >= a and x < L):
                fields.psi[it,i,j] = 0.
            elif (ya < y < yb and x < L and ya >= a and yb >= b) or (yb < ya < y):
                fields.psi[it,i,j] = 0.5*config.ujet_max*(0.5*(b - a)/np.pi*np.sin(2.*np.pi*(y - x - a)/(b - a)) + a - y + x)*np.cos(np.pi*(it + 0.5)*config.dt/config.T)
            elif y >= yb >= b and x < L:
                fields.psi[it,i,j] = 0.5*config.ujet_max*(a - b)*np.cos(np.pi*(it + 0.5)*config.dt/config.T)
            else:
                print('I dont have this case covered:', i, j, x, y, ya, yb)
    
    velocities_from_streamfunction(config, fields, it)


def swift_ujet45rev(config, fields, it):
    """Zonal jet to create a zonal velocity field of:
    u = 0 when y < a
    u = 0.5(1-cos(2pi(y-a)/(b-a))) when a <= y <= b
    u = 0 when y > b"""

    # Calculate SWIFT streamfunction 
    Lx = config.xmax - config.xmin
    Ly = config.ymax - config.ymin
    if Lx != Ly:
        raise ValueError("SWIFT nondiv velocity is only nondivergent for square domains (Lx=Ly).")
    coeff = 0.5*Lx - config.u0*(it + 0.5)*config.dt # using one coeff assumes Lx=Ly
    fields.psi[it] = - config.u0*Lx/np.pi*np.sin(np.pi*(fields.xffb + coeff)/Lx)*np.sin(np.pi*(fields.xffb + coeff)/Lx)*np.sin(np.pi*(fields.yffb + coeff)/Ly)*np.sin(np.pi*(fields.yffb + coeff)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) - config.u0*fields.yffb + config.u0*fields.xffb # assumes Lx=Ly

    # Calculate jet streamfunction
    jetpsi = np.zeros((config.nx+1, config.ny+1))

    a = config.ujet_a - config.ymin
    b = config.ujet_b - config.ymin

    L = config.ymax - config.ymin
    for j in range(config.ny+1):
        y = fields.yffb[0,j] - config.ymin
        for i in range(config.nx+1):
            x = fields.xffb[i,0] -  config.xmin
            ya = a + x
            yb = b + x
            if ya > L: ya = ya%L
            if yb > L: yb = yb%L

            if y <= ya <= a and x > 0.:
                jetpsi[i,j] = -0.5*config.ujet_max*(a - b)*np.cos(np.pi*(it + 0.5)*config.dt/config.T)
            elif (ya < y < yb and ya <= a and yb <= b and x > 0.) or (yb < ya and y < yb):
                jetpsi[i,j] = 0.5*config.ujet_max*(0.5*(b - a)/np.pi*np.sin(2.*np.pi*(y - a - x)/(b - a)) + a + x - y)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.ujet_max*(a - b)*np.cos(np.pi*(it + 0.5)*config.dt/config.T)
            elif (y >= yb and a < yb <= b and x > 0.) or (yb <= y <= ya) or (y <= ya and ya >= a and x < L):
                jetpsi[i,j] = 0.
            elif (ya < y < yb and x < L and ya >= a and yb >= b) or (yb < ya < y):
                jetpsi[i,j] = 0.5*config.ujet_max*(0.5*(b - a)/np.pi*np.sin(2.*np.pi*(y - x - a)/(b - a)) + a - y + x)*np.cos(np.pi*(it + 0.5)*config.dt/config.T)
            elif y >= yb >= b and x < L:
                jetpsi[i,j] = 0.5*config.ujet_max*(a - b)*np.cos(np.pi*(it + 0.5)*config.dt/config.T)
            else:
                print('I dont have this case covered:', i, j, x, y, ya, yb)
    
    # Add SWIFT and jet together
    fields.psi[it] = fields.psi[it] + jetpsi
    
    velocities_from_streamfunction(config, fields, it)


def velocities_from_streamfunction(config, fields, it):
    """Deriving the u and v velocity components from the streamfunction psi.
    psi[i,j] is defined at i-1/2, j-1/2 -> bottom left cell corner"""

    fields.u[it] = - (fields.psi[it][:-1,1:] - fields.psi[it][:-1,:-1])/config.dy
    fields.v[it] = (fields.psi[it][1:,:-1] - fields.psi[it][:-1,:-1])/config.dx