import numpy as np

def velocity(config, fields, it):
    # Define velocity fields at time step it (actually velocities taken at it + 0.5 for second-order accuracy)
    globals()[config.velocity_setting](config, fields, it)


def constant_u(config, fields, it):
    fields.u[config.store_all_timesteps*it,:,:] = config.constant_u
    fields.v[config.store_all_timesteps*it,:,:] = 0.


def constant_v(config, fields, it):
    fields.u[config.store_all_timesteps*it,:,:] = 0.
    fields.v[config.store_all_timesteps*it,:,:] = config.constant_v


def constant_uv(config, fields, it):
    fields.u[config.store_all_timesteps*it,:,:] = config.constant_u
    fields.v[config.store_all_timesteps*it,:,:] = config.constant_v


def swift_div(config, fields, it):
    # Divergent velocity field at half time levels from SWIFT test case

    Lx = config.xmax - config.xmin
    Ly = config.ymax - config.ymin
    ut = config.u0*(it + 0.5)*config.dt
    fields.u[config.store_all_timesteps*it,:,:] = 0.5*config.u0*np.sin(np.pi*(fields.xfc + 0.5*Lx - ut)/Lx)*np.sin(np.pi*(fields.xfc + 0.5*Lx - ut)/Lx)*np.sin(2.*np.pi*(fields.yfc + 0.5*Ly - ut)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.u0
    fields.v[config.store_all_timesteps*it,:,:] = 0.5*config.u0*np.sin(2.*np.pi*(fields.xcf + 0.5*Lx - ut)/Lx)*np.sin(np.pi*(fields.ycf + 0.5*Ly - ut)/Ly)*np.sin(np.pi*(fields.ycf + 0.5*Ly - ut)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.u0


#def swift_div_on_steroids(config, fields, it):
#    # Divergent velocity field at half time levels from SWIFT test case
#
#    Lx = config.xmax - config.xmin
#    Ly = config.ymax - config.ymin
#    ut = config.u0*(it + 0.5)*config.dt
#    fields.u[config.store_all_timesteps*it,:,:] = 5.*config.u0*np.sin(np.pi*(fields.xfc + 0.5*Lx - ut)/Lx)*np.sin(np.pi*(fields.xfc + 0.5*Lx - ut)/Lx)*np.sin(2.*np.pi*(fields.yfc + 0.5*Ly - ut)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.u0
#    fields.v[config.store_all_timesteps*it,:,:] = 5.*config.u0*np.sin(2.*np.pi*(fields.xcf + 0.5*Lx - ut)/Lx)*np.sin(np.pi*(fields.ycf + 0.5*Ly - ut)/Ly)*np.sin(np.pi*(fields.ycf + 0.5*Ly - ut)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.u0
#
#
def strong_convergence(config, fields, it):
    Lx = config.xmax - config.xmin
    Ly = config.ymax - config.ymin
    fields.u[config.store_all_timesteps*it,:,:] = 10.*config.u0*np.sin(np.pi*(fields.xfc/Lx - 0.5))**4
    fields.v[config.store_all_timesteps*it,:,:] = 10.*config.u0*np.sin(np.pi*(fields.ycf/Ly - 0.5))**4

    #ut = config.u0*(it + 0.5)*config.dt
    #fields.u[config.store_all_timesteps*it,:,:] = config.u0*np.sin(np.pi*(fields.xfc + 0.5*Lx - ut)/Lx)*np.sin(np.pi*(fields.xfc + 0.5*Lx - ut)/Lx)*np.sin(2.*np.pi*(fields.yfc + 0.5*Ly - ut)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.u0


def swift_nondiv(config, fields, it):
    # Non-divergent velocity field at half time levels from SWIFT test case

    Lx = config.xmax - config.xmin
    Ly = config.ymax - config.ymin
    ut = config.u0*(it + 0.5)*config.dt
    fields.u[config.store_all_timesteps*it,:,:] = config.u0*np.sin(np.pi*(fields.xfc + 0.5*Lx - ut)/Lx)*np.sin(np.pi*(fields.xfc + 0.5*Lx - ut)/Lx)*np.sin(2.*np.pi*(fields.yfc + 0.5*Ly - ut)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.u0
    fields.v[config.store_all_timesteps*it,:,:] = -config.u0*np.sin(2.*np.pi*(fields.xcf + 0.5*Lx - ut)/Lx)*np.sin(np.pi*(fields.ycf + 0.5*Ly - ut)/Ly)*np.sin(np.pi*(fields.ycf + 0.5*Ly - ut)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.u0


def swift_nondiv_streamfunction(config, fields, it):
    # Non-divergent velocity field at half time levels from SWIFT testcase using the streamfunction

    Lx = config.xmax - config.xmin
    Ly = config.ymax - config.ymin
    if Lx != Ly:
        raise ValueError("SWIFT nondiv velocity is only nondivergent for square domains (Lx=Ly).")
    coeff = 0.5*Lx - config.u0*(it + 0.5)*config.dt # using one coeff assumes Lx=Ly
    fields.psi[config.store_all_timesteps*it,:,:] = - config.u0*Lx/np.pi*np.sin(np.pi*(fields.xffb + coeff)/Lx)*np.sin(np.pi*(fields.xffb + coeff)/Lx)*np.sin(np.pi*(fields.yffb + coeff)/Ly)*np.sin(np.pi*(fields.yffb + coeff)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) - config.u0*fields.yffb + config.u0*fields.xffb # assumes Lx=Ly
    velocities_from_streamfunction(config, fields, it)


def swift_nondiv_double_streamfunction(config, fields, it):
    Lx = config.xmax - config.xmin
    Ly = config.ymax - config.ymin
    coeff = 0.5*Lx - config.u0*(it + 0.5)*config.dt # using one coeff assumes Lx=Ly
    fields.psi[config.store_all_timesteps*it,:,:] = - 2.*config.u0*Lx/np.pi*np.sin(np.pi*(fields.xffb + coeff)/Lx)*np.sin(np.pi*(fields.xffb + coeff)/Lx)*np.sin(np.pi*(fields.yffb + coeff)/Ly)*np.sin(np.pi*(fields.yffb + coeff)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) - config.u0*fields.yffb + config.u0*fields.xffb # assumes Lx=Ly
    velocities_from_streamfunction(config, fields, it)


def swift_nondiv_double(config, fields, it):
    # Non-divergent velocity field at half time levels

    Lx = config.xmax - config.xmin
    Ly = config.ymax - config.ymin
    ut = config.u0*(it + 0.5)*config.dt
    fields.u[config.store_all_timesteps*it,:,:] = 2.*config.u0*np.sin(np.pi*(fields.xfc + 0.5*Lx - ut)/Lx)*np.sin(np.pi*(fields.xfc + 0.5*Lx - ut)/Lx)*np.sin(2.*np.pi*(fields.yfc + 0.5*Ly - ut)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.u0
    fields.v[config.store_all_timesteps*it,:,:] = -2.*config.u0*np.sin(2.*np.pi*(fields.xcf + 0.5*Lx - ut)/Lx)*np.sin(np.pi*(fields.ycf + 0.5*Ly - ut)/Ly)*np.sin(np.pi*(fields.ycf + 0.5*Ly - ut)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.u0


def swift_nondiv_try(config, fields, it):
    # Non-divergent velocity field at half time levels

    Lx = config.xmax - config.xmin
    Ly = config.ymax - config.ymin
    ut = config.u0*(it + 0.5)*config.dt

    fields.u[config.store_all_timesteps*it,:,:] = 5.*config.u0*np.sin(np.pi*(fields.xfc + 0.5*Lx - ut)/Lx)**4*np.sin(np.pi*(fields.yfc + 0.5*Ly - ut)/Ly)**3*np.cos(np.pi*(fields.yfc + 0.5*Ly - ut)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.u0
    fields.v[config.store_all_timesteps*it,:,:] = -5.*config.u0*np.sin(np.pi*(fields.xcf + 0.5*Lx - ut)/Lx)**3*np.cos(np.pi*(fields.xcf + 0.5*Lx - ut)/Lx)*np.sin(np.pi*(fields.ycf + 0.5*Ly - ut)/Ly)**4*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.u0


def solid_body_rotation(config, fields, it): # Chen, Weller et al 2017
    xc = 0.5 * (config.xmax + config.xmin)
    yc = 0.5 * (config.ymax + config.ymin)
    A = 5.*np.pi/3000. # s (angular velocity = 2A)
    fields.psi[config.store_all_timesteps*it,:,:] = A*((fields.xffb - xc)**2 + (fields.yffb - yc)**2)
    velocities_from_streamfunction(config, fields, it)


def blossey_durran(config, fields, it): # blossey durran 2008
    xc = 0.5 * (config.xmax + config.xmin)
    yc = 0.5 * (config.ymax + config.ymin)
    r = np.sqrt((fields.xffb - xc)**2 + (fields.yffb - yc)**2)
    t = (it + 0.5)*config.dt
    fields.psi[config.store_all_timesteps*it,:,:] = 4.*np.pi/config.T*(0.5*r*r + np.cos(2.*np.pi*t/config.T)*(0.5*r*r + np.log(1. - 16.*r*r + 256.*r*r*r*r)/96. - np.log(1. + 16.*r*r)/48. - np.sqrt(3.)/48.*np.arctan((-1. + 32.*r*r)/np.sqrt(3.))))
    velocities_from_streamfunction(config, fields, it)


def new_blossey_durran(config, fields, it): # blossey durran 2008
    xc = 0.5 * (config.xmax + config.xmin)
    yc = 0.5 * (config.ymax + config.ymin)
    r = np.sqrt((fields.xffb - xc)**2 + (fields.yffb - yc)**2)
    t = (it + 0.5)*config.dt
    fields.psi[config.store_all_timesteps*it,:,:] = np.cos(2.*np.pi*t/config.T)*(-4.*np.pi/config.T*(np.log(1. - 16.*r*r + 256.*r*r*r*r)/96. - np.log(1. + 16.*r*r)/48. - np.sqrt(3.)/48.*np.arctan((-1. + 32.*r*r)/np.sqrt(3.))) - np.pi*np.pi/(8.*np.sqrt(3)*config.T))
    velocities_from_streamfunction(config, fields, it)


def new_blossey_durran_plusmean(config, fields, it): # blossey durran 2008
    u0 = (config.xmax - config.xmin)/config.T # mean flow velocity
    xc = 0.5 * (config.xmax + config.xmin)
    yc = 0.5 * (config.ymax + config.ymin)
    t = (it + 0.5)*config.dt
    r = np.sqrt((fields.xffb - xc - u0*t)**2 + (fields.yffb - yc - u0*t)**2)
    fields.psi[config.store_all_timesteps*it,:,:] = np.cos(2.*np.pi*t/config.T)*(-4.*np.pi/config.T*(np.log(1. - 16.*r*r + 256.*r*r*r*r)/96. - np.log(1. + 16.*r*r)/48. - np.sqrt(3.)/48.*np.arctan((-1. + 32.*r*r)/np.sqrt(3.))) - np.pi*np.pi/(8.*np.sqrt(3)*config.T)) - u0*fields.yffb + u0*fields.xffb
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
    fields.psi[config.store_all_timesteps*it,:,:] = np.where(r < R, 0.5*config.u0*L/np.pi*np.cos(2.*np.pi*r/L)*np.cos(2.*np.pi*r/L), 0.) - config.u0*fields.yffb + config.u0*fields.xffb
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
    fields.psi[config.store_all_timesteps*it,:,:] = np.where(r < R, 0.5*config.u0*L/np.pi*np.cos(2.*np.pi*r/L)*np.cos(2.*np.pi*r/L), 0.) - config.u0*fields.yffb + config.u0*fields.xffb
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
            fields.psi[config.store_all_timesteps*it,:,j] = 0.
        elif a < y < b:
            fields.psi[config.store_all_timesteps*it,:,j] = 0.5*config.ujet_max*(0.5*(b-a)/np.pi*np.sin(2.*np.pi*(y-a)/(b-a)) + a - y)
        else: # y >= b
            fields.psi[config.store_all_timesteps*it,:,j] = 0.5*config.ujet_max*(a-b)
    
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
            fields.psi[config.store_all_timesteps*it,:,j] = 0.
        elif a < y < b:
            fields.psi[config.store_all_timesteps*it,:,j] = 0.5*config.ujet_max*(0.5*(b-a)/np.pi*np.sin(2.*np.pi*(y-a)/(b-a)) + a - y)*np.cos(np.pi*(it + 0.5)*config.dt/config.T)
        else: # y >= b
            fields.psi[config.store_all_timesteps*it,:,j] = 0.5*config.ujet_max*(a-b)*np.cos(np.pi*(it + 0.5)*config.dt/config.T)
    
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
                fields.psi[config.store_all_timesteps*it,i,j] = -0.5*config.ujet_max*(a - b)*np.cos(np.pi*(it + 0.5)*config.dt/config.T)
            elif (ya < y < yb and ya <= a and yb <= b and x > 0.) or (yb < ya and y < yb):
                fields.psi[config.store_all_timesteps*it,i,j] = 0.5*config.ujet_max*(0.5*(b - a)/np.pi*np.sin(2.*np.pi*(y - a - x)/(b - a)) + a + x - y)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.ujet_max*(a - b)*np.cos(np.pi*(it + 0.5)*config.dt/config.T)
            elif (y >= yb and a < yb <= b and x > 0.) or (yb <= y <= ya) or (y <= ya and ya >= a and x < L):
                fields.psi[config.store_all_timesteps*it,i,j] = 0.
            elif (ya < y < yb and x < L and ya >= a and yb >= b) or (yb < ya < y):
                fields.psi[config.store_all_timesteps*it,i,j] = 0.5*config.ujet_max*(0.5*(b - a)/np.pi*np.sin(2.*np.pi*(y - x - a)/(b - a)) + a - y + x)*np.cos(np.pi*(it + 0.5)*config.dt/config.T)
            elif y >= yb >= b and x < L:
                fields.psi[config.store_all_timesteps*it,i,j] = 0.5*config.ujet_max*(a - b)*np.cos(np.pi*(it + 0.5)*config.dt/config.T)
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
    fields.psi[config.store_all_timesteps*it] = - config.u0*Lx/np.pi*np.sin(np.pi*(fields.xffb + coeff)/Lx)*np.sin(np.pi*(fields.xffb + coeff)/Lx)*np.sin(np.pi*(fields.yffb + coeff)/Ly)*np.sin(np.pi*(fields.yffb + coeff)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) - config.u0*fields.yffb + config.u0*fields.xffb # assumes Lx=Ly

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
    fields.psi[config.store_all_timesteps*it] = fields.psi[config.store_all_timesteps*it] + jetpsi
    
    velocities_from_streamfunction(config, fields, it)


def velocities_from_streamfunction(config, fields, it):
    """Deriving the u and v velocity components from the streamfunction psi.
    psi[i,j] is defined at i-1/2, j-1/2 -> bottom left cell corner, with psi of size (nx+1, ny+1) including the right
    and top boundary corners. u[i,j] is defined at [i-1/2,j], v[i,j] at [i,j-1/2].

    The schemes store nx faces in x (ny in y) and take the right (top) boundary face from the left (bottom) one via
    np.roll, i.e. they reuse u[0,:] and v[:,0] there. The flux divergence of each cell is only zero to rounding error
    (needed for constancy) if the psi values on the boundary are consistent to the last bit:
      - periodic: psi at xmax (index -1) is the periodic image of psi at xmin (index 0), up to the mean flow;
      - noflux:   psi is one constant along the walls, so that the wall-normal velocity is exactly zero.
    The testcase psi only satisfies this up to rounding (about 1e-16*max|psi|, i.e. ~1e-12 for SWIFT where |psi| ~ 1e4),
    which gave O(1e-13) 1/s divergence in the last column and row. This function enforces it exactly.
    fields.psi itself is left unchanged."""

    k = config.store_all_timesteps*it
    psi = fields.psi[k]
    Lx, Ly = config.xmax - config.xmin, config.ymax - config.ymin
    X, Y = fields.xffb - config.xmin, fields.yffb - config.ymin # corner coordinates relative to the lower left corner

    # 1. Mean velocities U and V, i.e. the net flow through the domain, which is only possible across periodic boundaries.
    #    The volume flux through the left boundary is -(psi[0,-1] - psi[0,0]) = U*Ly and through the bottom boundary
    #    psi[-1,0] - psi[0,0] = V*Lx, so that psi = psi_p + V*X - U*Y with psi_p periodic.
    U = -(psi[0,-1] - psi[0,0])/Ly if config.BC_x == 'periodic' else 0.
    V =  (psi[-1,0] - psi[0,0])/Lx if config.BC_y == 'periodic' else 0.

    # 2. Remove the linear (mean flow) part, which is added back as U and V in step 4 (so the velocities don't change).
    #    After this, the two copies of a periodic seam are (analytically) equal instead of offset by V*Lx or U*Ly, so step 3
    #    can copy them. Copying with an offset instead would round each value separately and give the mismatch again.
    #    This step does not remove the rounding errors already in psi; step 3 does that. psi_p is a new array.
    psi_p = psi - (V*X - U*Y)

    # 3. Check that the boundary values are analytically consistent (up to tol), then make them consistent to the last bit.
    #    A mismatch larger than tol means the testcase psi doesn't fit the BCs (e.g. flow through a noflux wall), which is
    #    an error in the setup that we don't want to hide by overwriting. Rounding mismatches are ~1e-16*max|psi|.
    #    (tol uses max|psi|, not max|psi_p|, as psi_p is ~0 for a pure mean flow.)
    tol = 1e-10*np.abs(psi).max()
    c = psi_p[0,0] # psi_p on all walls: with noflux in both directions the boundary is one streamline; with noflux in
                   # one direction both walls have the same psi_p, as the mean flow along the walls has been removed.
    if config.BC_x == 'periodic':
        mismatch = np.abs(psi_p[-1,:] - psi_p[0,:]).max()
        if mismatch > tol:
            raise ValueError(f'psi (minus the mean flow) is not periodic in x: mismatch {mismatch:.1e} > tolerance {tol:.1e}.')
        psi_p[-1,:] = psi_p[0,:] # corners at xmax become the same floats as at xmin (only enters v[-1,:])
    elif config.BC_x == 'noflux':
        mismatch = max(np.abs(psi_p[0,:] - c).max(), np.abs(psi_p[-1,:] - c).max())
        if mismatch > tol:
            raise ValueError(f'psi is not constant along the x walls (flow through a noflux boundary): mismatch {mismatch:.1e} > tolerance {tol:.1e}.')
        psi_p[0,:] = c  # left wall: u[0,:] = 0 exactly, also used (via np.roll) as the flux through the right wall
        psi_p[-1,:] = c # right wall: consistent with the zero flux through it (only enters v[-1,:])
    else:
        raise ValueError(f'Unknown BC_x: {config.BC_x}')
    if config.BC_y == 'periodic':
        mismatch = np.abs(psi_p[:,-1] - psi_p[:,0]).max()
        if mismatch > tol:
            raise ValueError(f'psi (minus the mean flow) is not periodic in y: mismatch {mismatch:.1e} > tolerance {tol:.1e}.')
        psi_p[:,-1] = psi_p[:,0] # corners at ymax become the same floats as at ymin (only enters u[:,-1])
    elif config.BC_y == 'noflux':
        mismatch = max(np.abs(psi_p[:,0] - c).max(), np.abs(psi_p[:,-1] - c).max())
        if mismatch > tol:
            raise ValueError(f'psi is not constant along the y walls (flow through a noflux boundary): mismatch {mismatch:.1e} > tolerance {tol:.1e}.')
        psi_p[:,0] = c  # bottom wall: v[:,0] = 0 exactly, also used (via np.roll) as the flux through the top wall
        psi_p[:,-1] = c # top wall: consistent with the zero flux through it (only enters u[:,-1])
    else:
        raise ValueError(f'Unknown BC_y: {config.BC_y}')

    # 4. Velocities from differences of psi_p, plus the mean flow. Every cell now uses the same corner floats as its
    #    neighbours (also across the boundary), so the face fluxes of each cell cancel to rounding error.
    fields.u[k] = - (psi_p[:-1,1:] - psi_p[:-1,:-1])/fields.dycc + U # at [i-1/2,j]
    fields.v[k] =   (psi_p[1:,:-1] - psi_p[:-1,:-1])/fields.dxcc + V # at [i,j-1/2]


def double_rotation(config, fields, it):
    Lx = config.xmax - config.xmin # domain size in x direction (assumed square domain)
    Ly = config.ymax - config.ymin

    fields.psi[config.store_all_timesteps*it] = -10000.*np.sin(2.*np.pi*(fields.xffb - config.xmin)/Lx)*np.sin(np.pi*(fields.yffb - config.ymin)/Ly)
    velocities_from_streamfunction(config, fields, it)


def double_rotation_rising_bubble_capped(config, fields, it):
    Lx = config.xmax - config.xmin # domain size in x direction (assumed square domain)
    Ly = config.ymax - config.ymin

    fields.psi[config.store_all_timesteps*it] = -8000.*np.sin(2.*np.pi*(fields.xffb - config.xmin)/Lx)*np.sin(1.25*np.pi*(fields.yffb - config.ymin)/Ly)
    fields.psi[config.store_all_timesteps*it] = np.where(fields.yffb < 0.8*(config.ymax - config.ymin) + config.ymin, fields.psi[config.store_all_timesteps*it], 0.)
    #plt.contourf(fields.xffb, fields.yffb, fields.psi[config.store_all_timesteps*it])
    #plt.show()
    #exit()
    velocities_from_streamfunction(config, fields, it)


def sech(x):
    return 2.*np.cosh(x)/(np.cosh(2.*x) + 1.)


def doswell(config, fields, it):

    Lx = config.xmax - config.xmin
    Ly = config.ymax - config.ymin

    xc = 0.5 * (config.xmax + config.xmin)
    yc = 0.5 * (config.ymax + config.ymin)

    alpha = np.arctan((fields.yffb/Ly) / (fields.xffb/Lx +1.e-16))
    r = np.sqrt((fields.xffb - xc)**2 + (fields.yffb - yc)**2)
    Vt = sech(r*np.pi/(0.5*Lx))**2*np.tanh(r*np.pi/(0.5*Lx))#*Lx # assumes Lx=Ly
    alpha = np.arctan2(fields.yffb, fields.xffb)# + np.pi # angle from x-axis, radians
    fields.psi[config.store_all_timesteps*it] = -Lx*0.5*sech(r*np.pi/(0.5*Lx))**2 # assumes Lx=Ly
    
    velocities_from_streamfunction(config, fields, it)


def hadley(config, fields, it):
    # Kent et al 2014: using v and w in eqs 37 and 38
    # fields.xffb is latitude (phi)
    # fields.yffb is height (z)
    # fields.u is v, meridional wind
    # fields.v is w, vertical wind
    # default settings to be used with this: xmin = -90, xmax = 90 (degrees), ymin = 0, ymax = ztop = 1.2e4 (m), nx = 90 or 180 or 360 (corresponds to 220, 110, 55 km grid spacing), ny = 30 or 60 or 120 (uniform spacing, corresponds to 400, 200, 100 m grid spacing). 
    # run the simulation for 1 day, i.e., 86400s ... for time step ???

    # 03-03-2026: This is on the sphere, and I think this description when used on the plane is divergent.

    a = 6.37122e6 # m Earth radius
    w0 = 0.15 # ms-1 reference vertical velocity
    K = 5 # number of overturning cells
    ztop = 1.2e4 # m height position of model top 
    p0 = 1.e5 # Pa reference pressure
    Rd = 287. # J kg-1 K-1 gas constant for dry air
    T0 = 300. # K isothermal atmospheric temperature
    rho0 = p0/(Rd*T0) # kg m-3 reference density
    tau = 86400. # s period of motion (1 day)
    g = 9.80616 # ms-2 gravity
    H = Rd*T0/g # m scale height
    rhofc = rho0*np.exp(-fields.yfc/H)
    rhocf = rho0*np.exp(-fields.ycf/H)

    fields.u[config.store_all_timesteps*it] = -a*w0*np.pi*rho0/(K*ztop*rhofc)*np.cos(np.radians(fields.xfc))*np.sin(K*np.radians(fields.xfc))*np.cos(np.pi*fields.yfc/ztop)*np.cos(np.pi*(it + 0.5)*config.dt/tau)/(360.*a)
    fields.v[config.store_all_timesteps*it] = w0*rho0/(K*rhocf)*(-2.*np.sin(K*np.radians(fields.xcf))*np.sin(np.radians(fields.xcf)) + K*np.cos(np.radians(fields.xcf))*np.cos(K*np.radians(fields.xcf)))*np.sin(np.pi*fields.ycf/ztop)*np.cos(np.pi*(it + 0.5)*config.dt/tau)


def hadley_plane(config, fields, it):
    # Attempting to rewrite the hadley test case on the plane instead of the sphere, using the same equations but with latitude replaced by y. This might be nondivergent, but needs checking. And it needs checking whether it mimics the spherical result just in different coordinates somehow.

    a = 6.37122e6 # m Earth radius
    w0 = 0.15 # ms-1 reference vertical velocity
    K = 5 # number of overturning cells
    ztop = 1.2e4 # m height position of model top 
    #p0 = 1. #1.e5 # Pa reference pressure
    #Rd = 287. # J kg-1 K-1 gas constant for dry air
    #T0 = 300. # K isothermal atmospheric temperature
    #rho0 = 1. #p0/(Rd*T0) # kg m-3 reference density # assuming rho=rho0=1 for all of space
    tau = 86400. # s period of motion (1 day)
    #print(min(fields.xffb[:,0]/a), max(fields.xffb[:,0]/a))

    #fields.psi[config.store_all_timesteps*it] = - w0/K*np.sin(K*np.radians(fields.xffb))*np.sin(np.pi*fields.yffb/ztop)*np.cos(np.pi*(it + 0.5)*config.dt/tau)
    #fields.psi[config.store_all_timesteps*it] = - a*w0/K*np.sin(K*fields.xffb/a)*np.sin(np.pi*fields.yffb/ztop)*np.cos(np.pi*(it + 0.5)*config.dt/tau)
    fields.psi[config.store_all_timesteps*it] = a*w0/K*np.sin(K*fields.xffb/a)*np.sin(np.pi*fields.yffb/ztop)*np.cos(np.pi*(it + 0.5)*config.dt/tau)
    #fields.psi[config.store_all_timesteps*it] = - a*w0/K*np.sin(K*fields.xffb*0.4*np.pi/a)*np.sin(np.pi*fields.yffb/ztop)*np.cos(np.pi*(it + 0.5)*config.dt/tau)

    velocities_from_streamfunction(config, fields, it) # assuming rho=rho0=1 for all of space
    # fields.u[config.store_all_timesteps*it,0,:] = 0. # these fields arent inherently set to zero in hadley_plane.
    # fields.u[config.store_all_timesteps*it,-1,:] = 0.
    # fields.v[config.store_all_timesteps*it,:,0] = 0.
    # fields.v[config.store_all_timesteps*it,:,-1] = 0.


def hadley_HW(config, fields, it):
    """See HW email 15-03-2026 - nondimensional"""
    #a = 6.37122e6 # m Earth radius
    w0 = 0.15 # ms-1 reference vertical velocity
    K = 5 # number of overturning cells
    ztop = 1.2e4 # m height position of model top 
    #p0 = 1.e5 # Pa reference pressure
    #Rd = 287. # J kg-1 K-1 gas constant for dry air
    #T0 = 300. # K isothermal atmospheric temperature
    #rho0 = p0/(Rd*T0) # kg m-3 reference density
    tau = 86400. # s period of motion (1 day)
    #g = 9.80616 # ms-2 gravity
    #H = Rd*T0/g # m scale height
    #rhofc = rho0*np.exp(-fields.yfc/H)
    #rhocf = rho0*np.exp(-fields.ycf/H)

    #fields.u[config.store_all_timesteps*it] = -a*w0*np.pi*rho0/(K*ztop*rhofc)*np.cos(np.radians(fields.xfc))*np.sin(K*np.radians(fields.xfc))*np.cos(np.pi*fields.yfc/ztop)*np.cos(np.pi*(it + 0.5)*config.dt/tau)/(360.*a)
    #fields.v[config.store_all_timesteps*it] = w0*rho0/(K*rhocf)*(-2.*np.sin(K*np.radians(fields.xcf))*np.sin(np.radians(fields.xcf)) + K*np.cos(np.radians(fields.xcf))*np.cos(K*np.radians(fields.xcf)))*np.sin(np.pi*fields.ycf/ztop)*np.cos(np.pi*(it + 0.5)*config.dt/tau)

    #fields.psi[config.store_all_timesteps*it] = 2.*w0*tau/(K*np.pi*ztop)*np.cos(np.pi*fields.xffb*0.5)*np.sin(K*np.pi*fields.xffb*0.5)*np.sin(np.pi*fields.yffb)*np.cos(np.pi*(it + 0.5)*config.dt)
    #######fields.psi[config.store_all_timesteps*it] = 2.*w0*tau/(K*np.pi*ztop)*np.cos(fields.xffb)*np.sin(K*fields.xffb)*np.sin(np.pi*fields.yffb/ztop)*np.cos(np.pi*(it + 0.5)*config.dt/tau)
    #fields.psi[config.store_all_timesteps*it] = w0/a*tau/K*np.cos(fields.xffb)*np.sin(K*fields.xffb)*np.sin(np.pi*fields.yffb/ztop)*np.cos(np.pi*(it + 0.5)*config.dt/tau)
    #fields.psi[config.store_all_timesteps*it] = w0/a/K*np.cos(fields.xffb)*np.sin(K*fields.xffb)*np.sin(np.pi*fields.yffb/ztop)*np.cos(np.pi*(it + 0.5)*config.dt/tau)


    #fields.psi[config.store_all_timesteps*it] = 2.*w0*tau/(K*np.pi*np.pi*ztop)*np.cos(fields.xffb)*np.sin(K*fields.xffb)*np.sin(np.pi*fields.yffb/ztop)*np.cos(np.pi*(it + 0.5)*config.dt/tau)
    
    #Gtilde = w0/K*a
    #fields.psi[config.store_all_timesteps*it] = Gtilde*np.cos(fields.xffb/a)*np.sin(K*fields.xffb/a)*np.sin(np.pi*fields.yffb/ztop)*np.cos(np.pi*(it + 0.5)*config.dt/tau)
    G = w0/K
    fields.psi[config.store_all_timesteps*it] = 180./np.pi*G*np.cos(np.radians(fields.xffb))*np.sin(K*np.radians(fields.xffb))*np.sin(np.pi*fields.yffb/ztop)*np.cos(np.pi*(it + 0.5)*config.dt/tau)

    velocities_from_streamfunction(config, fields, it) # assuming rho=rho0=1 for all of space



    #12000
    #86400
    #0.15



def hadley_HW_differentscales(config, fields, it):
    """See HW email 15-03-2026 - trying to use physical scales (22-03-2026 not quite working yet I think, I went with nondim hadley_HW in the end for MO talk)."""
    a = 6.37122e6 # m Earth radius
    w0 = 0.15 # ms-1 reference vertical velocity
    K = 5 # number of overturning cells
    ztop = 1.2e4 # m height position of model top 
    #p0 = 1.e5 # Pa reference pressure
    #Rd = 287. # J kg-1 K-1 gas constant for dry air
    #T0 = 300. # K isothermal atmospheric temperature
    #rho0 = p0/(Rd*T0) # kg m-3 reference density
    tau = 86400. # s period of motion (1 day)
    #g = 9.80616 # ms-2 gravity
    #H = Rd*T0/g # m scale height
    #rhofc = rho0*np.exp(-fields.yfc/H)
    #rhocf = rho0*np.exp(-fields.ycf/H)

    #fields.u[config.store_all_timesteps*it] = -a*w0*np.pi*rho0/(K*ztop*rhofc)*np.cos(np.radians(fields.xfc))*np.sin(K*np.radians(fields.xfc))*np.cos(np.pi*fields.yfc/ztop)*np.cos(np.pi*(it + 0.5)*config.dt/tau)/(360.*a)
    #fields.v[config.store_all_timesteps*it] = w0*rho0/(K*rhocf)*(-2.*np.sin(K*np.radians(fields.xcf))*np.sin(np.radians(fields.xcf)) + K*np.cos(np.radians(fields.xcf))*np.cos(K*np.radians(fields.xcf)))*np.sin(np.pi*fields.ycf/ztop)*np.cos(np.pi*(it + 0.5)*config.dt/tau)

    #fields.psi[config.store_all_timesteps*it] = a*w0/K*np.cos(np.pi*0.5*np.radians(fields.xffb)/a)*np.sin(np.pi*0.5*K*np.radians(fields.xffb)/a)*np.sin(np.pi*fields.yffb/ztop)*np.cos(np.pi*(it + 0.5)*config.dt/tau)
    fields.psi[config.store_all_timesteps*it] = a*w0/K*np.cos(np.radians(fields.xffb)/a)*np.cos(np.radians(fields.xffb)/a)*np.sin(K*np.radians(fields.xffb)/a)*np.sin(np.pi*fields.yffb/ztop)*np.cos(np.pi*(it + 0.5)*config.dt/tau)




    #fields.u[config.store_all_timesteps*it] = -a*w0*np.pi/(K*ztop)*np.cos(np.radians(fields.xfc))*np.sin(K*np.radians(fields.xfc))*np.cos(np.pi*fields.yfc/ztop)*np.cos(np.pi*(it + 0.5)*config.dt/tau)/(360.*a)
    #fields.v[config.store_all_timesteps*it] = w0/(K)*(-2.*np.sin(K*np.radians(fields.xcf))*np.sin(np.radians(fields.xcf)) + K*np.cos(np.radians(fields.xcf))*np.cos(K*np.radians(fields.xcf)))*np.sin(np.pi*fields.ycf/ztop)*np.cos(np.pi*(it + 0.5)*config.dt/tau)

    velocities_from_streamfunction(config, fields, it) # assuming rho=rho0=1 for all of space


def swift1D_x(config, fields, it, L=1000., U=10., T=100.): #velocity_varying_time_space_swift() in 1D code
    """Returns the velocity varying in space and time, 1D version of the 2D nondivergent winds in the Bendall and Kent (2025) SWIFT paper. 
    nt : number of time steps
    dt : time step size
    x  : points in domain to calculate velocity for
    L  : domain size
    U  : velocity coefficient
    T  : period of oscillation
    """
    t = (it + 0.5)*config.dt # +0.5 for velocity at the half level in time for second-order accuracy
    x_prime = fields.xfc + 0.5*L - U*t
    y_prime = 0.75*L - U*t

    fields.u[config.store_all_timesteps*it] = U*np.sin(np.pi*x_prime/L)*np.sin(np.pi*x_prime/L)*np.sin(2.*np.pi*y_prime/L)*np.cos(np.pi*t/T) + U

    ##u_x = np.zeros((nt, len(x)))
    #for it in range(nt):
    #    t = (it+0.5)*dt # +0.5 for velocity at the half level in time for second-order accuracy
#
    #    u_x[config.store_all_timesteps*it] = U*np.sin(np.pi*x_prime/L)*np.sin(np.pi*x_prime/L)*np.sin(2.*np.pi*y_prime/L)*np.cos(np.pi*t/T) + U
    #return u_x