import numpy as np

def velocity(config, fields, it):
    # Define velocity fields at time step it (actually velocities taken at it + 0.5 for second-order accuracy)
    globals()[config.velocity_setting](config, fields, it)


def swift_nondiv_streamfunction(config, fields, it):
    # Non-divergent velocity field at half time levels from SWIFT testcase using the streamfunction

    Lx = config.xmax - config.xmin
    Ly = config.ymax - config.ymin
    if Lx != Ly:
        raise ValueError("SWIFT nondiv velocity is only nondivergent for square domains (Lx=Ly).")
    coeff = 0.5*Lx - config.u0*(it + 0.5)*config.dt # using one coeff assumes Lx=Ly
    fields.psi[it] = - config.u0*Lx/np.pi*np.sin(np.pi*(fields.xffb + coeff)/Lx)*np.sin(np.pi*(fields.xffb + coeff)/Lx)*np.sin(np.pi*(fields.yffb + coeff)/Ly)*np.sin(np.pi*(fields.yffb + coeff)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) - config.u0*fields.yffb + config.u0*fields.xffb # assumes Lx=Ly
    velocities_from_streamfunction(config, fields, it)


def velocities_from_streamfunction(config, fields, it):
    """Deriving the u and v velocity components from the streamfunction psi.
    psi[i,j] is defined at i-1/2, j-1/2 -> bottom left cell corner"""

    fields.u[it] = - (fields.psi[it][:-1,1:] - fields.psi[it][:-1,:-1])/fields.dycc
    fields.v[it] = (fields.psi[it][1:,:-1] - fields.psi[it][:-1,:-1])/fields.dxcc


def hadley(config, fields, it):
    """Inspired by Hadley-like circulation in Kent et al. 2014."""
    w0 = 0.15 # ms-1 reference vertical velocity
    K = 5 # number of overturning cells
    ztop = 1.2e4 # m height position of model top 
    tau = 86400. # s period of motion (1 day)
    G = w0/K

    fields.psi[it] = 180./np.pi*G*np.cos(np.radians(fields.xffb))*np.sin(K*np.radians(fields.xffb))*np.sin(np.pi*fields.yffb/ztop)*np.cos(np.pi*(it + 0.5)*config.dt/tau)

    velocities_from_streamfunction(config, fields, it) # assuming rho=rho0=1 for all of space
