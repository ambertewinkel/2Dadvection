import numpy as np


def velocity(config, fields, it):
    # Define velocity fields at time step it
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


# !!! when nonconstant, I need to make sure to take the half level velocity for second-order accuracy

def swift_nondiv(config, fields, it):
    # Non-divergent velocity field at half time levels

    Lx = config.xmax - config.xmin
    Ly = config.ymax - config.ymin
    
    ut = config.u0*(it + 0.5)*config.dt

    fields.u = config.u0*np.sin(np.pi*(fields.xfc + 0.5*Lx - ut)/Lx)*np.sin(np.pi*(fields.xfc + 0.5*Lx - ut)/Lx)*np.sin(2.*np.pi*(fields.yfc + 0.5*Ly - ut)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.u0
    
    fields.v = -config.u0*np.sin(2.*np.pi*(fields.xcf + 0.5*Lx - ut)/Lx)*np.sin(np.pi*(fields.ycf + 0.5*Ly - ut)/Ly)*np.sin(np.pi*(fields.ycf + 0.5*Ly - ut)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.u0




    #fields.v[:,:] = 0.
    
    ##fields.u = config.u0*np.sin(np.pi*(fields.xfc/Lx + 0.5))*np.sin(np.pi*(fields.xfc/Lx + 0.5))*np.sin(2.*np.pi*(fields.yfc/Ly + 0.5))*np.cos(np.pi*(it+0.5)*config.dt/config.T) + config.u0
    ##fields.v = -config.u0*np.sin(2.*np.pi*(fields.xfc/Lx + 0.5))*np.sin(np.pi*(fields.yfc/Ly + 0.5))*np.sin(np.pi*(fields.yfc/Ly + 0.5))*np.cos(np.pi*(it+0.5)*config.dt/config.T) + config.u0



def swift_nondiv_double(config, fields, it):
    # Non-divergent velocity field at half time levels

    Lx = config.xmax - config.xmin
    Ly = config.ymax - config.ymin
    
    ut = config.u0*(it + 0.5)*config.dt

    fields.u = 2.*config.u0*np.sin(np.pi*(fields.xfc + 0.5*Lx - ut)/Lx)*np.sin(np.pi*(fields.xfc + 0.5*Lx - ut)/Lx)*np.sin(2.*np.pi*(fields.yfc + 0.5*Ly - ut)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.u0
    
    fields.v = -2.*config.u0*np.sin(2.*np.pi*(fields.xcf + 0.5*Lx - ut)/Lx)*np.sin(np.pi*(fields.ycf + 0.5*Ly - ut)/Ly)*np.sin(np.pi*(fields.ycf + 0.5*Ly - ut)/Ly)*np.cos(np.pi*(it + 0.5)*config.dt/config.T) + config.u0




    #fields.v[:,:] = 0.
    
    ##fields.u = config.u0*np.sin(np.pi*(fields.xfc/Lx + 0.5))*np.sin(np.pi*(fields.xfc/Lx + 0.5))*np.sin(2.*np.pi*(fields.yfc/Ly + 0.5))*np.cos(np.pi*(it+0.5)*config.dt/config.T) + config.u0
    ##fields.v = -config.u0*np.sin(2.*np.pi*(fields.xfc/Lx + 0.5))*np.sin(np.pi*(fields.yfc/Ly + 0.5))*np.sin(np.pi*(fields.yfc/Ly + 0.5))*np.cos(np.pi*(it+0.5)*config.dt/config.T) + config.u0





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
    pass