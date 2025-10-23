


def velocity(config, fields, it):
    # Define velocity fields at time step it
    globals()[config.velocity_setting](config, fields)


def constant_u(config, fields):
    fields.u[:,:] = config.constant_u
    fields.v[:,:] = 0.


def constant_v(config, fields):
    fields.u[:,:] = 0.
    fields.v[:,:] = config.constant_v


def constant_uv(config, fields):
    fields.u[:,:] = config.constant_u
    fields.v[:,:] = config.constant_v


# !!! when nonconstant, I need to make sure to take the half level velocity for second-order accuracy