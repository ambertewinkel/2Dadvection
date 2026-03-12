# This implements the time stepping loop.


import src.schemes as sch
import testcases.velocity


def time_stepping(config, fields, **kwargs):
    scheme = getattr(sch, config.scheme, **kwargs)

    # Time stepping loop
    for it in range(config.nt):
        # Update velocity fields
        testcases.velocity.velocity(config, fields, it)

        #print('Velocity at boundaries u left', fields.u[it,0,:]) # needs to go to zero but doesnt, but that is done with the cos phi factor I think
        #print('Velocity at boundaries u right', fields.u[it,-1,:]) # needs to go to zero but doesnt, but that is done with the cos phi factor I think
        #print('Velocity at boundaries v bottom', fields.v[it,:,0])
        #print('Velocity at boundaries v top', fields.v[it,:,-1]) #not quite the top
        #exit()
        # Call the advection scheme
        scheme(config, fields, it)