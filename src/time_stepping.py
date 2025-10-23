# This implements the time stepping loop.


import src.schemes as sch
import testcases.velocity


def time_stepping(config, fields, **kwargs):
    scheme = getattr(sch, config.scheme, **kwargs)

    # Time stepping loop
    for it in range(config.nt):
        # Update velocity fields
        testcases.velocity.velocity(config, fields, it) # !!! when nonconstant, I need to make sure to take the half level velocity for second-order accuracy

        # Call the advection scheme
        scheme(config, fields)