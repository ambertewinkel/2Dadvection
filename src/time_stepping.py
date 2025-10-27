# This implements the time stepping loop.


import src.schemes as sch
import testcases.velocity
import matplotlib.pyplot as plt


def time_stepping(config, fields, **kwargs):
    scheme = getattr(sch, config.scheme, **kwargs)
    #plt.contourf(fields.xfc, fields.yfc, fields.u, label='initial')
    #plt.show()
    # Time stepping loop
    for it in range(config.nt):
        # Update velocity fields
        testcases.velocity.velocity(config, fields, it) # !!! when nonconstant, I need to make sure to take the half level velocity for second-order accuracy

        # Call the advection scheme
        scheme(config, fields)
        
        
        #if it == config.nt-1: #% 10 == 0:# and config.verbose:
        #    plt.contourf(fields.xcc, fields.ycc, fields.tracer)#, label='tracer at it='+str(it))
        #    plt.quiver(fields.xfc, fields.yfc, fields.u, fields.v)#, label='vel at it='+str(it))
        #    plt.colorbar()
        #    plt.title('it='+str(it))
        #    plt.show()