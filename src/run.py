# This takes the namelist and test_constant_u.py and grid.py information to then run the schemes with the settings specified

from src.fields import FieldContainer
from src.grid import grid_coordinates
from src.time_stepping import time_stepping
from testcases.initial.tracer import initial_tracer
import matplotlib.pyplot as plt
import numpy as np

def run(config):

    # Set up fields
    fields = FieldContainer(config)

    # Set up grid
    grid_coordinates(config, fields)

    # Set up initial condition (goes into the first instance of tracer)
    initial_tracer(config, fields)

    fields.maxCcc = np.zeros(fields.Ccc.shape)

    if config.ny == 1:
        plt.plot(fields.xcc, fields.tracer)
    else:
        #plt.plot(fields.xcc[:,10], fields.tracer[:,10])
        plt.contourf(fields.xcc, fields.ycc, fields.tracer)
        plt.colorbar()
    plt.savefig('initial.pdf')
    plt.close()

    # Run the scheme (final result goes into fields.tracer)
    time_stepping(config, fields)#, **kwargs)


    if config.ny == 1:
        plt.plot(fields.xcc, fields.tracer)
    else:
        #plt.plot(fields.xcc[:,10], fields.tracer[:,10])
        #plt.imshow(fields.tracer)
        #plt.pcolormesh(fields.xcc, fields.ycc, fields.tracer)#, shading='gouraud')
        #plt.pcolormesh(fields.xcc, fields.ycc, fields.tracer, shading='gouraud')
        plt.contourf(fields.xcc, fields.ycc, fields.tracer)
        plt.colorbar()
    plt.title('Field nt:' + str(config.nt))
    plt.savefig(config.filename + '_' + str(config.scheme) + '.pdf')
    plt.close()

    plt.contourf(fields.xcc, fields.ycc, fields.Ccc)
    plt.colorbar()
    plt.quiver(fields.xcc, fields.ycc, fields.u, fields.v)
    plt.title('Courant number nt:' + str(config.nt))
    #plt.show()
    plt.savefig('courant_' + str(config.scheme) + '.pdf')
    plt.close()

    #plt.contourf(fields.xcc, fields.ycc, fields.u)
    #plt.colorbar()
    ##plt.quiver(fields.xcc, fields.ycc, fields.u, fields.v)
    #plt.title('u nt:' + str(config.nt))
    ##plt.show()
    #plt.savefig('u_' + str(config.scheme) + '.pdf')
    #plt.close()    
    #
    #plt.contourf(fields.xcc, fields.ycc, fields.v)
    #plt.colorbar()
    ##plt.quiver(fields.xcc, fields.ycc, fields.u, fields.v)
    #plt.title('v nt:' + str(config.nt))
    ##plt.show()
    #plt.savefig('v_' + str(config.scheme) + '.pdf')
    #plt.close()

    #plt.contourf(fields.xcc, fields.ycc, fields.maxCcc)
    #plt.colorbar()
    #plt.title('Max Courant number over time')
    #plt.show()
    ##plt.savefig('courant_' + str(config.scheme) + '.pdf')
    #plt.close()

    #plt.contourf(fields.xcc, fields.ycc, fields.thetacc)
    #plt.colorbar()
    #plt.title('Implicitness at cell centers')
    #plt.show()
    ##plt.savefig('courant_' + str(config.scheme) + '.pdf')
    #plt.close()
#
    ## Store result
    