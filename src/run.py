# This takes the namelist and test_constant_u.py and grid.py information to then run the schemes with the settings specified

from src.fields import FieldContainer
from src.grid import grid_coordinates
from src.time_stepping import time_stepping
from testcases.initial.tracer import initial_tracer
import matplotlib.pyplot as plt
import numpy as np
import os
import utils.animation as anim

def run(config):

    if not os.path.exists("./output/" + config.outputdir):
        os.mkdir("./output/" + config.outputdir)
    if not os.path.exists("./output/" + config.outputdir + '/plots/'):
        os.mkdir("./output/" + config.outputdir + '/plots/')

    # Change hatch density/size
    plt.rcParams['hatch.linewidth'] = 0.01   # thickness of hatch lines
    plt.rcParams['hatch.color'] = 'black'   # optional: hatch color
    
    # Set up fields
    fields = FieldContainer(config)

    # Set up grid
    grid_coordinates(config, fields)

    # Set up initial condition (goes into the first instance of tracer)
    initial_tracer(config, fields)
    initialtracer = np.copy(fields.tracer)

    fields.maxCcc = np.zeros(fields.Ccc.shape)

    if config.ny == 1:
        plt.plot(fields.xcc, fields.tracer)
    else:
        #plt.plot(fields.xcc[:,10], fields.tracer[:,10])
        #plt.contourf(fields.xcc, fields.ycc, fields.tracer, levels=[0.,0.2,0.4,0.6,0.8,1.], cmap='cividis')
        plt.contourf(fields.xcc, fields.ycc, fields.tracer, levels=[0.,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,1.], cmap='viridis')
        cbar = plt.colorbar()
        cbar.set_label('$\\Psi$')
        plt.contour(fields.xcc, fields.ycc, fields.Ccc, colors='k', levels=[1.4], linewidths=0.5, linestyles='--')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
    plt.title('$\\Psi$ at t=0.0')
    plt.savefig('./output/' + config.outputdir + '/initial.svg')
    plt.savefig('./output/' + config.outputdir + '/plots/nt0.png', dpi=150)
    plt.savefig('./output/' + config.outputdir + '/plots/nt0.svg', dpi=150)
    plt.close()

    # Run the scheme (final result goes into fields.tracer)
    time_stepping(config, fields)#, **kwargs)


    #if config.ny == 1:
    #    plt.plot(fields.xcc, fields.tracer)
    #else:
    #    #plt.plot(fields.xcc[:,10], fields.tracer[:,10])
    #    #plt.imshow(fields.tracer)
    #    #plt.pcolormesh(fields.xcc, fields.ycc, fields.tracer)#, shading='gouraud')
    #    #plt.pcolormesh(fields.xcc, fields.ycc, fields.tracer, shading='gouraud')
    #    plt.contourf(fields.xcc, fields.ycc, fields.tracer, cmap='cividis')
    #    plt.colorbar()
    #plt.title('Field nt:' + str(config.nt))
    #plt.savefig(config.filename + '_' + str(config.scheme) + '.pdf')
    #plt.close()

    #plt.contourf(fields.xcc, fields.ycc, fields.tracer, cmap='cividis', levels=[0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.])
    plt.contourf(fields.xcc, fields.ycc, fields.tracer, cmap='viridis', levels=[0.,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,1.])
    cbar = plt.colorbar()
    cbar.set_label('$\\Psi$')
    plt.contour(fields.xcc, fields.ycc, initialtracer, colors='w', levels=[0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85], linewidths=0.5, linestyles=':')
    #plt.contour(fields.xcc, fields.ycc, initialtracer, colors='w', levels=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], linewidths=0.5, linestyles=':')
    #plt.contour(fields.xcc, fields.ycc, initialtracer, colors='w', levels=[0.15, 0.3, 0.45, 0.6, 0.75, 0.9], linewidths=0.5, linestyles=':')
    plt.contour(fields.xcc, fields.ycc, fields.Ccc, colors='k', levels=[1.4], linewidths=0.5, linestyles='--')
    bool_AdImEx = np.where(fields.thetacc > 0., 1, 0)
    plt.contourf(fields.xcc, fields.ycc, bool_AdImEx, levels=[0.5, 1], colors='none', hatches=['..'])
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('$\\Psi$ at t=' + str(config.nt*config.dt))
    plt.savefig('./output/' + config.outputdir + f'/final.svg')
    plt.close()

    #plt.contourf(fields.xcc, fields.ycc, fields.Ccc)
    #plt.colorbar()
    #plt.quiver(fields.xcc, fields.ycc, fields.u, fields.v)
    #plt.title('Courant number nt:' + str(config.nt))
    ##plt.show()
    #plt.savefig('courant_' + str(config.scheme) + '.pdf')
    #plt.close()

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
    

    if config.animate: 
        anim.create_animation(config, fields)