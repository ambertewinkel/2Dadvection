# This implements the time stepping loop.


import src.schemes as sch
import testcases.velocity
import matplotlib.pyplot as plt
import numpy as np


def time_stepping(config, fields, **kwargs):
    scheme = getattr(sch, config.scheme, **kwargs)
    #plt.contourf(fields.xfc, fields.yfc, fields.u, label='initial')
    #plt.show()
    maxC = 0.
    maxpsi = np.max(fields.psi)
    minpsi = np.min(fields.psi)
    # Time stepping loop
    for it in range(config.nt):
        # Update velocity fields
        testcases.velocity.velocity(config, fields, it) # !!! when nonconstant, I need to make sure to take the half level velocity for second-order accuracy
        #plt.contourf(fields.xfc, fields.yfc, fields.u)
        #plt.colorbar()
        #plt.title(f'u velocity at t= {str((it+1)*config.dt)} from {config.velocity_setting}')
        #plt.savefig('./output/' + config.outputdir + f'/plots/u_velocity_{config.velocity_setting}.png', dpi=150)
        #plt.close()
        #plt.contourf(fields.xcf, fields.ycf, fields.v)
        #plt.colorbar()
        #plt.title(f'v velocity at t= {str((it+1)*config.dt)} from {config.velocity_setting}')
        #plt.savefig('./output/' + config.outputdir + f'/plots/v_velocity_{config.velocity_setting}.png', dpi=150)
        #plt.close()

        #plt.contourf(fields.xcc, fields.ycc, config.dx)
        #plt.title('dx')
        #plt.colorbar()
        #plt.show()

        # Call the advection scheme
        scheme(config, fields)

        if config.animate:
            #plt.contourf(fields.xcc, fields.ycc, fields.tracer, levels=[0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.], cmap='cividis')
            plt.contourf(fields.xcc, fields.ycc, fields.tracer, levels=[0.,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,1.], cmap='viridis')
            cbar = plt.colorbar()
            cbar.set_label('$\\Psi$')            
            plt.contour(fields.xcc, fields.ycc, fields.Ccc, colors='k', levels=[1.4], linewidths=0.5, linestyles='--')
            bool_AdImEx = np.where(fields.thetacc > 0., 1, 0)
            plt.contourf(fields.xcc, fields.ycc, bool_AdImEx, levels=[0.5, 1], colors='none', hatches=['..'])
            plt.xlabel('$x$')
            plt.ylabel('$y$')
            plt.title('$\\Psi$ at t=' + str((it+1)*config.dt))
            plt.savefig('./output/' + config.outputdir + f'/plots/nt{it+1}.png', dpi=150)
            plt.savefig('./output/' + config.outputdir + f'/plots/nt{it+1}.svg', dpi=150)
            plt.close()
        
            maxCnew = np.max(fields.Ccc)
            if maxCnew > maxC:
                maxC = maxCnew
            print('max C at it=' + str(it+1) + ' : ' + str(maxCnew) + ' (temporal max: ' + str(maxC) + ')')
            #contour = plt.contourf(fields.xcc, fields.ycc, fields.Ccc, levels=10, vmin=0., vmax=maxC, cmap='viridis')
            contour = plt.contourf(fields.xcc, fields.ycc, fields.Ccc, levels=[0.,0.1*maxC,0.2*maxC,0.3*maxC,0.4*maxC,0.5*maxC,0.6*maxC,0.7*maxC,0.8*maxC,0.9*maxC,maxC], cmap='viridis')
            cbar = plt.colorbar(contour)
            cbar.set_label('$C$')
            plt.contour(fields.xcc, fields.ycc, fields.Ccc, colors='k', levels=[1.4], linewidths=0.5, linestyles='--')
            bool_AdImEx = np.where(fields.thetacc > 0., 1, 0)
            plt.contourf(fields.xcc, fields.ycc, bool_AdImEx, levels=[0.5, 1], colors='none', hatches=['..'])
            plt.xlabel('$x$')
            plt.ylabel('$y$')
            plt.title('$C$ at t=' + str((it+1)*config.dt))
            plt.savefig('./output/' + config.outputdir + f'/plots/Cnt{it+1}.png', dpi=150)
            plt.savefig('./output/' + config.outputdir + f'/plots/Cnt{it+1}.svg', dpi=150)
            plt.close()            
        
            maxpsinew = np.max(fields.psi)
            minpsinew = np.min(fields.psi)
            if maxpsinew > maxpsi:
                maxpsi = maxpsinew
            if minpsinew < minpsi:
                minpsi = minpsinew
            if minpsi < -maxpsi:
                maxpsi = - minpsi
            else:
                minpsi = - maxpsi
            contour = plt.contourf(fields.xffb, fields.yffb, fields.psi, levels=[minpsi, minpsi + 0.1*(maxpsi-minpsi), minpsi + 0.2*(maxpsi-minpsi),minpsi + 0.3*(maxpsi-minpsi),minpsi + 0.4*(maxpsi-minpsi),minpsi + 0.5*(maxpsi-minpsi),minpsi + 0.6*(maxpsi-minpsi),minpsi + 0.7*(maxpsi-minpsi),minpsi + 0.8*(maxpsi-minpsi),minpsi + 0.9*(maxpsi-minpsi),maxpsi], cmap='viridis')
            cbar = plt.colorbar(contour)
            cbar.set_label('Streamfunction')
            plt.xlabel('$x$')
            plt.ylabel('$y$')
            plt.title('Streamfunction at t=' + str((it+1)*config.dt))
            plt.savefig('./output/' + config.outputdir + f'/plots/strfn_nt{it+1}.png', dpi=150)
            plt.savefig('./output/' + config.outputdir + f'/plots/strfn_nt{it+1}.svg', dpi=150)
            plt.close()  

        #if it == config.nt-1: #% 10 == 0:# and config.verbose:
        #    plt.contourf(fields.xcc, fields.ycc, fields.tracer)#, label='tracer at it='+str(it))
        #    plt.quiver(fields.xfc, fields.yfc, fields.u, fields.v)#, label='vel at it='+str(it))
        #    plt.colorbar()
        #    plt.title('it='+str(it))
        #    plt.show()