# This implements the time stepping loop.

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

import src.schemes as sch
import testcases.velocity
from time import perf_counter
from logging import info
import numpy as np


def update_Ccc_minmax(config, fields, it):
    """Update the minimum and maximum Courant number over all time steps so far with those of time step it.
    Needed as fields.Ccc only holds the latest time step if store_all_timesteps is False."""
    Ccc = fields.Ccc[config.store_all_timesteps*it]
    fields.Ccc_min_alltimes = np.minimum(fields.Ccc_min_alltimes, Ccc.min())
    fields.Ccc_max_alltimes = np.maximum(fields.Ccc_max_alltimes, Ccc.max())


def time_stepping(config, fields, **kwargs):
    scheme = getattr(sch, config.scheme, **kwargs)
    fields.Ccc_min_alltimes, fields.Ccc_max_alltimes = np.inf, -np.inf # min and max Courant number over all time steps, updated by update_Ccc_minmax()


    if config.timing == False:
        # Time stepping loop
        iterations_convergence = np.zeros(config.nt)
        for it in range(config.nt):
            # Update velocity fields
            testcases.velocity.velocity(config, fields, it)

            #print('Velocity at boundaries u left', fields.u[it,0,:]) # needs to go to zero but doesnt, but that is done with the cos phi factor I think
            #print('Velocity at boundaries u right', fields.u[it,-1,:]) # needs to go to zero but doesnt, but that is done with the cos phi factor I think
            #print('Velocity at boundaries v bottom', fields.v[it,:,0])
            #print('Velocity at boundaries v top', fields.v[it,:,-1]) #not quite the top
            #exit()
            # Call the advection scheme
            scheme(config, fields, it, iterations_convergence=iterations_convergence)
            update_Ccc_minmax(config, fields, it)
    else: 
        iterations_convergence = np.zeros(config.nt)

        time_total, time_total_velocity, time_total_scheme = 0., 0., 0.
        start_total = perf_counter()
        # Time stepping loop
        for it in range(config.nt):
            
            start_velocity = perf_counter()
            # Update velocity fields
            testcases.velocity.velocity(config, fields, it)
            end_velocity = perf_counter()
            time_total_velocity += end_velocity - start_velocity

            start_scheme = perf_counter()
            # Call the advection scheme
            scheme(config, fields, it, iterations_convergence=iterations_convergence)
            end_scheme = perf_counter()
            time_total_scheme += end_scheme - start_scheme
            update_Ccc_minmax(config, fields, it) # after the scheme timer, so it doesn't count towards time_total_scheme

        end_total = perf_counter()
        time_total = end_total - start_total

        print(f'Time total time stepping: {time_total}')
        print(f'Time total velocity: {time_total_velocity}')
        print(f'Time total scheme: {time_total_scheme}')
        info(f'Time total time stepping: {time_total}')
        info(f'Time total velocity: {time_total_velocity}')
        info(f'Time total scheme: {time_total_scheme}')
        np.save(config.outputdir + f'data/time_total.npy', time_total)
        np.save(config.outputdir + f'data/time_total_velocity.npy', time_total_velocity)
        np.save(config.outputdir + f'data/time_total_scheme.npy', time_total_scheme)

        # Store iterations fields for convergence analysis
        total_iterations_convergence = np.sum(iterations_convergence)
        print(f'Total number of iterations over all times: {total_iterations_convergence}')
        info(f'iterations_convergence={iterations_convergence}')
        info(f'total_iterations_convergence: {total_iterations_convergence}')
        np.save(config.outputdir + f'data/iterations_convergence.npy', iterations_convergence)
        np.save(config.outputdir + f'data/total_iterations_convergence.npy', total_iterations_convergence)


    #print('Final tracer field difference ', abs(fields.tracer[-1] - fields.tracer[0]))
    #plt.pcolormesh(fields.xcc, fields.ycc, abs(fields.tracer[-1] - fields.tracer[0]), norm=LogNorm())
    #plt.colorbar()
    #plt.show()
        

            #print(f"Time step {it+1}/{config.nt} completed in {end - start:.4f} seconds.")
            # preconditioner?

            #time_avg += end - start
            #print('Min Courant number: ', fields.Ccc[it].min())
            #print('Max Courant number: ', fields.Ccc[it].max())

            # (None of these results are final, just first explorations)
            # 10x larger time step takes about 10x longer
            # velocity calc seems to be a small part of the time
            # see what happens with third order matrix!!!
            # GMRES(m) seems to be about 5 times faster than GCR(k)
    
        #print(f"Average time per time step: {time_avg / config.nt:.4f} seconds.")