# This implements the time stepping loop.


import src.schemes as sch
import testcases.velocity
from time import perf_counter
from logging import info
import numpy as np


def time_stepping(config, fields, **kwargs):
    scheme = getattr(sch, config.scheme, **kwargs)


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