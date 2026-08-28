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
        info(f'total_iterations_convergence: {total_iterations_convergence}')
        np.save(config.outputdir + f'data/iterations_convergence.npy', iterations_convergence)
        np.save(config.outputdir + f'data/total_iterations_convergence.npy', total_iterations_convergence)
