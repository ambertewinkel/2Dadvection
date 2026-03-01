# This implements the time stepping loop.


import src.schemes as sch
import testcases.velocity
from time import perf_counter


def time_stepping(config, fields, **kwargs):
    scheme = getattr(sch, config.scheme, **kwargs)

    time_avg = 0.0
    # Time stepping loop
    for it in range(config.nt):
        
        # Update velocity fields
        testcases.velocity.velocity(config, fields, it)

        start = perf_counter()
        # Call the advection scheme
        scheme(config, fields, it)

        end = perf_counter()

        print(f"Time step {it+1}/{config.nt} completed in {end - start:.4f} seconds.")
        # preconditioner?

        time_avg += end - start
        print('Min Courant number: ', fields.Ccc[it].min())
        print('Max Courant number: ', fields.Ccc[it].max())

        # (None of these results are final, just first explorations)
        # 10x larger time step takes about 10x longer
        # velocity calc seems to be a small part of the time
        # see what happens with third order matrix!!!
        # GMRES(m) seems to be about 5 times faster than GCR(k)

    print(f"Average time per time step: {time_avg / config.nt:.4f} seconds.")
    