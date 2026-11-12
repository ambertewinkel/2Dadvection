# This takes the namelist and test_constant_u.py and grid.py information to then run the schemes with the settings specified


from src.fields import FieldContainer
from src.grid import grid_coordinates
from src.time_stepping import time_stepping
from testcases.initial.tracer import initial_tracer
from src.output import store_output_npy, set_up_logging, set_up_output_directory, store_config, getminmax
from logging import info
from error import l2norm, linfnorm
import numpy as np

def run(config, configfile, configname):
    """This function is called by run_model() and
    -> Sets up output, fields, grid, initial conditions,
    -> Calls time_stepping to run the advection test,
    -> Stores output in .npy files.
    """

    set_up_output_directory(config)
    store_config(config, configfile, configname)
    set_up_logging(config, configname)
    
    # Set up fields
    fields = FieldContainer(config)

    # Set up grid
    grid_coordinates(config, fields)

    # Set up initial conditions
    initial_tracer(config, fields)

    # Run the scheme
    time_stepping(config, fields)

    # Store result in npy files
    if config.timing == False: store_output_npy(config, fields)

    if config.print_error: 
        # L2 error
        l2_error = l2norm(fields.tracer[-1], fields.tracer[0], fields.dxcc*fields.dycc)
        info(f'l2norm = {l2_error}')
        print(f'l2norm = {l2_error}')
        np.save(config.outputdir + f'data/l2_error.npy', l2_error)

        # Linf error
        linf_error = linfnorm(fields.tracer[-1], fields.tracer[0])
        info(f'linfnorm = {linf_error}')
        print(f'linfnorm = {linf_error}')
        np.save(config.outputdir + f'data/linf_error.npy', linf_error)

    info('Min and max tracer at final time step: ' + str(getminmax(fields.tracer, -1)))
    print('Min and max tracer at final time step: ' + str(getminmax(fields.tracer, -1)))
    np.save(config.outputdir + f'data/min_tracer.npy', getminmax(fields.tracer, -1)[0])
    np.save(config.outputdir + f'data/max_tracer.npy', getminmax(fields.tracer, -1)[1])

    # Min and max Courant number over all time steps (tracked during time_stepping, as fields.Ccc only holds the final time step if store_all_timesteps is False)
    info(f'Minimum and maximum C over all times: {fields.Ccc_min_alltimes}, {fields.Ccc_max_alltimes}')    
    print(f'Minimum and maximum C over all times: {fields.Ccc_min_alltimes}, {fields.Ccc_max_alltimes}')
    np.save(config.outputdir + f'data/min_Ccc.npy', fields.Ccc_min_alltimes)
    np.save(config.outputdir + f'data/max_Ccc.npy', fields.Ccc_max_alltimes)

