# This takes the namelist and test_constant_u.py and grid.py information to then run the schemes with the settings specified


from src.fields import FieldContainer
from src.grid import grid_coordinates
from src.time_stepping import time_stepping
from testcases.initial.tracer import initial_tracer
from src.output import store_output_npy, set_up_logging, set_up_output_directory, store_config, getminmax
from logging import info
from error import l2norm
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
        l2_error = l2norm(fields.tracer[-1], fields.tracer[0], fields.dxcc*fields.dycc)
        info(f'l2norm = {l2_error}')
        print(f'l2norm = {l2_error}')
        np.save(config.outputdir + f'data/l2_error.npy', l2_error)

    np.save(config.outputdir + f'data/min_tracer.npy', getminmax(fields.tracer, -1)[0])
    np.save(config.outputdir + f'data/max_tracer.npy', getminmax(fields.tracer, -1)[1])

    info(f'Minimum and maximum C over all times: {np.min(fields.Ccc)}, {np.max(fields.Ccc)}')    
    print(f'Minimum and maximum C over all times: {np.min(fields.Ccc)}, {np.max(fields.Ccc)}')
    np.save(config.outputdir + f'data/min_Ccc.npy', np.min(fields.Ccc))
    np.save(config.outputdir + f'data/max_Ccc.npy', np.max(fields.Ccc))

