# This code defines error functions. 


import numpy as np
from sys import argv, exit
from src.config import Config
from os.path import dirname
import logging
from pathlib import Path
import yaml
import testcases.initial.tracer as it



def l2norm(numerical, analytic, V):
    """This calculates the l2 norm from an output field compared to the analytic solution.
    field : 2D array of floats, output field from the numerical scheme
    analytic : 2D array of floats, analytic solution
    V : 2D array of floats, cell-centred widths
    """
    numerator = np.sum(V*(numerical - analytic)*(numerical - analytic))
    denominator = np.sum(V*analytic*analytic)
    return np.sqrt(numerator/(denominator + 1.e-16))


def load_config(path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config


import matplotlib.pyplot as plt
def error():
    if len(argv) < 2:
        print("Usage: python error.py <outputdir> <setting>")
        exit(1)

    outputdir = dirname(__file__) + '/output/' + argv[1] +'/'

    # Get setting
    setting = argv[2]

    # Set up logging
    logfile = outputdir + 'error.log'

    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )        
    print(f'See error log {logfile}')
    print(f"Running error analysis for dir={outputdir} and setting={setting}")
    logging.info(f"Error analysis for directory: {outputdir}")

    # Load config file
    configfile = [str(p) for p in Path(outputdir).glob("*.yml")]
    if len(configfile) == 0:
        print(f"No config file found in {outputdir}")
        exit(1)
    elif len(configfile) > 1:
        print(f"Multiple config files found in {outputdir}: {configfile}")
        exit(1)
    else: 
        print('Config file found: ' + configfile[0])
    config = Config.from_file(configfile[0])
    logging.info(f"Config file: {configfile[0]}")
    logging.info(f"Config loaded: {config}")

    # Load tracer and grid fields
    data, fieldnames = {}, ['tracer', 'dxcc', 'dycc', 'xcc', 'ycc']
    for f in fieldnames:
        data[f] = np.load(outputdir + 'data/' + f + '.npy')

    if setting == 'finaltoinitial':
        print("Computing error at final time compared to initial condition...")
        # Load initial condition
        l2_error = l2norm(data['tracer'][-1], data['tracer'][0], data['dxcc']*data['dycc'])

        # Output l2 norms to file
        print(l2_error)
        with open(outputdir + 'l2norms.out', 'w') as f:
            f.write('l2 norm (final compared to initial)\n')
            f.write(f'{l2_error:.6e}\n')
    elif setting == 'finaltoanalytic':
        # Find arguments from config file
        config_loaded = load_config(Path(configfile[0]))
        #print(config_loaded.nt)
        #exit()

        # Access values with .get() to fall back on defaults for optional keys.
        nt = config_loaded.get("nt")
        dt = config_loaded.get("dt")
        xmin = config_loaded.get("xmin")
        xmax = config_loaded.get("xmax")
        ymin = config_loaded.get("ymin")
        ymax = config_loaded.get("ymax")
        mref = config_loaded.get("mref", 0.5)
        mmag = config_loaded.get("mmag", 0.5)
        xcc = data['xcc']
        ycc = data['ycc']

        initial_tracer_func = config_loaded.get("initial_tracer") + '_analytic' # only implemented for sine_swift
        velocity_setting = config_loaded.get("velocity_setting")
        if velocity_setting == 'constant_uv':
            u = config_loaded.get("constant_u")
            v = config_loaded.get("constant_v")
        elif velocity_setting == 'constant_u':
            u = config_loaded.get("constant_u")
            v = 0.
        elif velocity_setting == 'constant_v':
            u = 0.
            v = config_loaded.get("constant_v")
        else:
            print('ERROR: No valid velocity setting.')
            exit()

        # Calculate analytic solution
        analytic = getattr(it, initial_tracer_func)(xmin, xmax, ymin, ymax, mref, mmag, xcc, ycc, u, v, nt*dt)

        # Calculate error difference with analytic solution
        l2_error = l2norm(data['tracer'][-1], analytic, data['dxcc']*data['dycc'])

        # Output l2 norms to file
        print(l2_error)
        with open(outputdir + 'l2norms.out', 'w') as f:
            f.write(f'l2 norm (final compared to analytic at time t={nt*dt})\n')
            f.write(f'{l2_error:.6e}\n')

    print('Done')


if __name__ == "__main__":
    error()