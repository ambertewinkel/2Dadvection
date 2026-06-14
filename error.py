# This code defines error functions. 


import numpy as np
from sys import argv, exit
from src.config import Config
from os.path import dirname
import logging
from pathlib import Path



def l2norm(numerical, analytic, V):
    """This calculates the l2 norm from an output field compared to the analytic solution.
    field : 2D array of floats, output field from the numerical scheme
    analytic : 2D array of floats, analytic solution
    V : 2D array of floats, cell-centred widths
    """
    numerator = np.sum(V*(numerical - analytic)*(numerical - analytic))
    denominator = np.sum(V*analytic*analytic)
    return np.sqrt(numerator/(denominator + 1.e-16))
    
import matplotlib.pyplot as plt
def error():
    if len(argv) < 2:
        print("Usage: python error.py <outputdir> <setting>")
        exit(1)

    #if argv[1] == 'test':
    #    dir = dirname(__file__) + '/output/' + argv[1] +'/'
    #else:
    #    dir = dirname(__file__) + '/output/dated/' + argv[1] +'/'

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
    data, fieldnames = {}, ['tracer', 'dxcc', 'dycc']
    for f in fieldnames:
        data[f] = np.load(outputdir + 'data/' + f + '.npy')

    if setting == 'finaltoinitial':
        print("Computing error at final time compared to initial condition...")
        # Load initial condition
        l2_error = l2norm(data['tracer'][-1], data['tracer'][0], data['dxcc']*data['dycc'])
        #l2_error = np.linalg.norm((data['tracer'][-1]-data['tracer'][0])*(data['dxcc']*data['dycc']))/np.linalg.norm(data['tracer'][0]*(data['dxcc']*data['dycc']))
        #l2_error = np.linalg.norm((data['tracer'][-1]-data['tracer'][0])*(data['dxcc']*data['dycc'])*(data['dxcc']*data['dycc']))/np.linalg.norm(data['tracer'][0]*(data['dxcc']*data['dycc'])*(data['dxcc']*data['dycc']))
        print(data['dxcc'])
        #plt.plot(data['dxcc'][:,0])
        #plt.show()
        print()
        print(data['dycc'])
        #plt.plot(data['dycc'][0,:])
        #plt.contourf(data['tracer'][0])
        #plt.show()
        #plt.contourf(data['tracer'][-1]-data['tracer'][0])
        #plt.show()
        #np.linalg.norm(np.abs(data['tracer'][-1]-data['tracer'][0])*data['dxcc']*data['dycc']/np.sum(data['tracer'][0]*data['dxcc']*data['dycc']))
        # Output l2 norms to file
        print(l2_error)
        with open(outputdir + 'l2norms.out', 'w') as f:
            f.write('l2 norm (final compared to initial)\n')
            f.write(f'{l2_error:.6e}\n')

    #elif exact == 'analytic': # other option to perhaps include later

    print('Done')


if __name__ == "__main__":
    error()