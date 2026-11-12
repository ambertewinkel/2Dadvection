"""This code logs and prints the minimum and maximum Courant numbers over the whole of a simulation.
Author: Amber te Winkel"""


import numpy as np
import os
from sys import argv, exit
from src.config import Config
from os.path import dirname
import logging
from pathlib import Path


def output_Courant_minmax():


    if len(argv) < 1:
        print("Usage: python Cminmax.py <outputdir>")
        exit(1)

    outputdir = dirname(__file__) + '/output/' + argv[1] +'/'

    # Set up logging
    logfile = outputdir + 'Cminmax.log'
    
    # Load data. For runs from 2026-11-08 on, min_Ccc.npy and max_Ccc.npy hold the min and max over all time steps
    # (tracked during the run). Earlier runs saved the min and max of the stored Ccc field in them, which is only the
    # final time step unless store_all_timesteps was True, and runs before 2026-05-01 have no such files (-> Ccc.npy).
    if os.path.exists(outputdir + 'data/max_Ccc.npy'):
        Ccc_min = np.load(outputdir + 'data/min_Ccc.npy')
        Ccc_max = np.load(outputdir + 'data/max_Ccc.npy')
    else:
        Ccc = np.load(outputdir + 'data/Ccc.npy')
        Ccc_min = np.min(Ccc)
        Ccc_max = np.max(Ccc)
        print('No max_Ccc.npy found, so using Ccc.npy (only the final time step, unless store_all_timesteps was True).')

    # Output values
    print('Looking at', outputdir)
    print(f'Minimum Courant number over all times: {Ccc_min}')
    print(f'Maximum Courant number over all times: {Ccc_max}')
    with open(logfile, 'w') as f:
        f.write(f"Minimum Courant number over all times: {Ccc_min}\n")
        f.write(f"Maximum Courant number over all times: {Ccc_max}\n")
    

if __name__ == "__main__":
    output_Courant_minmax()