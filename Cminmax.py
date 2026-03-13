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
    
    # Load data
    Ccc = np.load(outputdir + 'data/Ccc.npy')

    Ccc_min = np.min(Ccc)
    Ccc_max = np.max(Ccc)

    # Output values
    print('Looking at', outputdir)
    print(f'Minimum Courant number over all times: {Ccc_min}')
    print(f'Maximum Courant number over all times: {Ccc_max}')
    with open(logfile, 'w') as f:
        f.write(f"Minimum Courant number over all times: {Ccc_min}\n")
        f.write(f"Maximum Courant number over all times: {Ccc_max}\n")
    

if __name__ == "__main__":
    output_Courant_minmax()