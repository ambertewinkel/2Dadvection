# This code takes the output from a directory in the terminal input to check whether the result is conserved for each step in time.

import numpy as np
import os
import sys
import logging
from datetime import datetime


def set_up_file(outputdir):
    """Sets up the conservation output file."""
            
    consfile = outputdir + 'conservation.txt'
    if os.path.exists(consfile):
        os.remove(consfile)
    print(f'See file {consfile}')    
    logging.basicConfig(
        filename=consfile,
        level=logging.INFO,
        format="%(message)s",
    )    
    logging.info(f'Time and date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logging.info("Time \t (n) - (n0) \t (n) - (n-1) \t normalised: ((n) - (n0)) / domain_area \t normalised: ((n) - (n-1)) / domain_area \n")


def check_conservation():
    """Main plotting function called when running this script from the terminal."""

    # Get command line arguments
    if len(sys.argv) < 2:
        print("Usage: python conservation.py <outputdir>")
        exit(1)
    else:
        outputdir = os.path.dirname(__file__) + '/output/' + sys.argv[1] + '/'

    set_up_file(outputdir)

    # Load stored .npy files
    fieldtracer = np.load(outputdir + 'data/tracer.npy')
    fielddxcc = np.load(outputdir + 'data/dxcc.npy') # needed for nonuniform grid
    fielddycc = np.load(outputdir + 'data/dycc.npy') # needed for nonuniform grid
    total_area = np.sum(fielddxcc[0,:])*np.sum(fielddycc[0,:])

    bool_conservation = True

    nt = np.shape(fieldtracer)[0] - 1 # number of time steps
    M0 = np.sum(fieldtracer[0, :, :] * fielddxcc[:, :] * fielddycc[:, :]) # initial total amount
    Mtm1 = M0
    for it in range(1, nt + 1):
        Mt = np.sum(fieldtracer[it, :, :] * fielddxcc[:, :] * fielddycc[:, :]) # total amount at time t
        cons_0 = Mt - M0
        cons_m1 = Mt - Mtm1
        normalized_0 = cons_0 / total_area
        normalized_m1 = cons_m1 / total_area
        if abs(normalized_0) > 1e-10 or abs(normalized_m1) > 1e-10:
            bool_conservation = False
        logging.info(f'n {it} --> M(n)-M(n0): {cons_0}, M(n)-M(n-1): {cons_m1}, NORM M(n)-M(n0): {normalized_0}, NORM M(n)-M(n-1): {normalized_m1}')
        Mtm1 = Mt.copy() # total amount at time t-1

    logging.info('')
    logging.info(f'Conservation (based on normalised)? {bool_conservation}')
    print(f'Conservation (based on normalised)? {bool_conservation}')
    logging.shutdown()


if __name__ == '__main__':
    check_conservation()
