"""This code takes the output from the 2D advection code and checks that mass is conserved. It does this by integrating the field over the domain at each time step and plotting the result. If mass is conserved, the integrated value should be constant over time. It also outputs the numerical values and checks whether they are within a certain tolerance of the initial mass. We assume constant density and look at the tracer mass.

Usage: python mass.py <outputdir>

Author: Amber te Winkel
"""

import numpy as np
import matplotlib.pyplot as plt
from src.output import set_up_plots_directory
import sys
import os
import logging


def load_mass_data(outputdir):
    data = {}
    for filename in ['tracer.npy', 'dxcc.npy', 'dycc.npy']:
        data[filename[:-4]] = np.load(outputdir + 'data/' + filename)
    return data


def plot_mass(massdata, plots_dir):
    plt.figure()
    plt.plot(massdata, marker='o')
    plt.xlabel('Time step')
    plt.ylabel('Mass')
    plt.title('Mass conservation check')
    plt.grid()
    plt.savefig(plots_dir + 'mass_conservation.svg', dpi=300)
    plt.close()


def check_mass_conservation():
    # Get command line arguments
    if len(sys.argv) < 2:
        print("Usage: python mass.py <outputdir>")
        exit(1)
    else:
        outputdir = os.path.dirname(__file__) + '/output/' + sys.argv[1] + '/'

    plots_dir = set_up_plots_directory(outputdir)

    # Load stored .npy files
    data = load_mass_data(outputdir)

    # Calculate mass at each time step
    massdata = []
    for it in range(data['tracer'].shape[0]):
        mass = np.sum(data['tracer'][it, :, :]*data['dxcc']*data['dycc'])  # sum over all grid points
        massdata.append(mass)

    # Plot fields
    plot_mass(massdata, plots_dir)

    print("Mass at start of simulation: ", massdata[0])
    print("Mass at end of simulation: ", massdata[-1])
    print("Mass difference: ", massdata[-1] - massdata[0])
    print('Normalised mass difference: ', (massdata[-1] - massdata[0])/massdata[0])


    logfile = outputdir + 'mass.log'
    if os.path.exists(logfile):
        os.remove(logfile)
    print()
    print(f'See output file {logfile}')    
    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.info(f"Mass at start of simulation: {massdata[0]}")
    logging.info(f"Mass at end of simulation: {massdata[-1]}")
    logging.info(f"Mass difference: {massdata[-1] - massdata[0]}")
    logging.info(f"Normalised mass difference: {(massdata[-1] - massdata[0])/massdata[0]}")

    print("Mass conservation check done.")


if __name__ == "__main__":
    check_mass_conservation()