"""This code plots the grid used for a certain case, it is used for the AdHImEx paper for the lowerleft case, and thus provides a zoomed area inside of it for that as well.""" 

import numpy as np
from sys import argv, exit
from os.path import dirname
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset



def plot_grid_with_inset_region():


    if len(argv) < 1:
        print("Usage: python plot_grid.py <outputdir>")
        exit(1)

    outputdir = dirname(__file__) + '/output/' + argv[1] +'/'
    
    # Load data
    xf = 0.001*np.load(outputdir + 'data/xfc.npy')[:,0]
    yf = 0.001*np.load(outputdir + 'data/ycf.npy')[0,:]

    xf = np.append(xf, 0.5)
    yf = np.append(yf, 0.5)

    fig, ax = plt.subplots(figsize=(6,5.8))

    # Plot vertical grid lines
    for x in xf:
        ax.plot([x, x], [yf[0], yf[-1]], color='grey', linewidth=0.5)

    # Plot horizontal grid lines
    for y in yf:
        ax.plot([xf[0], xf[-1]], [y, y], color='grey', linewidth=0.5)

    ax.set_aspect('equal')
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_xlim(-0.5,0.5)
    ax.set_ylim(-0.5,0.5)

    # Inset
    x_range, y_range = [-0.42, -0.38], [-0.22, -0.18]#[-0.395, -0.34], [-0.215, -0.16]

    inset = ax.inset_axes([0.3, 0.4, 0.55, 0.55])
    # Plot vertical grid lines
    for x in xf:
        inset.plot([x, x], [yf[0], yf[-1]], color='grey', linewidth=1)

    # Plot horizontal grid lines
    for y in yf:
        inset.plot([xf[0], xf[-1]], [y, y], color='grey', linewidth=1)

    inset.tick_params(labelsize=12)
    inset.set_xlim(*x_range)
    inset.set_ylim(*y_range)
    # Color the inset border
    for spine in inset.spines.values():
        spine.set_edgecolor("red")
    mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="red")  

    plt.tight_layout() 
    plt.savefig(outputdir + 'grid.pdf', dpi=300)
    plt.savefig(outputdir + 'grid.svg', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_grid_with_inset_region()