"""This file plots the results of the advection tests from stored .npy files.

Run as python plot_tracer.py <'swift' or 'hadley' or 'constancy'> <outputdir>

Always at the final time step. 

<outputdir> is the name of a specific output directory to take the data from (as specified in /output/)

Author: Amber te Winkel
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from utils.animation import create_animation
from src.config import Config
from src.output import set_up_plots_directory
import pycpt as pc
from error import l2norm


def load_data(outputdir):
    """Loads stored .npy files from the output directory."""

    data, fieldnames = {}, []
    for filename in ['tracer.npy', 'Ccc.npy', 'xcc.npy', 'ycc.npy','dxcc.npy','dycc.npy']:
        fieldnames.append(filename.replace('.npy', ''))
        data[fieldnames[-1]] = np.load(outputdir + 'data/' + filename)

    return fieldnames, data


def plot_fields(config, fieldnames, data, plots_dir, plot_type):
    """Plots the fields based on the specified setting."""

    # Plot the tracer initial condition if tracer is outputted (i.e. field_to_plot is 'all' or 'tracer')
    field = 'tracer'
    minval, maxval = np.min(data[field][-1]), np.max(data[field][-1])
    if plot_type == 'swift':
        l2_error = l2norm(data['tracer'][-1], data['tracer'][0], data['dxcc']*data['dycc'])
        plot_figure(data, data['xcc'], data['ycc'], data[field][-1], field, f'$\\Psi^{{ {config.nt} }} \\in [{minval:.2f},{maxval:.2f}]$, $l_2 =${l2_error:.5f}', 'x (m)', 'y (m)', 'viridis', plots_dir + f'{field}_nt{config.nt}', minval, maxval, plot_type='swift')        
    elif plot_type == 'hadley':
        l2_error = l2norm(data['tracer'][-1], data['tracer'][0], data['dxcc']*data['dycc'])
        plot_figure(data, data['xcc'], data['ycc']*1e-3, data[field][-1], field, f'$\\Psi^{{ {config.nt} }} \\in [{minval:.2f},{maxval:.2f}]$, $l_2 =${l2_error:.5f}', 'lat (deg)', 'z (km)', 'viridis', plots_dir + f'{field}_nt{config.nt}', minval, maxval, plot_type='hadley') # for final time
        #plot_figure(data, data['xcc'], data['ycc']*1e-3, data[field][-1], field, f'$\\Psi^{{ {config.nt} }} \\in [{minval:.2f},{maxval:.2f}]$', 'lat (deg)', 'z (km)', 'viridis', plots_dir + f'{field}_nt{config.nt}', minval, maxval, plot_type='hadley') # for halftime
    elif plot_type == 'constancy':
        plot_figure(data, data['xcc'], data['ycc'], data[field][-1], field, f'$\\Psi^{{ {config.nt} }}-0.5 \\in [{minval-0.5:.2e},{maxval-0.5:.2e}]$', 'x (m)', 'y (m)', 'viridis', plots_dir + f'{field}_nt{config.nt}', minval, maxval, plot_type='constancy')        
    else:
        print('plot_type not recognized. Please choose from: swift, hadley, constancy.')

    

def plot_figure(data, x, y, fielddata, fieldname, title, xlabel, ylabel, cmap, filename, minval, maxval, add_hatching=False, thetacc=None, add_initial=False, initialtracer=None, plot_type='swift'):
    
    no_x = False #True #False #True #False #True #False #True #False#False#True
    no_y = False #True #False #True #False #True#False #True
    no_colorbar = False #True #False #True #False#False #True #False #True
    
    if plot_type == 'swift':
        plt.figure(figsize=(5.5,5))
        palette = pc.read('wh-bl-gr-ye-re.cpt') # read in a color palette
        cmap = palette.cmap

    elif plot_type == 'hadley':
        plt.figure(figsize=(6,3.5)) # 6,3.5 for left plot, 7,3.5 for right plot
        palette = pc.read('wh-bl-gr-ye-re.cpt') # read in a color palette
        cmap = palette.cmap

    elif plot_type == 'constancy':
        plt.figure(figsize=(6.5,5))
        cmap = 'bwr'

    diff = maxval - minval
    extend = 'neither'
    if diff == 0.:
        contourlevels = None
        extend = 'both'
        if plot_type == 'constancy':
            #absextent = max(abs(minval-5.0E-1), abs(maxval-5.0E-1))
            #contourlevels = list(np.linspace(-absextent, absextent, 20, endpoint=True))
            #contourlevels = [-0.02,-0.015,-0.01,-0.005,-0.000000001,0.000000001,0.005,0.01,0.015,0.02]
            contourlevels = [-0.02,-0.001,0.001,0.02]
            #contourlevels = [-0.02,-0.015,-0.01,-0.005,-0.001,0.001,0.005,0.01,0.015,0.02]
            plt.contourf(x, y, fielddata-5.0E-1, cmap=cmap, levels=contourlevels, extend=extend)
    else:
        extend = 'both'
        if plot_type == 'swift' or plot_type == 'hadley':
            contourlevels = list(np.arange(-0.15, 1.16, 0.1)) # np.linspace(-0.15, 1.15, 13))  # list(np.linspace(minval, maxval, 20, endpoint=True))
            plt.contourf(x, y, fielddata, cmap=cmap, levels=contourlevels, extend=extend)
        elif plot_type == 'constancy':
            #absextent = max(abs(minval-5.0E-1), abs(maxval-5.0E-1))
            #contourlevels = list(np.linspace(-absextent, absextent, 20, endpoint=True))
            #contourlevels = [-0.02,-0.015,-0.01,-0.005,-0.000000001,0.000000001,0.005,0.01,0.015,0.02]
            contourlevels = [-0.02,-0.001,0.001,0.02]
            #contourlevels = [-0.02,-0.015,-0.01,-0.005,-0.001,0.001,0.005,0.01,0.015,0.02]
            plt.contourf(x, y, fielddata-5.0E-1, cmap=cmap, levels=contourlevels, extend=extend)
    if not no_colorbar: 
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=12)  # change colorbar tick label size
    plt.contour(x, y, data['Ccc'][-1], colors='k', levels=[1.4, 2.8, 4.2, 5.6, 7., 8.4, 9.8], linewidths=0.5, linestyles=':')
    plt.title(title, size=15)
    if no_x: 
        plt.tick_params(axis='x', labelcolor='none') # if I don't want anything apart from the tickmarks in the plot
    else: 
        plt.tick_params(labelsize=12)
        plt.xlabel(xlabel, size=15)
    if no_y:
        plt.tick_params(axis='y', labelcolor='none') # if I don't want anything apart from the tickmarks in the plot
    else:
        plt.tick_params(labelsize=12)
        plt.ylabel(ylabel, size=15)
    if plot_type == 'swift' or plot_type == 'constancy': plt.gca().set_aspect('equal', adjustable='box')  # Ensures equal aspect ratio
    plt.tight_layout()
    #plt.show()
    plt.savefig(filename + '.pdf', dpi=300)
    plt.savefig(filename + '.svg', dpi=300)
    plt.close()   


def plot_tracer():
    """Main plotting function called when running this script from the terminal."""

    plot_type = sys.argv[1]  # 'all' or specific field name
    outputdir = os.path.dirname(__file__) + '/output/' + sys.argv[2] + '/'
    for filename in os.listdir(outputdir):
        if filename.endswith('.yml'):   
            configfile = outputdir + filename

    config = Config.from_file(configfile)
    plots_dir = set_up_plots_directory(outputdir)

    # Load stored .npy files
    fieldnames, data = load_data(outputdir)

    # Plot fields
    plot_fields(config, fieldnames, data, plots_dir, plot_type)

    print("Plotting done.")


if __name__ == '__main__':
    plot_tracer()