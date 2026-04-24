"""This file plots the results of the advection tests from stored .npy files.

Run as python plot.py <field> <setting> <outputdir>

<field> can be
- 'all' to plot all fields
- specific field name as in the .npy file to plot the individual field

<setting> can be 
- 'all' to plot all time results in individual plots
- 'final' to plot only the final time result
- 'anim' to create an animation of the time results

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


def load_data(outputdir, field_to_plot):
    """Loads stored .npy files from the output directory."""

    data, fieldnames = {}, []
    if field_to_plot == 'all':
        for filename in os.listdir(outputdir + 'data/'):
            if filename.endswith('.npy'):
                field_name = filename[:-4]  # Remove .npy extension
                fieldnames.append(field_name)
                data[field_name] = np.load(outputdir + 'data/' + filename)
    elif field_to_plot == 'tracer' or field_to_plot == 'Ccc' or field_to_plot == 'thetacc':
        filename = field_to_plot + '.npy'
        fieldnames.append(field_to_plot)
        data[field_to_plot] = np.load(outputdir + 'data/' + filename)
        filename = 'xcc.npy'
        fieldnames.append('xcc')
        data['xcc'] = np.load(outputdir + 'data/' + filename)
        filename = 'ycc.npy'
        fieldnames.append('ycc')
        data['ycc'] = np.load(outputdir + 'data/' + filename)
        filename = 'dxcc.npy'
        fieldnames.append('dxcc')
        data['dxcc'] = np.load(outputdir + 'data/' + filename)
        filename = 'dycc.npy'
        fieldnames.append('dycc')
        data['dycc'] = np.load(outputdir + 'data/' + filename)
    elif field_to_plot == 'u' or field_to_plot == 'thetafc':
        filename = field_to_plot + '.npy'
        fieldnames.append(field_to_plot)
        data[field_to_plot] = np.load(outputdir + 'data/' + filename)
        filename = 'xfc.npy'
        fieldnames.append('xfc')
        data['xfc'] = np.load(outputdir + 'data/' + filename)
        filename = 'yfc.npy'
        fieldnames.append('yfc')
        data['yfc'] = np.load(outputdir + 'data/' + filename)
    elif field_to_plot == 'v' or field_to_plot == 'thetacf':
        filename = field_to_plot + '.npy'
        fieldnames.append(field_to_plot)
        data[field_to_plot] = np.load(outputdir + 'data/' + filename)
        filename = 'xcf.npy'
        fieldnames.append('xcf')
        data['xcf'] = np.load(outputdir + 'data/' + filename)
        filename = 'ycf.npy'
        fieldnames.append('ycf')
        data['ycf'] = np.load(outputdir + 'data/' + filename)
    elif field_to_plot == 'psi':
        filename = field_to_plot + '.npy'
        fieldnames.append(field_to_plot)
        data[field_to_plot] = np.load(outputdir + 'data/' + filename)
        filename = 'xffb.npy'
        fieldnames.append('xffb')
        data['xffb'] = np.load(outputdir + 'data/' + filename)
        filename = 'yffb.npy'
        fieldnames.append('yffb')
        data['yffb'] = np.load(outputdir + 'data/' + filename)
    else: 
        raise ValueError('Plotting field not recognized.')

    return fieldnames, data


def plot_fields(config, fieldnames, data, plots_dir, setting):
    """Plots the fields based on the specified setting."""

    # Plot fields based on setting
    if setting == 'final':
        # Plot the tracer initial condition if tracer is outputted (i.e. field_to_plot is 'all' or 'tracer')
        if 'tracer' in fieldnames:
            minval, maxval = np.min(data['tracer'][0]), np.max(data['tracer'][0])
            plot_figure(data['xcc'], data['ycc'], data['tracer'][0], 'tracer', '$\\Psi$ at $n_t$=0', 'x', 'y', 'viridis', plots_dir + f'tracer_nt0.pdf', minval, maxval)
        for field in fieldnames:
            minval, maxval = np.min(data[field][-1]), np.max(data[field][-1])
            if field == 'tracer':
                #if 'thetacc' in fieldnames:
                #    add_hatching, thetacc = True, data['thetacc'][-1] 
                #else:
                #    add_hatching, thetacc = False, None
                l2_error = l2norm(data['tracer'][-1], data['tracer'][0], data['dxcc']*data['dycc'])
                #plot_figure(data['xcc'], data['ycc'], data[field][-1], field, f'$\\Psi$ at $n_t$ = {config.nt} \n min = {minval:.3f}, max = {maxval:.3f}, l2 = {l2_error:.3f}', 'x', 'y', 'viridis', plots_dir + f'{field}_nt{config.nt}.svg', minval, maxval)#, add_hatching, thetacc, True, data['tracer'][0])
                #plot_figure(data['xcc'], data['ycc'], data[field][-1], field, f'$\\Psi^{{ {config.nt} }} \\in [{minval:.2f},{maxval:.2f}]$, $l_2$ = {l2_error:.5f}', 'x', 'y', 'viridis', plots_dir + f'{field}_nt{config.nt}.pdf', minval, maxval)#, add_hatching, thetacc, True, data['tracer'][0])
                plot_figure(data['xcc'], data['ycc'], data[field][-1], field, f'$\\Psi^{{ {config.nt} }} \\in [{minval:.2f},{maxval:.2f}]$', 'x', 'y', 'viridis', plots_dir + f'{field}_nt{config.nt}.pdf', minval, maxval)#, add_hatching, thetacc, True, data['tracer'][0])
            elif field == 'Ccc' or field == 'thetacc' or field == 'dthetafc' or field == 'dthetacf':
                plot_figure(data['xcc'], data['ycc'], data[field][-1], field, f'{field} at $n_t$ = {config.nt}, min = {minval:.4f}, max = {maxval:.4f}', 'x', 'y', 'viridis', plots_dir + f'{field}_nt{config.nt}.svg', minval, maxval)
            elif field == 'u' or field == 'thetafc':
                plot_figure(data['xfc'], data['yfc'], data[field][-1], field, f'{field} at $n_t$ = {config.nt}, min = {minval:.4f}, max = {maxval:.4f}', 'x', 'y', 'viridis', plots_dir + f'{field}_nt{config.nt}.svg', minval, maxval)
            elif field == 'v' or field == 'thetacf':
                plot_figure(data['xcf'], data['ycf'], data[field][-1], field, f'{field} at $n_t$ = {config.nt}, min = {minval:.4f}, max = {maxval:.4f}', 'x', 'y', 'viridis', plots_dir + f'{field}_nt{config.nt}.svg', minval, maxval)
            elif field == 'psi':
                plot_figure(data['xffb'], data['yffb'], data[field][-1], field, f'{field} at $n_t$ = {config.nt}, min = {minval:.4f}, max = {maxval:.4f}', 'x', 'y', 'viridis', plots_dir + f'{field}_nt{config.nt}.svg', minval, maxval)
    elif setting == 'all':
        for field in fieldnames:
            minval, maxval, add_hatching = np.min(data[field]), np.max(data[field]), False
            if field == 'tracer':# or field == 'density':
                plot_figure(data['xcc'], data['ycc'], data[field][0], field, f'$\\Psi$ at $n_t$=0', 'x', 'y', 'viridis', plots_dir + f'{field}_nt0.svg', minval, maxval)
                #add_hatching = True if 'thetacc' in fieldnames else False
                for it in range(1,config.nt+1):
                    plot_figure(data['xcc'], data['ycc'], data[field][it], field, f'$\\Psi$ at $n_t$ = {it}', 'x', 'y', 'viridis', plots_dir + f'{field}_nt{it}.svg', minval, maxval, add_hatching, data['thetacc'][it-1])
            elif field == 'Ccc' or field == 'thetacc':
                for it in range(config.nt):
                    plot_figure(data['xcc'], data['ycc'], data[field][it], field, f'{field} at $n_t$ = {it+0.5}', 'x', 'y', 'viridis', plots_dir + f'{field}_nt{it+1}.svg', minval, maxval)
            elif field == 'u' or field == 'thetafc':
                for it in range(config.nt):
                    plot_figure(data['xfc'], data['yfc'], data[field][it], field, f'{field} at $n_t$ = {it+0.5}', 'x', 'y', 'viridis', plots_dir + f'{field}_nt{it+1}.svg', minval, maxval)
            elif field == 'v' or field == 'thetacf':
                for it in range(config.nt):
                    plot_figure(data['xcf'], data['ycf'], data[field][it], field, f'{field} at $n_t$ = {it+0.5}', 'x', 'y', 'viridis', plots_dir + f'{field}_nt{it+1}.svg', minval, maxval)
            elif field == 'psi':
                for it in range(config.nt):
                    plot_figure(data['xffb'], data['yffb'], data[field][it], field, f'{field} at $n_t$ = {it+0.5}', 'x', 'y', 'viridis', plots_dir + f'{field}_nt{it+1}.svg', minval, maxval)
    elif setting == 'anim':
        anim_dir = plots_dir + '../animations/'
        if not os.path.exists(anim_dir): os.mkdir(anim_dir)
        for field in fieldnames:
            if field in ['tracer', 'density', 'Ccc', 'thetacc', 'u', 'v', 'thetafc', 'thetacf', 'psi']:
                # Determine min and max values for consistent color scale
                add_hatching, minval, maxval = False, np.min(data[field]), np.max(data[field])
                if field == 'tracer' or field == 'density':
                    plot_figure(data['xcc'], data['ycc'], data[field][0], field, f'$\\Psi$ at $n_t$=0', 'x', 'y', 'viridis', plots_dir + f'{field}_nt0.png', minval, maxval)
                    for it in range(1, config.nt+1):
                        #if 'thetacc' in fieldnames:
                        #    add_hatching, thetacc = True, data['thetacc'][it-1]
                        #else:
                        #    add_hatching, thetacc = False, None
                        plot_figure(data['xcc'], data['ycc'], data[field][it], field, f'$\\Psi$ at $n_t$ = {it}', 'x', 'y', 'viridis', plots_dir + f'{field}_nt{it}.png', minval, maxval)#, add_hatching, thetacc)
                else:
                    if field == 'Ccc' or field == 'thetacc':
                        x, y = data['xcc'], data['ycc']
                    elif field == 'u' or field == 'thetafc':
                        x, y = data['xfc'], data['yfc']
                    elif field == 'v' or field == 'thetacf':
                        x, y = data['xcf'], data['ycf']
                    elif field == 'psi':
                        x, y = data['xffb'], data['yffb']
                    for it in range(config.nt):
                        plot_figure(x, y, data[field][it], field, f'{field} at $n_t$ = {it+0.5}', 'x', 'y', 'viridis', plots_dir + f'{field}_nt{it+1}.png', minval, maxval)
                create_animation(config, plots_dir, anim_dir, field)
    else: 
        raise ValueError('Plotting setting not recognized.')


def plot_figure(x, y, fielddata, fieldname, title, xlabel, ylabel, cmap, filename, minval, maxval, add_hatching=False, thetacc=None, add_initial=False, initialtracer=None):
    plt.figure(figsize=(10, 5))
    #palette = pc.read('wh-bl-gr-ye-re.cpt') # read in a color palette
    cmap = 'bwr'
    diff = maxval - minval
    print(minval, maxval)
    extend = 'neither'
    if diff == 0.:
        contourlevels = None
    else:
        if fieldname == 'tracer' or fieldname == 'density':
            extend = 'both'
            absextent = max(abs(minval-5.0E-1), abs(maxval-5.0E-1))
            #contourlevels = list(np.linspace(-absextent, absextent, 20, endpoint=True))
            contourlevels = list(np.linspace(-1.0E-9, 1.0E-9, 20, endpoint=True))
            ##contourlevels = list(np.linspace(-maxval-0.5+1., maxval-0.5, 10)) # np.linspace(-0.15, 1.15, 13))  # list(np.linspace(minval, maxval, 20, endpoint=True))
            #contourlevels = list(np.arange(-0.15, 1.16, 0.1)) # np.linspace(-0.15, 1.15, 13))  # list(np.linspace(minval, maxval, 20, endpoint=True))
            #contourlevels = list(np.linspace(-maxval+1., maxval, 10)) # np.linspace(-0.15, 1.15, 13))  # list(np.linspace(minval, maxval, 20, endpoint=True))
        else: 
            contourlevels = list(np.linspace(minval, maxval, 20, endpoint=True))
        #contourlevels = list(np.linspace(minval, maxval, 20, endpoint=True)) #list(np.arange(-0.15, 1.15, 0.1)) # np.linspace(-0.15, 1.15, 13))  # list(np.linspace(minval, maxval, 20, endpoint=True))
        #contourlevels = [minval, (minval + 0.15*diff), (minval + 0.25*diff), (minval + 0.35*diff), (minval + 0.45*diff), (minval + 0.55*diff), (minval + 0.65*diff), (minval + 0.75*diff), (minval + 0.85*diff), maxval]
    #plt.contourf(x, y, fielddata, cmap=palette.cmap, levels=contourlevels, extend=extend)
    plt.contourf(x, y, fielddata-5.0E-1, cmap=cmap, levels=contourlevels, extend=extend)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)  # change colorbar tick label size
    if add_initial:
        if diff == 0.:
            contourlevels_reduced = None
        else:
            contourlevels_reduced = contourlevels[1:-1]
            #contourlevels_reduced = [(minval + 0.15*diff), (minval + 0.25*diff), (minval + 0.35*diff), (minval + 0.45*diff), (minval + 0.55*diff), (minval + 0.65*diff), (minval + 0.75*diff), (minval + 0.85*diff)]
        plt.contour(x, y, initialtracer, colors='w', levels=contourlevels_reduced, linewidths=0.5, linestyles=':')
    if add_hatching:
        plt.contour(x, y, thetacc, colors='k', levels=[0.], linewidths=0.5, linestyles='--')
        bool_AdImEx = np.where(thetacc > 0., 1, 0)
        plt.contourf(x, y, bool_AdImEx, levels=[0.5, 1], colors='none', hatches=['..'])
    plt.title(title, size=15)
    plt.xlabel(xlabel, size=15)
    plt.ylabel(ylabel, size=15)
    plt.tick_params(labelsize=12)
    plt.gca().set_aspect('equal', adjustable='box')  # Ensures equal aspect ratio
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()   


def plot():
    """Main plotting function called when running this script from the terminal."""

    # Get command line arguments
    if len(sys.argv) < 4:
        print("Usage: python plot.py <field> <setting> <outputdir>")
        exit(1)
    else:
        field_to_plot = sys.argv[1]  # 'all' or specific field name
        setting = sys.argv[2]  # 'all', 'final', 'anim'
        outputdir = os.path.dirname(__file__) + '/output/' + sys.argv[3] + '/'
        for filename in os.listdir(outputdir):
            if filename.endswith('.yml'):   
                configfile = outputdir + filename

    config = Config.from_file(configfile)
    plots_dir = set_up_plots_directory(outputdir)

    # Load stored .npy files
    fieldnames, data = load_data(outputdir, field_to_plot)

    # Change hatch density/size 
    plt.rcParams['hatch.linewidth'] = 0.01   # thickness of hatch lines
    plt.rcParams['hatch.color'] = 'black'   # optional: hatch color

    # Plot fields
    plot_fields(config, fieldnames, data, plots_dir, setting)

    print("Plotting done.")


if __name__ == '__main__':
    plot()