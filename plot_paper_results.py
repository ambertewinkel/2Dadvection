"""Plots the Hadley-like circulation, constancy preservation, and BK25 tracer fields.

Run as python plot_paper_results.py

Produces three figures from the stored .npy files in ./output_paper/, saved to
./output_paper/plots/:

  hadley     a 2x2 comparison of the Hadley-like circulation.  Columns are the
             two runs, rows are the halfway time step (maximum deformation,
             top) and the final time step (return to the initial condition,
             bottom).  All four panels share a color scale, a single
             colorbar, and the x and y axes.

  constancy  a side-by-side comparison of constancy preservation, both panels
             at the final time step, sharing a color scale, a single
             colorbar, and the y axis.

  bk25      a 3x2 comparison of the BK25 nondivergent flow at the final time
             step.  Columns are the grid, the unlimited result, and the
             FCT-limited result; rows are the explicit reference (top) and
             AdHImEx (bottom).  All six panels share the x and y axes, and the
             four tracer panels share a color scale and a single colorbar.

Author: Amber te Winkel
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import pycpt as pc
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from src.config import Config
from error import l2norm


CCONTOURLEVELS = [1.4, 2.8, 4.2, 5.6, 7., 8.4, 9.8]

# Hadley
CONTOURLEVELS = list(np.arange(-0.15, 1.16, 0.1))
XLABEL_HADLEY, YLABEL_HADLEY = 'lat (deg)', 'z (km)'

# Constancy
CONTOURLEVELS_CP = [-0.02, -0.001, 0.001, 0.02]
XLABEL_CP, YLABEL_CP = 'x (m)', 'y (m)'

# BK25
XLABEL_BK, YLABEL_BK = 'x (m)', 'y (m)'
XLIM_BK, YLIM_BK = [-500., 500.], [-500., 500.]
INSET_X, INSET_Y = [-419., -379.], [-222., -182.]


# Shared
def load_data(outputdir):
    """Loads stored .npy files from the output directory."""

    data, fieldnames = {}, []
    for filename in ['tracer.npy', 'Ccc.npy', 'xcc.npy', 'ycc.npy','dxcc.npy','dycc.npy']:
        fieldnames.append(filename.replace('.npy', ''))
        data[fieldnames[-1]] = np.load(outputdir + 'data/' + filename)

    return fieldnames, data


# Shared
def load_run(dirname):
    """Loads the config and stored .npy files for a single output directory."""

    outputdir = os.path.dirname(__file__) + '/' + dirname + '/'
    configfile = next(outputdir + f for f in os.listdir(outputdir)
                      if f.endswith('.yml'))

    _, data = load_data(outputdir)

    return Config.from_file(configfile), data


# Shared
def add_label(ax, label):
    """Adds a textbox naming the scheme to the top left corner of a panel."""

    ax.text(0.04, 0.96, label, transform=ax.transAxes, va='top', ha='left',
            size=12, bbox=dict(boxstyle='square', fc='white', ec='k'))


# Shared
def finish_figure(fig, cf, axes, figname):
    """Adds the shared colorbar and saves the figure to the plots directory."""

    cbar = fig.colorbar(cf, ax=axes, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=12)

    plots_dir = './output_paper/plots/'
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
    for ext in ('pdf', 'svg'):
        fig.savefig(f'{plots_dir}{figname}.{ext}', dpi=300)
    plt.close(fig)


# Hadley
def panel_title_hadley(config, data, index, show_l2):
    """Builds the title for one panel: value range, and l2 if requested."""

    field = data['tracer'][index]
    minval, maxval = np.min(field), np.max(field)
    nt = index  # tracer.npy stores time levels 0 to nt inclusive

    title = f'$\\Psi^{{ {nt} }} \\in [{minval:.2f},{maxval:.2f}]$'
    if show_l2:
        l2_error = l2norm(field, data['tracer'][0],
                          data['dxcc'] * data['dycc'])
        title += f', $\\ell_2 =${l2_error:.5f}'

    return title


# Hadley
def plot_panel_hadley(ax, config, data, index, show_l2, label, cmap):
    """Draws one tracer panel with a scheme label and returns the filled contour set."""

    x, y = data['xcc'], data['ycc'] * 1e-3

    cf = ax.contourf(x, y, data['tracer'][index], cmap=cmap,
                     levels=CONTOURLEVELS, extend='both')
    ax.contour(x, y, data['Ccc'][index-1], colors='k', levels=CCONTOURLEVELS,
               linewidths=0.5, linestyles=':')
    ax.set_title(panel_title_hadley(config, data, index, show_l2), size=15)
    add_label(ax, label)
    ax.tick_params(labelsize=12)

    return cf


# Constancy
def panel_title_cp(config, data):
    """Builds the title for one panel: the range of the departure from 0.5."""

    field = data['tracer'][-1]
    minval, maxval = np.min(field), np.max(field)

    return (f'$\\Psi^{{ {config.nt} }}-0.5 \\in '
            f'[{minval-5.0E-1:.2e},{maxval-5.0E-1:.2e}]$')


# Constancy
def plot_panel_cp(ax, config, data, label):
    """Draws one departure-from-0.5 panel with a label and returns the filled contour set."""

    x, y = data['xcc'], data['ycc']

    cf = ax.contourf(x, y, data['tracer'][-1] - 5.0E-1, cmap='bwr',
                     levels=CONTOURLEVELS_CP, extend='both')
    ax.contour(x, y, data['Ccc'][-1], colors='k', levels=CCONTOURLEVELS,
               linewidths=0.5, linestyles=':')
    ax.set_title(panel_title_cp(config, data), size=14)
    add_label(ax, label)
    ax.set_xlabel(XLABEL_CP, size=14)
    ax.tick_params(labelsize=12)
    ax.set_box_aspect(1) 

    return cf


# BK25
def load_grid(dirname):
    """Loads the cell face positions defining the grid of an output directory."""

    outputdir = os.path.dirname(__file__) + '/' + dirname + '/'

    xf = np.load(outputdir + 'data/xfc.npy')[:, 0]
    yf = np.load(outputdir + 'data/ycf.npy')[0, :]

    return np.append(xf, 500.), np.append(yf, 500.)


# BK25
def draw_grid_lines(ax, xf, yf, linewidth):
    """Draws the vertical and horizontal grid lines on a set of axes."""

    for x in xf:
        ax.plot([x, x], [yf[0], yf[-1]], color='grey', linewidth=linewidth)
    for y in yf:
        ax.plot([xf[0], xf[-1]], [y, y], color='grey', linewidth=linewidth)


# BK25
def plot_panel_grid(ax, xf, yf):
    """Draws one grid panel with a zoomed inset of the refined region."""

    draw_grid_lines(ax, xf, yf, 0.5)
    ax.tick_params(labelsize=12)
    ax.set_box_aspect(1)

    inset = ax.inset_axes([0.3, 0.4, 0.55, 0.55])
    draw_grid_lines(inset, xf, yf, 1)
    inset.tick_params(labelsize=12)
    inset.set_xlim(*INSET_X)
    inset.set_ylim(*INSET_Y)
    for spine in inset.spines.values():
        spine.set_edgecolor("red")
    pp, p1, p2 = mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="red")
    for artist in (pp, p1, p2):
        artist.set_zorder(5)


# BK25
def panel_title_bk25(config, data):
    """Builds the title for one panel: value range and l2 against the initial condition."""

    field = data['tracer'][-1]
    minval, maxval = np.min(field), np.max(field)
    l2_error = l2norm(field, data['tracer'][0], data['dxcc'] * data['dycc'])

    return (f'$\\Psi^{{ {config.nt} }} \\in [{minval:.2f},{maxval:.2f}]$'
            f', $\\ell_2 =${l2_error:.4f}')


# BK25
def plot_panel_bk25(ax, config, data, label, cmap):
    """Draws one tracer panel with a scheme label and returns the filled contour set."""

    x, y = data['xcc'], data['ycc']

    cf = ax.contourf(x, y, data['tracer'][-1], cmap=cmap,
                     levels=CONTOURLEVELS, extend='both')
    ax.contour(x, y, data['Ccc'][-1], colors='k', levels=CCONTOURLEVELS,
               linewidths=0.5, linestyles=':')
    ax.set_title(panel_title_bk25(config, data), size=14)
    add_label(ax, label)
    ax.tick_params(labelsize=12)
    ax.set_box_aspect(1)

    return cf


# Hadley
def plot_hadley():
    """Plots the 2x2 Hadley-like circulation figure."""

    figname = 'hadley'

    config_l, data_l = load_run('output_paper/hadley_ex') 
    config_r, data_r = load_run('output_paper/hadley_adhimex')

    # Time levels 0 to nt are stored, so the halfway time step is at nt // 2
    half_l, half_r = config_l.nt // 2, config_r.nt // 2

    cmap = pc.read('wh-bl-gr-ye-re.cpt').cmap

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True, sharey=True,
                             constrained_layout=True)

    panels = [(axes[0, 0], config_l, data_l, half_l, False, 'mid-time, explicit'),
              (axes[0, 1], config_r, data_r, half_r, False, 'mid-time, AdHImEx'),
              (axes[1, 0], config_l, data_l, 200, True, 'end-time, explicit'),
              (axes[1, 1], config_r, data_r, 20, True, 'end-time, AdHImEx')]

    for ax, config, data, index, show_l2, label in panels:
        cf = plot_panel_hadley(ax, config, data, index, show_l2, label, cmap)

    for ax in axes[1, :]:
        ax.set_xlabel(XLABEL_HADLEY, size=15)
    for ax in axes[:, 0]:
        ax.set_ylabel(YLABEL_HADLEY, size=15)

    finish_figure(fig, cf, axes, figname)


# Constancy
def plot_constancy():
    """Plots the side-by-side constancy preservation figure."""

    figname = 'constancy'

    config_l, data_l = load_run('output_paper/swiftnondiv_constant_adhimexncp')
    config_r, data_r = load_run('output_paper/swiftnondiv_constant_adhimex')

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5), sharey=True,
                             constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.02, wspace=0.02)

    for ax, config, data, label in [
            (axes[0], config_l, data_l, 'without constancy preservation'),
            (axes[1], config_r, data_r, 'with constancy preservation')]:
        cf = plot_panel_cp(ax, config, data, label)

    axes[0].set_ylabel(YLABEL_CP, size=14)

    finish_figure(fig, cf, axes, figname)


# BK25
def plot_bk25():
    """Plots the 3x2 BK25 nondivergent flow figure."""

    figname = 'bk25'

    config_ex, data_ex = load_run('output_paper/swiftnondiv_slotcyl_unif')
    config_ex_fct, data_ex_fct = load_run('output_paper/swiftnondiv_slotcyl_unif_FCT')
    config_ad, data_ad = load_run('output_paper/swiftnondiv_slotcyl_llgp')
    config_ad_fct, data_ad_fct = load_run('output_paper/swiftnondiv_slotcyl_llgp_FCT')

    grid_ex = load_grid('output_paper/swiftnondiv_slotcyl_unif')
    grid_ad = load_grid('output_paper/swiftnondiv_slotcyl_llgp')

    cmap = pc.read('wh-bl-gr-ye-re.cpt').cmap

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True,
                             constrained_layout=True)

    for ax, grid in [(axes[0, 0], grid_ex), (axes[1, 0], grid_ad)]:
        plot_panel_grid(ax, *grid)

    panels = [(axes[0, 1], config_ex, data_ex, 'explicit'),
              (axes[0, 2], config_ex_fct, data_ex_fct, 'explicit, FCT-limited'),
              (axes[1, 1], config_ad, data_ad, 'AdHImEx'),
              (axes[1, 2], config_ad_fct, data_ad_fct, 'AdHImEx, FCT-limited')]

    for ax, config, data, label in panels:
        cf = plot_panel_bk25(ax, config, data, label, cmap)

    axes[0, 0].set_xlim(*XLIM_BK)
    axes[0, 0].set_ylim(*YLIM_BK)

    for ax in axes[1, :]:
        ax.set_xlabel(XLABEL_BK, size=14)
    for ax in axes[:, 0]:
        ax.set_ylabel(YLABEL_BK, size=14)

    finish_figure(fig, cf, axes[:, 1:], figname)


if __name__ == '__main__':

    plot_bk25()
    plot_constancy()
    plot_hadley()

    print("Plotting BK25, constancy, and Hadley done.")
