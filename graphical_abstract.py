"""Builds the graphical abstract for the AdHImEx paper.

Run as python plot_graphical_abstract.py

Three columns, built from the BK25 nondivergent flow results in ./output_paper/:

  left    the explicit reference tracer field at the final time
  middle  the two grids, the uniform grid on top and the locally refined grid
          below, together spanning the height of the tracer panels.  Only
          every GRIDSTRIDE'th line is drawn, plus the last one, so that the
          meshes stay legible without leaving a gap at the top and right.
  right   the AdHImEx tracer field at the final time

Each grid is joined to the tracer field it produced by an arrow.  No axes
numbers and no colorbars: the annotations carry the quantities.

Author: Amber te Winkel
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import pycpt as pc
from matplotlib.patches import ConnectionPatch
from matplotlib.transforms import blended_transform_factory

from error import l2norm
from plot_paper_results import CONTOURLEVELS, draw_grid_lines, load_grid, load_run


# Output directories
DIR_EX = 'output_paper/swiftnondiv_slotcyl_unif'
DIR_AD = 'output_paper/swiftnondiv_slotcyl_llgp'

# Domain
XLIM, YLIM = (-500., 500.), (-500., 500.)

# Only every GRIDSTRIDE'th grid line is drawn, to keep the meshes legible
GRIDSTRIDE = 5
GRIDLINEWIDTH = 0.5

# Labels above and below the grids
GRIDLABELSIZE = 11
GRIDLABELPAD = 0.04

# Arrows joining each grid to its tracer field
COLOUR_ARROW = 'black'
ARROWLW, ARROWSCALE = 4., 32.
ARROWSTART, ARROWEND = 0.07, 0.05
ARROWHEIGHT = 0.5      # height on the grid axes the arrow runs at

# In-panel annotation
LABELSIZE, L2SIZE = 19, 16
LABELCOLOUR = 'black'

# Textbox between the two grids
DTTEXT = r'$\Delta t$ identical'
DTSIZE = 16


def thin(faces):
    """Returns every GRIDSTRIDE'th face position, always keeping the last one."""

    kept = faces[::GRIDSTRIDE]
    if kept[-1] != faces[-1]:
        kept = np.append(kept, faces[-1])

    return kept


def plot_grid(ax, xf, yf, label, above=True):
    """Draws the whole grid, thinned to every GRIDSTRIDE'th line.

    The label sits just above the panel when above is True, just below it
    otherwise.
    """

    draw_grid_lines(ax, thin(xf), thin(yf), GRIDLINEWIDTH)

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(1)

    y, va = (1. + GRIDLABELPAD, 'bottom') if above else (-GRIDLABELPAD, 'top')
    ax.text(0.5, y, label, transform=ax.transAxes, ha='center', va=va,
            size=GRIDLABELSIZE, style='italic',
            bbox=dict(boxstyle='square', fc='none', ec='none'))


def plot_tracer(ax, data, label, cmap):
    """Draws one tracer panel, labelled inside with its name and l2 norm."""

    x, y = data['xcc'], data['ycc']
    field = data['tracer'][-1]

    ax.contourf(x, y, field, cmap=cmap, levels=CONTOURLEVELS, extend='both')

    l2_error = l2norm(field, data['tracer'][0], data['dxcc'] * data['dycc'])
    ax.text(0.5, 0.96, label, transform=ax.transAxes, ha='center', va='top',
            size=LABELSIZE, color=LABELCOLOUR, weight='bold')
    ax.text(0.96, 0.04, f'$\\ell_2 =${l2_error:.4f}', transform=ax.transAxes,
            ha='right', va='bottom', size=L2SIZE, color=LABELCOLOUR)

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(1)


def arrow_between(fig, ax_from, ax_to, side):
    """Draws a horizontal arrow from one edge of ax_from to the facing edge of ax_to.

    The far end takes its x from ax_to but its height from ax_from, so the
    arrow stays level even though the two panels have different centres.
    """

    if side == 'left':      # out of ax_from's left edge, into ax_to's right
        start, end = -1.8*ARROWSTART, 1. + 0.2*ARROWEND
    else:                   # out of ax_from's right edge, into ax_to's left
        start, end = 1. + 1.8*ARROWSTART, -0.2*ARROWEND
    level = blended_transform_factory(ax_to.transAxes, ax_from.transAxes)



    #if side == 'left':      # out of ax_from's left edge, into ax_to's right
    #    start, end = -ARROWSTART, 1. + ARROWEND
    #else:                   # out of ax_from's right edge, into ax_to's left
    #    start, end = 1. + ARROWSTART, -ARROWEND
#



    fig.add_artist(ConnectionPatch(
        xyA=(start, ARROWHEIGHT), coordsA=ax_from.transAxes,
        xyB=(end, ARROWHEIGHT), coordsB=level,
        arrowstyle='-|>', mutation_scale=ARROWSCALE, lw=ARROWLW,
        color=COLOUR_ARROW))


def plot_graphical_abstract():
    """Assembles and saves the graphical abstract."""

    figname = 'graphical_abstract'

    _, data_ex = load_run(DIR_EX)
    _, data_ad = load_run(DIR_AD)
    grid_ex = load_grid(DIR_EX)
    grid_ad = load_grid(DIR_AD)

    cmap = pc.read('wh-bl-gr-ye-re.cpt').cmap

    fig = plt.figure(figsize=(11., 4.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1., 0.35, 1.],
                          wspace=0.15, hspace=0.05)

    ax_ex = fig.add_subplot(gs[:, 0])
    ax_grid_unif = fig.add_subplot(gs[0, 1])
    ax_grid_nonunif = fig.add_subplot(gs[1, 1])
    ax_ad = fig.add_subplot(gs[:, 2])

    plot_tracer(ax_ex, data_ex, 'explicit reference', cmap)
    plot_grid(ax_grid_unif, *grid_ex, 'uniform grid')
    plot_grid(ax_grid_nonunif, *grid_ad, 'nonuniform grid', above=False)
    plot_tracer(ax_ad, data_ad, 'AdHImEx', cmap)

    # Each grid out to the tracer field it produced
    arrow_between(fig, ax_grid_unif, ax_ex, 'left')
    arrow_between(fig, ax_grid_nonunif, ax_ad, 'right')

    # The two runs differ only in their grid: centred on the gap between them
    top, bottom = ax_grid_unif.get_position(), ax_grid_nonunif.get_position()
    fig.text(0.5 * (top.x0 + top.x1), 0.5 * (top.y0 + bottom.y1), DTTEXT,
             ha='center', va='center', size=DTSIZE,
             bbox=dict(boxstyle='square', fc='white', ec='none'))

    plots_dir = './output_paper/plots/'
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(f'{plots_dir}{figname}.{ext}', dpi=300,
                    bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    plot_graphical_abstract()
    print("Plotting done.")