"""Animates the BK25 tracer fields over one full revolution.

Run as python plot_paper_bk25_animation.py

Produces a gif from the stored .npy files in ./output_paper/, saved to
./output_paper/plots/:

  bk25_animation  the bk25 figure of plot_paper_results.py at every time
                  level, from the initial condition to the end of the
                  revolution.  Columns are the grid, the unlimited result, and
                  the FCT-limited result; rows are the explicit reference (top)
                  and AdHImEx (bottom).  The figure title gives the time, and
                  the panel titles the time level and value range (without the
                  l2 error of the bk25 figure).

The four runs, with every time level stored, are produced by
run_paper_bk25_animation.sh.

Author: Amber te Winkel
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import pycpt as pc
from PIL import Image

from plot_paper_results import (CONTOURLEVELS, CCONTOURLEVELS, XLABEL_BK,
                                YLABEL_BK, XLIM_BK, YLIM_BK, load_run,
                                load_grid, add_label, plot_panel_grid)


FPS = 25 # time levels per second of animation
DPI = 100


def time_title(config, n):
    """Builds the figure title: the time at time level n."""

    return f'$t = {config.starttime + n*config.dt:.1f}$ s'


def panel_title(data, n):
    """Builds the title for one panel: the value range at time level n."""

    field = data['tracer'][n]
    minval, maxval = np.min(field), np.max(field)

    return f'$\\Psi^{{ {n} }} \\in [{minval:.2f},{maxval:.2f}]$'


def plot_frame(ax, data, n, cmap):
    """Draws the tracer at time level n on one panel and returns the contour sets, to be removed for the next frame."""

    x, y = data['xcc'], data['ycc']

    cf = ax.contourf(x, y, data['tracer'][n], cmap=cmap,
                     levels=CONTOURLEVELS, extend='both')
    # Ccc[n-1] is the Courant number of the time step to time level n (Ccc[0] for the initial condition)
    cc = ax.contour(x, y, data['Ccc'][max(n-1, 0)], colors='k',
                    levels=CCONTOURLEVELS, linewidths=0.5, linestyles=':')
    ax.set_title(panel_title(data, n), size=14)

    return cf, cc


def animate_bk25():
    """Animates the 3x2 BK25 nondivergent flow figure over all time levels."""

    figname = 'bk25_animation'

    config_ex, data_ex = load_run('output_paper/swiftnondiv_slotcyl_unif')
    config_ex_fct, data_ex_fct = load_run('output_paper/swiftnondiv_slotcyl_unif_FCT')
    config_ad, data_ad = load_run('output_paper/swiftnondiv_slotcyl_llgp')
    config_ad_fct, data_ad_fct = load_run('output_paper/swiftnondiv_slotcyl_llgp_FCT')

    grid_ex = load_grid('output_paper/swiftnondiv_slotcyl_unif')
    grid_ad = load_grid('output_paper/swiftnondiv_slotcyl_llgp')

    # Each frame shows the same time level in all four tracer panels
    configs = [config_ex, config_ex_fct, config_ad, config_ad_fct]
    if len({(c.starttime, c.dt, c.nt) for c in configs}) > 1:
        raise ValueError('The four runs need the same starttime, dt, and nt to be animated together.')

    cmap = pc.read('wh-bl-gr-ye-re.cpt').cmap

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), dpi=DPI, sharex=True,
                             sharey=True, constrained_layout=True)

    for ax, grid in [(axes[0, 0], grid_ex), (axes[1, 0], grid_ad)]:
        plot_panel_grid(ax, *grid)

    panels = [(axes[0, 1], data_ex, 'explicit'),
              (axes[0, 2], data_ex_fct, 'explicit, FCT-limited'),
              (axes[1, 1], data_ad, 'AdHImEx'),
              (axes[1, 2], data_ad_fct, 'AdHImEx, FCT-limited')]

    for ax, data, label in panels:
        add_label(ax, label)
        ax.tick_params(labelsize=12)
        ax.set_box_aspect(1)
    contours = [plot_frame(ax, data, 0, cmap) for ax, data, _ in panels]

    axes[0, 0].set_xlim(*XLIM_BK)
    axes[0, 0].set_ylim(*YLIM_BK)

    for ax in axes[1, :]:
        ax.set_xlabel(XLABEL_BK, size=14)
    for ax in axes[:, 0]:
        ax.set_ylabel(YLABEL_BK, size=14)

    suptitle = fig.suptitle(time_title(config_ex, 0), size=15)
    cbar = fig.colorbar(contours[0][0], ax=axes[:, 1:], fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=12)

    # Fix the layout of the first frame, so that the panels don't shift between frames
    fig.canvas.draw()
    fig.set_layout_engine('none')

    frames = []
    for n in range(config_ex.nt + 1):
        for i, (ax, data, _) in enumerate(panels):
            for cs in contours[i]:
                cs.remove()
            contours[i] = plot_frame(ax, data, n, cmap)
        suptitle.set_text(time_title(config_ex, n))
        fig.canvas.draw()
        # Stored in palette mode (1 byte per pixel) rather than RGBA, to keep the memory use of all frames low
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert('RGB').quantize())
        if n % 100 == 0:
            print(f'Time level {n} of {config_ex.nt}')
    plt.close(fig)

    plots_dir = './output_paper/plots/'
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
    filename = f'{plots_dir}{figname}.gif'
    frames[0].save(filename, save_all=True, append_images=frames[1:],
                   duration=int(1000/FPS), loop=0)

    print(f'Saved {filename}')


if __name__ == '__main__':

    animate_bk25()

    print("Animating BK25 done.")
