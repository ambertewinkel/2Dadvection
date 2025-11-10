from logging import config
import os 
import matplotlib.pyplot as plt
import imageio
import numpy as np



### copied from 1D advectionschemes code
def design_figure(filename, title, xlabel, ylabel, xlim1, xlim2, bool_ylim = False, ylim1=0.0, ylim2=0.0, ax=plt):#, legend_lines=[]):
    
    # Create a single legend for both axes
    #lns = line_cEx + line_cAdImEx + line_thetaEx + line_thetaAdImEx
    if ax == plt:
        if bool_ylim == True: ax.ylim(ylim1, ylim2)
        ax.xlim(xlim1, xlim2)
        ax.xlabel(xlabel)
        ax.ylabel(ylabel)
        ax.title(title)
        ax.legend()
    else:
        if bool_ylim == True: ax.set_ylim(ylim1, ylim2)
        ax.set_xlim(xlim1, xlim2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        #labs = [l.get_label() for l in legend_lines]
        #ax.legend(legend_lines, labs, loc='best')
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()

### copied from 1D advectionschemes code
def create_animation_from_data(fields, nfields, analytic, field_in, nt, dt, dxc, xc, xf, uf, u_setting, outputdir, plot_args, xmax, ymin=-0.1, ymax=None, plot_Courant=False):
    """This function creates an animation from a given data file of a single scheme. The input is a 2D field of a single scheme (shape = 1d time x 1d space), analytic solution (shape = 1d time x 1d space), nt, dx in the centers (dxc; shape 1d space) and ....
    This is a function that is to be called from other files, not from produce_standalone_animation()."""

    # Directory to put animation and subdirectory for plots in
    plotdir = outputdir + 'plots/'
    os.mkdir(plotdir)
    if ymax is None:
        ymax = 1.1
    if plot_Courant: uf_plot = np.concatenate((np.full((1,len(xf)), np.nan), uf), axis=0)

    # Timestepping loop to create plots and save filenames
    filenames, images = [], []
    for it in range(nt+1):   
        fig, ax1 = plt.subplots(figsize=(7,4))
        # Plot each timestep in a figure and save in the plots subdirectory
        #lns = []
        if plot_Courant:
            #uf = np.concatenate((uf, np.full(len(xf), np.nan)), axis=0)
            ax2 = ax1.twinx()
            ax2.set_ylim(0., 6.)#2.)
            ax2.set_ylabel(f'$C$ at faces at $n_t$ = {np.where(it-0.5>0, it-0.5, None)}', color='purple')
        #    #lns.append(ax2.plot(xf, uf[it], linestyle='--', color='purple'))
            ax2.plot(xf, uf_plot[it]*dt/dxc, linestyle='--', color='purple')
            ax2.axhline(1., linestyle=':', color='grey')
        #lns.append(ax1.plot(xc, field_in, label='Initial', linestyle='-', color='grey'))
        ax1.plot(xc, field_in, label='Initial', linestyle='-', color='grey')
        #lns.append(ax1.plot(xc, analytic[it], label='Analytic', linestyle='-', color='k'))
        if u_setting == 'constant': ax1.plot(xc, analytic[it], label='Analytic', linestyle='-', color='k')
        for si in range(nfields):   
            field = fields[si]        
            #lns.append(ax1.plot(xc, field[it], **plot_args[si]) )
            ax1.plot(xc, field[it], **plot_args[si]) 
        #print(type(lns))
        design_figure(f'{plotdir}timestep_{it}.png', f'$\\Psi$ at t={it*dt:.2f}', \
                        'x', '$\\Psi$', 0., xmax, True, ymin, ymax, ax=ax1)#, legend_lines=lns)
        plt.close()
        filenames.append(f'{plotdir}timestep_{it}.png')

    # Create animation from plots in the plots subdirectory
    for filename in filenames:
        images.append(imageio.imread(filename))
    anim_filename = f'{outputdir}animation.gif'
    imageio.mimsave(anim_filename, images, duration=500)

    # Remove .png files used to create the animation
    for filename in filenames:
        os.remove(filename)
    os.rmdir(plotdir)









def create_animation(config, fields):
    """This function creates an animation from a given data file of a single scheme. The input is a 2D field of a single scheme (shape = 1d time x 1d space), analytic solution (shape = 1d time x 1d space), nt, dx in the centers (dxc; shape 1d space) and ....
    This is a function that is to be called from other files, not from produce_standalone_animation()."""

    ## Directory to put animation and subdirectory for plots in
    #plotdir = outputdir + 'plots/'
    #os.mkdir(plotdir)
    #if ymax is None:
    #    ymax = 1.1
    #if plot_Courant: uf_plot = np.concatenate((np.full((1,len(xf)), np.nan), uf), axis=0)
#
    ## Timestepping loop to create plots and save filenames
    #filenames, images = [], []
    #for it in range(nt+1):   
    #    fig, ax1 = plt.subplots(figsize=(7,4))
    #    # Plot each timestep in a figure and save in the plots subdirectory
    #    #lns = []
    #    if plot_Courant:
    #        #uf = np.concatenate((uf, np.full(len(xf), np.nan)), axis=0)
    #        ax2 = ax1.twinx()
    #        ax2.set_ylim(0., 6.)#2.)
    #        ax2.set_ylabel(f'$C$ at faces at $n_t$ = {np.where(it-0.5>0, it-0.5, None)}', color='purple')
    #    #    #lns.append(ax2.plot(xf, uf[it], linestyle='--', color='purple'))
    #        ax2.plot(xf, uf_plot[it]*dt/dxc, linestyle='--', color='purple')
    #        ax2.axhline(1., linestyle=':', color='grey')
    #    #lns.append(ax1.plot(xc, field_in, label='Initial', linestyle='-', color='grey'))
    #    ax1.plot(xc, field_in, label='Initial', linestyle='-', color='grey')
    #    #lns.append(ax1.plot(xc, analytic[it], label='Analytic', linestyle='-', color='k'))
    #    if u_setting == 'constant': ax1.plot(xc, analytic[it], label='Analytic', linestyle='-', color='k')
    #    for si in range(nfields):   
    #        field = fields[si]        
    #        #lns.append(ax1.plot(xc, field[it], **plot_args[si]) )
    #        ax1.plot(xc, field[it], **plot_args[si]) 
    #    #print(type(lns))
    #    design_figure(f'{plotdir}timestep_{it}.png', f'$\\Psi$ at t={it*dt:.2f}', \
    #                    'x', '$\\Psi$', 0., xmax, True, ymin, ymax, ax=ax1)#, legend_lines=lns)
    #    plt.close()
    #    filenames.append(f'{plotdir}timestep_{it}.png')

    # Create animation from plots in the plots subdirectory
    images = []
    for it in range((config.nt)+1):
        images.append(imageio.imread("./output/" + config.outputdir + f'/plots/nt{it}.png'))
    anim_filename = f'./output/{config.outputdir}/animation.gif'
    imageio.mimsave(anim_filename, images, duration=60)

    # Remove .png files used to create the animation
    for it in range((config.nt)+1):        
        os.remove("./output/" + config.outputdir + f'/plots/nt{it}.png')    
        
    images = []
    for it in range((config.nt)):
        images.append(imageio.imread("./output/" + config.outputdir + f'/plots/Cnt{it+1}.png'))
    anim_filename = f'./output/{config.outputdir}/Courant_animation.gif'
    imageio.mimsave(anim_filename, images, duration=60)

    # Remove .png files used to create the animation
    for it in range((config.nt)):        
        os.remove("./output/" + config.outputdir + f'/plots/Cnt{it+1}.png')
