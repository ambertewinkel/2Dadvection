import os 
import imageio


def create_animation(config, plots_dir, anim_dir, fieldname):
    """This function creates an animation from a given data file of a single scheme. The input is a 2D field of a single scheme (shape = 1d time x 1d space), analytic solution (shape = 1d time x 1d space), nt, dx in the centers (dxc; shape 1d space) and ....
    This is a function that is to be called from other files, not from produce_standalone_animation()."""

    # Create animation from plots in the plots subdirectory
    images = []
    if fieldname == 'tracer':
        for it in range(config.nt+1):
            images.append(imageio.imread(f'{plots_dir}{fieldname}_nt{it}.png'))
        anim_filename = f'{anim_dir}{fieldname}.gif'
        imageio.mimsave(anim_filename, images, duration=60)

        # Remove .png files used to create the animation
        for it in range(config.nt+1):        
            os.remove(f'{plots_dir}{fieldname}_nt{it}.png')        
    else:
        # Create animation from plots
        for it in range(1, config.nt+1):
            images.append(imageio.imread(f'{plots_dir}{fieldname}_nt{it}.png'))
        anim_filename = f'{anim_dir}{fieldname}.gif'
        imageio.mimsave(anim_filename, images, duration=60)

        # Remove .png files used to create the animation
        for it in range(1,config.nt+1):        
            os.remove(f'{plots_dir}{fieldname}_nt{it}.png')    
