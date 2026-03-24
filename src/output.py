"""This file contains functions to set up the output directories, logging, and store the fields in data files."""

import numpy as np
import os
import logging
import shutil
from src.fields import variables
from datetime import date


def set_up_output_directory(config):
    """Sets up the output directory for storing results."""

    if not os.path.exists('./output/'):
        os.mkdir('./output/')
    if config.outputdir == 'test':
        config.outputdir = './output/test/'
        if not os.path.exists(config.outputdir):
            os.mkdir(config.outputdir)
            os.mkdir(config.outputdir + 'data/')        
    else:
        config.outputdir = f'./output/dated/{date.today().strftime("%Y%m%d")}/' + config.outputdir
        if not os.path.exists(f'./output/dated/'):
            os.mkdir(f'./output/dated/')
        if not os.path.exists(f'./output/dated/{date.today().strftime("%Y%m%d")}/'):
            os.mkdir(f'./output/dated/{date.today().strftime("%Y%m%d")}/')
        if not os.path.exists(config.outputdir):
            os.mkdir(config.outputdir)
            config.outputdir = config.outputdir + '/'
            os.mkdir(config.outputdir + 'data/')        
        else:
            i = 1
            while os.path.exists(config.outputdir + f"_{i}"):
                i += 1
            os.mkdir(config.outputdir + f"_{i}")
            config.outputdir = config.outputdir + f"_{i}" + '/'
            os.mkdir(config.outputdir + 'data/')        


def set_up_plots_directory(outputdir):
    """Sets up the plots directory inside the output directory for storing plots."""

    plots_dir = outputdir + '/plots/'
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
    return plots_dir


def store_config(config, configfile, configname):
    """Stores the config file in the output directory for reference."""

    shutil.copy(configfile, config.outputdir + configname)


def set_up_logging(config, configname):
    """Sets up logging for the simulation."""

    logfile = config.outputdir + 'out.log'
    if os.path.exists(logfile):
        os.remove(logfile)
    print(f'See output file {logfile}')    
    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )    
    logging.info("Script has started")
    logging.info(f'Config: {configname}')
    logging.info(f'Output directory: {config.outputdir}')


def store_output_npy(config, fields):
    """Stores the output fields in .npy files."""

    for field in variables:
        data = getattr(fields, field)
        np.save(config.outputdir + f'data/{field}.npy', data)


def getminmax(field, it):
    """Returns the minimum and maximum values of `field` at time step `it` for logging purposes."""
    return np.min(field[it]), np.max(field[it])