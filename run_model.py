# Call run in src from here to run the model with the specified namelist
# This file is run from the terminal with the config name as an argument
# e.g. python run_model.py <config_name>
# Author: Amber te Winkel
# Email: a.j.tewinkel@pgr.reading.ac.uk


from sys import argv, exit
from os.path import dirname
import logging
from src.config import Config
from src.run import run

def run_model():
    """Overarching function to run the model with the terminal-specified config file."""

    print('Running model...')
    
    # Get the config from the command line argument
    if len(argv) < 2:
        print("Usage: python run_schemes.py <config>")
        exit(1)
    else: 
        configname = argv[1] + '.yml'
        configfile = dirname(__file__) + '/config/' + configname
        config = Config.from_file(configfile)

    # Run the model
    run(config, configfile, configname)

    logging.info('Done')
    print('Done')


if __name__ == '__main__':
    run_model()
