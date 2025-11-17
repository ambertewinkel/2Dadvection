# Call run in src from here to run the model with the specified namelist
# This file is run from the terminal with the config name as an argument
# e.g. python run_model.py <config_name>


from sys import argv, exit
from os.path import dirname

# Add root directory to the path
#root_dir = '../2Dadvection'
#sys.path.insert(0, root_dir)

from src.config import Config
from src.run import run

def run_model():

    # Get the config from the command line argument
    if len(argv) < 2:
        print("Usage: python run_schemes.py <config>")
        exit(1)
    else: 
        config = Config.from_file(dirname(__file__) + '/config/' + argv[1] + '.yml')

    # Run the model
    run(config)

if __name__ == '__main__':
    run_model()
    print('Done')