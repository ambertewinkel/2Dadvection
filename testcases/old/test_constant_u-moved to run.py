# Set up the test with constant u velocity. Scheme definition and settings will be in namelist. This sets up the velocity over time.

import sys

# Add root directory to the path
root_dir = '../../2Dadvection'
sys.path.insert(0, root_dir)

from src.config import Config
from src.fields import FieldContainer
from src.grid import grid_coordinates
from src.run import run_scheme


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_schemes.py <config>")
        sys.exit(1)
    else: 
        config = Config.from_file(root_dir + '/config/' + sys.argv[1] + '.yml')

    # Set up fields
    fields = FieldContainer(config)

    # Set up grid
    grid_coordinates(config, fields)

    # Set up initial condition



    # Run the scheme
    result = run_scheme(config, fields)#, **kwargs)

    # Store result
    



if __name__ == '__main__':
    main()
    print('Done')
