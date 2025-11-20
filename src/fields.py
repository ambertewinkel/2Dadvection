import numpy as np


grid_nodes_xy = [
                      "xcc",
                      "ycc",
                      "xcf", # x coordinates: cell centers in x, cell faces in y # this definition with no extra face at the right boundary is okay for periodic as it is effectively copied with np.roll(). Otherwise I would need ny + 1 here (and similarly for xfc would need nx + 1)
                      "ycf", # y coordinates: cell centers in x, cell faces in y
                      "xfc",
                      "yfc",
]

grid_nodes_xy_boundaries = [
                      "xffb", # x coordinates: cell corners (faces in x, faces in y) -> i,j defined at [i-1/2,j-1/2], i.e., bottom left corner of cell, including right and top boundaries
                      "yffb", # y coordinates: cell corners (faces in x, faces in y) -> i,j defined at [i-1/2,j-1/2], i.e., bottom left corner of cell, including right and top boundaries
]


fieldnames_nodes_xy_init = [
                      "tracer",
                      #"density",
]

fieldnames_nodes_xy =   [ 
                      "u", # defined at faces in x, centers in y # defined at [i-1/2,j]
                      "v", # defined at centers in x, faces in y # defined at [i,j-1/2]
                      "Ccc", # Courant number at cell centers
                      "thetacc", # implicitness at cell centers
                      "thetafc", # implicitness at faces in x, centers in y
                      "thetacf",  # implicitness at centers in x, faces in y
                     ]

fieldnames_nodes_xy_boundaries =   [
                      "psi", # streamfunction at cell corners # defined at [i-1/2,j-1/2], also including right and top boundaries for easy differentiation to find velocities
                     ]

variables = grid_nodes_xy + grid_nodes_xy_boundaries + fieldnames_nodes_xy_init + fieldnames_nodes_xy + fieldnames_nodes_xy_boundaries
class FieldContainer:

    dtype = np.float64

    def __init__(self, config):
        for field in grid_nodes_xy:
            setattr(self, field, np.zeros((config.nx, config.ny), dtype=self.dtype))
        for field in grid_nodes_xy_boundaries:
            setattr(self, field, np.zeros((config.nx+1, config.ny+1), dtype=self.dtype))
        for field in fieldnames_nodes_xy_init:
            setattr(self, field, np.zeros((config.nt+1, config.nx, config.ny), dtype=self.dtype))
        for field in fieldnames_nodes_xy:
            setattr(self, field, np.zeros((config.nt, config.nx, config.ny), dtype=self.dtype))        
        for field in fieldnames_nodes_xy_boundaries:
            setattr(self, field, np.zeros((config.nt, config.nx+1, config.ny+1), dtype=self.dtype))