import numpy as np

fieldnames_nodes_xy =   [ 
                      "tracer",
                      #"density",
                      "xcc",
                      "ycc",
                      "xcf", # x coordinates: cell centers in x, cell faces in y # this definition with no extra face at the right boundary is okay for periodic as it is effectively copied with np.roll(). Otherwise I would need ny + 1 here (and similarly for xfc would need nx + 1)
                      "ycf", # y coordinates: cell centers in x, cell faces in y
                      "xfc",
                      "yfc",
                      "u", # defined at faces in x, centers in y # defined at [i-1/2,j]
                      "v", # defined at centers in x, faces in y # defined at [i,j-1/2]
                      "flxx", # defined at faces in x, centers in y # defined at [i-1/2,j]          
                      "flxy", # defined at centers in x, faces in y # defined at [i,j-1/2]
                      "Ccc", # Courant number at cell centers
                      "Cfc", # Courant number at faces in x, centers in y
                      "Ccf", # Courant number at centers in x, faces in y
                      "thetacc", # implicitness at cell centers
                      "thetafc", # implicitness at faces in x, centers in y
                      "thetacf",  # implicitness at centers in x, faces in y
                      "maxCcc", # temporal max Courant number at cell centers
                      "maxthetacc", # temporal max implicitness at cell centers
                     ]

fieldnames_nodes_xy_boundaries =   [
                      "psi", # streamfunction at cell corners # defined at [i-1/2,j-1/2], also including right and top boundaries for easy differentiation to find velocities
                      "xffb", # x coordinates: cell corners (faces in x, faces in y) -> i,j defined at [i-1/2,j-1/2], i.e., bottom left corner of cell, including right and top boundaries
                      "yffb", # y coordinates: cell corners (faces in x, faces in y) -> i,j defined at [i-1/2,j-1/2], i.e., bottom left corner of cell, including right and top boundaries
                     ]

class FieldContainer:

    dtype = np.float64

    def __init__(self, config):
        for field in fieldnames_nodes_xy:
            setattr(self, field, np.zeros((config.nx, config.ny), dtype=self.dtype))        
        for field in fieldnames_nodes_xy_boundaries:
            setattr(self, field, np.zeros((config.nx+1, config.ny+1), dtype=self.dtype))