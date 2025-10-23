import numpy as np

fieldnames_nodes_xy =   [ 
                      "tracer",
                      #"density",
                      "xcc",
                      "ycc",
                      "xcf", # x coordinates: cell centers in x, cell faces in y # this definition with no extra face at the right boundary is okay for periodic as it is effectively copied with np.roll(). Otherwise I would need ny + 1 here (and similarly for xfc would need nx + 1)
                      "ycf", # y coordinates: cell centers in x, cell faces in y
                      "xcf",
                      "ycf",
                      "xfc",
                      "yfc",
                      "u", # defined at faces in x, centers in y # defined at [i-1/2,j]
                      "v", # defined at centers in x, faces in y # defined at [i,j-1/2]
                      "flxx", # defined at faces in x, centers in y # defined at [i-1/2,j]          
                      "flxy", # defined at centers in x, faces in y # defined at [i,j-1/2]
                     ]


class FieldContainer:

    dtype = np.float64

    def __init__(self, config):
        for field in fieldnames_nodes_xy:
            setattr(self, field, np.zeros((config.nx, config.ny), dtype=self.dtype))