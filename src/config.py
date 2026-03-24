
from typing import Any
import numpy as np
import yaml
from pydantic.dataclasses import dataclass

@dataclass
class Config():

    scheme: str
    
    nx: int
    ny: int

    xmin: float
    xmax: float
    ymin: float
    ymax: float

    grid_x: str # e.g., 'uniform' or 'cosine' or 'lowerleft' (and other options)
    grid_y: str # e.g., 'uniform' or 'cosine' or 'lowerleft' (and other options)

    BC_x: str # e.g., 'periodic' or 'noflux'
    BC_y: str # e.g., 'periodic' or 'noflux'

    velocity_setting: str

    dt: float
    nt: int
    starttime: float

    initial_tracer: str
    thirdordermatrix: bool = False
    FCT_2pass: bool = False
    FCT: bool = False
    tracermin: float = None
    tracermax: float = None

    timing: bool = False
    print_error: bool = False
    verbose: bool = False
    outputdir: str = 'test'

    solver: str = 'gmresm' #gcrk or gmresm --- both matrixfree (gmresm kinda)
    constant_tracer: float = 1.
    constant_u: float = 1.
    constant_v: float = 1.
    nondivergent: bool = None

    # SWIFT defaults (u0 and T also used for other testcases, with potentially different values)
    mref: float = 0.5 # kg kg-1
    mmag: float = 0.5 # kg kg-1
    u0: float = 10. # m s-1    
    T: float = 100. # s

    # Jet testcase defaults
    ujet_max: float = 1. # maximum jet velocity magnitude


    def __post_init__(self):

        # Jet testcase defaults
        self.ujet_a: float = self.ymax/3.
        self.ujet_b: float = 2.*self.ymax/3.
        
        self.endtime = self.starttime + self.nt * self.dt

        if self.nondivergent is False:
            self.factordiv = 0.5
        elif self.nondivergent is True:
            self.factordiv = 1.
        elif self.nondivergent is None:
            print('Abort: nondivergent is not set - please specify whether the winds are nondivergent (True) or not (False).')

    @classmethod
    def from_file(cls, file):
        """
        Initialize config from `file`
        """
        with open(file) as f:
            config = yaml.safe_load(f)
            return cls(**config)
