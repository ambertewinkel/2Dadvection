
from typing import Any
import numpy as np
import yaml
import dataclasses
from pydantic.dataclasses import dataclass

@dataclass
class Config:

    scheme: str
    
    nx: int
    ny: int

    xmin: float
    xmax: float
    ymin: float
    ymax: float

    BC_x: str
    BC_y: str
    grid_setting: str

    velocity_setting: str

    xc: Any = dataclasses.field(init=False)
    yc: Any = dataclasses.field(init=False)

    dt: float
    nt: int
    starttime: float

    initial_tracer: str
    filename: str
    verbose: bool
    outputdir: str
    animate: bool

    solver: str = 'gcrk_matrixfree' # numpy, gcrk_matrix, gcrk_matrixfree
    constant_tracer: float = 1.
    constant_u: float = 1.
    constant_v: float = 1.

    mref: float = 0.5 # kg kg-1
    mmag: float = 0.5 # kg kg-1
    u0: float = 10. # m s-1    
    T: float = 100. # s, SWIFT period of oscillation

    def __post_init__(self):
        self.dx = (self.xmax - self.xmin) / self.nx
        self.dy = (self.ymax - self.ymin) / self.ny
        
        self.xf = np.linspace(self.xmin, self.xmax, self.nx, endpoint=False)
        self.yf = np.linspace(self.ymin, self.ymax, self.ny, endpoint=False)
        
        self.xc = self.xf + 0.5*self.dx
        self.yc = self.yf + 0.5*self.dy

        self.endtime = self.starttime + self.nt * self.dt


    @classmethod
    def from_file(cls, file):
        """
        Initialize config from `file`
        """
        with open(file) as f:
            config = yaml.safe_load(f)
            return cls(**config)
