from __future__ import division

import sys
import math
import numpy as np

import scipy.ndimage as ndi
import scipy.linalg.lapack as lp
import scipy.linalg.blas as bl
import scipy.linalg as sla

from scipy.optimize import fmin, fmin_powell, fmin_cg
from scipy.stats import norm
from scipy.ndimage import median_filter as mf
from scipy.ndimage import label

from numpy import asfortranarray as afa
from numpy import s_, inf, median, zeros_like, array, argmax, where

## Utility functions
## =================
def amax(v):
    return v[np.argmax(np.abs(v))]

class Jump(object):
    def __init__(self, pos, dy, jtype=None):
        self.pos  = int(pos)
        self.amp   = dy
        self.type = jtype
        self._pv  = None
        
    def __str__(self):
        return 'Jump {:4.1f}  {:4.1f} {:}'.format(self.pos, self.amp, self.type or -1)
 
    def __repr__(self):
        return 'Jump({:4.1f}, {:4.1f}, {:})'.format(self.pos, self.amp, self.type)
