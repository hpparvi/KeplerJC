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

try:
    import pandas as pd
    with_pandas = True
except ImportError:
    with_pandas = False

jump_classes = 'noise slope jump transit flare'.split()

## Utility functions
## =================
def amax(v):
    return v[np.argmax(np.abs(v))]

class KData(object):
    def __init__(self, cadence, flux):
        self._cadence = cadence.copy()
        self._flux = flux.copy()
        self._flux_o = flux.copy()
        self._mask = np.isfinite(cadence) & np.isfinite(flux)
        
    @property
    def cadence(self):
        return self._cadence[self._mask]

    @property
    def flux(self):
        return self._flux[self._mask]

    @property
    def normalized_flux(self):
        return self.flux / self.median - 1.

    @property
    def median(self):
        return np.median(self.flux)

    @property
    def original_flux(self):
        return self._flux_o[self._mask]

    
            
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


class JumpSet(list):
    def __init__(self, values=[]):
        if np.all([isinstance(v, Jump) for v in values]):
            super(JumpSet, self).__init__(values)
        else:
            raise TypeError('JumpSet can contain only Jumps')
        
    def append(self, v):
        if isinstance(v, Jump):
            super(JumpSet, self).append(v)
        else:
            raise TypeError('JumpSet can contain only Jumps')

    @property
    def types(self):
        return [j.type for j in self]
            
    @property
    def amplitudes(self):
        return [j.amp for j in self]
    
    @property
    def bics(self):
        if with_pandas:
            return pd.DataFrame([j.bics for j in self], columns=jump_classes)
        else:
            return np.array([j.bics for j in self])
