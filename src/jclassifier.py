from __future__ import division
from math import copysign, log
from .core import *
from .mugp import MuGP
from pyde.de import DiffEvol
from .fmodels import models
from scipy.optimize import minimize
from models import *

import math as mt

fm_transit = models.m_transit
fm_jump = models.m_jump
fm_flare = models.m_flare

ln_two_pi = log(2*mt.pi)

def bic(nln, npv, npt):
    return 2*nln + npv*log(npt)

def lnlikelihood(obs, mod, err):
    n = float(obs.size)
    return -0.5 * (n*ln_two_pi + n*log(err**2) + ((obs-mod)**2).sum()/err**2)

class JumpClassifier(object):
    
    def __init__(self, kdata, hp, window_width=75, kernel='e', use_gp=False):
        assert isinstance(kdata, KData)
        assert isinstance(window_width, int) and window_width > 5
        assert kernel in ['e', 'g']
        assert isinstance(use_gp, bool)

        self.gp = MuGP(kernel=kernel)
        self._kdata = kdata
        self.cadence = self._kdata.cadence
        self.flux = self._kdata.mf_normalized_flux
        self.hp = hp
        self._ww = window_width
        self._hw = self._ww//2
        self.gp.set_parameters(self.hp)
        self.use_gp = use_gp
        
            
    def classify(self, jumps, use_de=False, de_niter=200, de_npop=30):
        """Classify flux discontinuities

        Classifies given flux discontinuities (jumps) as noise, jump, transit, or flare. 

        parameters
        ----------
        jumps    : Jump or a list of Jumps

        de_niter : int, optional
                   number of differential evolution iterations

        de_npop  : int, optional,
                   size of the differential evolution population

        Notes
        -----
        The classification is based on a simple BIC comparison. That is, we fit several
        models to the discontinuities and select the one with the lowest BIC value.

        The ln likelihood space can be pretty nasty even when our models are simple
        (thanks to GPs), so we do a small global optimisation run using Differential
        Evolution (DE) before a local optimisation. The DE run is the most time consuming
        part of the process, but necessary for realiable classification.
        """
        if isinstance(jumps, list):
            [self._classify_single(j, use_de, de_niter, de_npop) for j in jumps]
        elif isinstance(jumps, Discontinuity):
            self._classify_single(jumps, use_de, de_niter, de_npop)
        else:
            raise NotImplementedError('jumps must be a list of jumps or a single jump.')
        
            
    def _classify_single(self, jump, use_de=False, de_niter=150, de_npop=30):
        idx = np.argmin(np.abs(self.cadence-jump.pos))
        self._sl = sl   = np.s_[max(0, idx-self._hw) : min(idx+self._hw, self.cadence.size)]
        self._cd = cad  = self.cadence[sl].copy()
        self._fl = flux = self._kdata.mf_flux[sl].copy()
        local_median = median(flux)
        flux[:] = flux / local_median - 1.

        if self.use_gp:
            self.gp.compute(cad)
        
        models = [M(jump.position, jump.amplitude, cad, flux, self.hp, self.gp, self.use_gp) for M in dmodels]
        bics   = [m.fit(use_de, de_niter, de_npop) for m in models]

        selected_model = models[np.argmin(bics)]
        selected_model._models = models
        selected_model._bics = bics
        selected_model._median = local_median
        return selected_model
