from __future__ import division
from math import copysign, log
from .core import *
from .mugp import MuGP
from pyde.de import DiffEvol
from .fmodels import models
from scipy.optimize import minimize

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
    classes = 'noise slope jump transit flare'.split()
    npar    = [0, 0, 4, 4, 4]
    
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

        if self.use_gp:
            self.nlnlike_noise   = self.nlnlike_noise_gp
            self.nlnlike_slope   = self.nlnlike_slope_gp
            self.nlnlike_jump    = self.nlnlike_jump_gp
            self.nlnlike_transit = self.nlnlike_transit_gp
            self.nlnlike_flare   = self.nlnlike_flare_gp
        else:
            self.nlnlike_noise   = self.nlnlike_noise_wn
            self.nlnlike_slope   = self.nlnlike_slope_wn
            self.nlnlike_jump    = self.nlnlike_jump_wn
            self.nlnlike_transit = self.nlnlike_transit_wn
            self.nlnlike_flare   = self.nlnlike_flare_wn

            
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
        elif isinstance(jumps, Jump):
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
        
        jamp, jpos = abs(jump.amp), jump.pos

        ## Calculate the maximum log likelihoods
        ## -------------------------------------

        ## Noise
        ## -----
        nlns = self.nlnlike_noise([], cad, flux)

        pvsl = np.polyfit(cad, flux, 2)
        nlsl = self.nlnlike_slope(pvsl, cad, flux)

        ## Jump
        ## ----
        pvjm, nljm = self.fit_jump(jump, cad, flux, use_de, de_npop, de_niter)

        ## Transit
        ## -------
        pvtr, nltr = self.fit_transit(jump, cad, flux, use_de, de_npop, de_niter)

        ## Flare
        ## -----
        pvfl, nlfl = self.fit_flare(jump, cad, flux, use_de, de_npop, de_niter)

        pvs  = [[], pvsl, pvjm, pvtr, pvfl]
        nlns = [nlns, nlsl, nljm, nltr, nlfl]
        bics = [bic(nln, npv, cad.size) for nln,npv in zip(nlns, self.npar)]
        
        cid = np.argmin(bics)
        jump.type = self.classes[cid]
        jump.bics = bics
        jump._pv = pvs[cid]
        jump._sl = sl
        jump._median = local_median

        if jump.type == 'jump':
            jump.pos = pvjm[0]
            jump.amp = pvjm[2]
        
            
    def fit_jump(self, jump, cadence, flux, use_de=False, de_npop=30, de_niter=100):
        jamp, jpos, fstd = abs(jump.amp), jump.pos, flux.std()
        if use_de:
            de = DiffEvol(lambda pv: self.nlnlike_jump( pv, cadence, flux, jump),
                            [[    jpos-2,     jpos+2],  # 0 - center
                            [          1,          3],  # 1 - width
                            [  0.75*jamp,  1.25*jamp],  # 2 - amplitude
                            [ -0.20*fstd,  0.20*fstd],  # 3 - baseline constant
                            [      -1e-3,       1e-3]], # 4 - baseline slope 
                            npop=de_npop)
            de.optimize(de_niter)
            pv0 = de.minimum_location
        else:
            pv0 = [jpos, 2, jamp, fstd, 0]
            
        res = minimize(self.nlnlike_jump, pv0, (cadence, flux, jump), method = 'Nelder-Mead')
        return res.x, res.fun

    
    def fit_transit(self, jump, cadence, flux, use_de=False, de_npop=30, de_niter=100):
        jamp, jpos, fstd = abs(jump.amp), jump.pos, flux.std()
        if use_de:
            de = DiffEvol(lambda pv: self.nlnlike_transit(pv, cadence, flux),
                        [[ 0.8*jamp, 1.2*jamp],  # 0 - transit depth
                        [    jpos-5,   jpos+5],  # 1 - center 
                        [       1.2,      50.],  # 2 - duration
                        [ -0.2*fstd, 0.2*fstd],  # 3 - baseline constant
                        [     -1e-3,     1e-3]], # 4 - baseline slope
                        npop=de_npop)
            de.optimize(de_niter)
            pv0 = de.minimum_location
        else:
            pv0 = [jamp, jpos, 10, fstd, 0]
            
        res = minimize(self.nlnlike_transit, pv0, (cadence, flux), method = 'Nelder-Mead')
        return res.x, res.fun

    
    def fit_flare(self, jump, cadence, flux, use_de=False, de_npop=30, de_niter=100):
        jamp, jpos, fstd = abs(jump.amp), jump.pos, flux.std()
        if use_de:
            de = DiffEvol(lambda pv: self.nlnlike_flare(pv, cadence, flux),
                        [[  jpos-5,   jpos+5],  # 0 - flare start
                        [      1.2,       7.],  # 1 - flare duration
                        [ 0.8*jamp, 1.2*jamp],  # 2 - amplitude
                        [-0.2*fstd, 0.2*fstd],  # 3 - baseline constant
                        [    -1e-3,     1e-3]], # 4 - baseline slope
                        npop=de_npop)
            de.optimize(de_niter)
            pv0 = de.minimum_location
        else:
            pv0 = array([jpos, 2.5, jamp, fstd, 0])
            
        res = minimize(self.nlnlike_flare, pv0, (cadence, flux), method = 'Nelder-Mead')
        return res.x, res.fun


        
    def nlnlike_noise_gp(self, pv, cadence, flux):
        return -self.gp.lnlikelihood(cadence, flux, freeze_k=True)

    def nlnlike_noise_wn(self, pv, cadence, flux):
        return -lnlikelihood(flux, zeros_like(flux), self.hp[2])

    
    def nlnlike_slope_gp(self, pv, cadence, flux):
        return -self.gp.lnlikelihood(cadence, flux-self.m_slope(pv, cadence), freeze_k=True)

    def nlnlike_slope_wn(self, pv, cadence, flux):
        return -lnlikelihood(flux, self.m_slope(pv, cadence), self.hp[2])

    
    def nlnlike_jump_gp(self, pv, cadence, flux, jump):
        if np.any(pv[:2] < 0) or not (0.5 < pv[1] < 3.0) or not (jump.pos-3 <= pv[0] <= jump.pos+3):
            return inf
        return -self.gp.lnlikelihood(cadence, flux-self.m_jump(pv, cadence), freeze_k=True)

    def nlnlike_jump_wn(self, pv, cadence, flux, jump):
        if np.any(pv[:2] < 0) or not (0.5 < pv[1] < 3.0) or not (jump.pos-3 <= pv[0] <= jump.pos+3):
            return inf
        return -lnlikelihood(flux, self.m_jump(pv, cadence), self.hp[2])

    
    def nlnlike_transit_gp(self, pv, cadence, flux):
        if np.any(pv[:-1] <= 0.) or not (self._cd[0]+0.55*pv[2] < pv[1] < self._cd[-1]-0.55*pv[2]) or not (1. < pv[2] < 50.):
            return inf
        return -self.gp.lnlikelihood(cadence, flux-self.m_transit(pv, cadence), freeze_k=True)

    def nlnlike_transit_wn(self, pv, cadence, flux):
        if np.any(pv[:-1] <= 0.) or not (self._cd[0]+0.55*pv[2] < pv[1] < self._cd[-1]-0.55*pv[2]) or not (1. < pv[2] < 50.):
            return inf
        return -lnlikelihood(flux, self.m_transit(pv, cadence), self.hp[2])


    def nlnlike_flare_gp(self, pv, cadence, flux):
        if np.any(pv <= 0.) or not (self._cd[0] < pv[0] < self._cd[-1]) or (pv[1] > 10):
            return inf
        return -self.gp.lnlikelihood(cadence, flux-self.m_flare(pv, cadence), freeze_k=True)

    def nlnlike_flare_wn(self, pv, cadence, flux):
        if np.any(pv <= 0.) or not (self._cd[0] < pv[0] < self._cd[-1]) or (pv[1] > 10):
            return inf
        return -lnlikelihood(flux, self.m_flare(pv, cadence), self.hp[2])

    

    def m_slope(self, pv, cadence):
        """
        0 : slope
        1 : intercept
        """
        return np.poly1d(pv)(cadence)


    def m_jump(self, pv, cadence):
        """
        0 : jump cadence
        1 : jump width
        2 : jump amplitude
        """
        return fm_jump(*pv, cadence=cadence)

    
    def m_transit(self, pv, cadence):
        """
        0 : transit depth
        1 : transit center
        2 : transit duration
        """
        return fm_transit(*pv, cadence=cadence)


    def m_flare(self, pv, cadence):
        """
        0 : start position
        1 : decay length
        2 : flare amplitude
        3 : baseline
        """
        return fm_flare(*pv, cadence=cadence)


