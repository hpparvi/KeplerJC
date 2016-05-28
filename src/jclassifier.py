from __future__ import division
from math import copysign, log
from core import *
from mugp import MuGP

def bic(nln, npv, npt):
    return 2*nln + npv*log(npt)


class JumpClassifier(object):
    classes = 'noise jump transit flare'.split()
    npar    = [0, 1, 3, 3]
    
    def __init__(self, cadence, flux, hp, window_width=75, kernel='e'):
        self.gp = MuGP(kernel=kernel)
        m = np.isfinite(cadence) & np.isfinite(flux)
        self.cadence = cadence[m]
        self.flux = flux[m] / np.median(flux[m]) - 1.
        self.hp = hp
        self._ww = window_width
        self._hw = self._ww//2
        self.gp.set_parameters(self.hp)


    def classify(self, jumps):
        if isinstance(jumps, list):
            [self._classify_single(j) for j in jumps]
        elif isinstance(jumps, Jump):
            self._classify_single(jumps)
        else:
            raise NotImplementedError('jumps must be a list of jumps or a single jump.')
        
            
    def _classify_single(self, jump):
        idx = np.argmin(np.abs(self.cadence-jump.pos))
        self._sl = sl   = np.s_[max(0, idx-self._hw) : min(idx+self._hw, self.cadence.size)]
        self._cd = cad  = self.cadence[sl].copy()
        self._fl = flux = self.flux[sl].copy()  + 1.
        flux[:] = flux / median(flux) - 1.
        self.gp.compute(cad)

        ## Calculate the maximum log likelihoods
        ## -------------------------------------
        nlns = self.nlnlike_noise([], cad, flux)
        nljm = self.nlnlike_jump([jump.pos], cad, flux)
        pvtr, nltr, _,_,_,_ = fmin_powell(self.nlnlike_transit,
                              [abs(jump.amp), jump.pos + copysign(1, -jump.amp)*5, 10],
                              (cad, flux), disp=False, full_output=True)
        pvfl, nlfl, _,_,_,_ = fmin_powell(self.nlnlike_flare,
                              [abs(jump.amp), jump.pos, 0.5],
                              (cad, flux), disp=False, full_output=True)
        pvs  = [pvtr, pvfl]
        nlns = [nlns, nljm, nltr, nlfl]
        bics = [bic(nln, npv, cad.size) for nln,npv in zip(nlns, self.npar)]

        cid = np.argmin(bics)
        jump.type = self.classes[cid]
        jump.bics = bics
        if jump.type in self.classes[2:]:
            jump._pv = pvs[cid-2]
        
    def nlnlike_noise(self, pv, cadence, flux):
        return -self.gp.lnlikelihood(cadence, flux)

    
    def nlnlike_jump(self, pv, cadence, flux):
        return -self.gp.lnlikelihood(cadence, flux, pv[0])

    def nlnlike_transit(self, pv, cadence, flux):
        if any(pv <= 0.) or not (self._cd[0] < pv[1] < self._cd[-1]) or (pv[2]>50):
            return inf
        return -self.gp.lnlikelihood(cadence, flux-self.m_transit(pv, cadence))

    def nlnlike_flare(self, pv, cadence, flux):
        if any(pv <= 0.) or not (self._cd[0] < pv[1] < self._cd[-1]) or (pv[2] > 10):
            return inf
        return -self.gp.lnlikelihood(cadence, flux-self.m_flare(pv, cadence))
    
    
    def m_transit(self, pv, cadence):
        """
        0 : transit depth
        1 : transit center
        2 : transit duration
        """
        if np.any(pv < 0):
            return None

        hdur = 0.5*pv[2]
        model = np.zeros(cadence.size, np.float64)
        cmask = (cadence-pv[1] > -hdur) & (cadence-pv[1] < hdur)
        model[cmask] = -pv[0]
        return model


    def m_flare(self, pv, cadence):
        """
        0 : flare amplitude
        1 : start position
        2 : decay length
        """
        if np.any(pv < 0):
            return None

        model = np.zeros(cadence.size, np.float64)
        cmask = cadence >= pv[1]
        model[cmask] = pv[0]*np.exp(-(cadence[cmask]-pv[1])/pv[2])
        return model

