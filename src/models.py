from __future__ import division
from math import copysign, log, pi
from scipy.optimize import minimize
from scipy.signal import medfilt as mf
from pyde.de import DiffEvol
from .core import *
from .mugp import MuGP
from .fmodels import models as fm

from numpy import ndarray, array, zeros

ln_two_pi = log(2*pi)

def bic(nln, npv, npt):
    return 2*nln + npv*log(npt)

def lnlikelihood(obs, mod, err):
    n = float(obs.size)
    return -0.5 * (n*ln_two_pi + n*log(err**2) + ((obs-mod)**2).sum()/err**2)


class JumpSet(list):
    def __init__(self, values=[]):
        if isinstance(values, Discontinuity):
            super(JumpSet, self).__init__([values])
        elif isinstance(values, list) and all([isinstance(v, Discontinuity) for v in values]):
            super(JumpSet, self).__init__(values)
        else:
            raise TypeError('JumpSet can contain only Jumps')
        
    def append(self, v):
        if isinstance(v, Discontinuity):
            super(JumpSet, self).append(v)
        else:
            raise TypeError('JumpSet can contain only Jumps')

    @property
    def types(self):
        return [j.name for j in self]
            
    @property
    def amplitudes(self):
        return [j.amplitude for j in self]
    
    @property
    def bics(self):
        if with_pandas:
            return pd.DataFrame([j.bics for j in self], columns=jump_classes)
        else:
            return np.array([j.bics for j in self])


class DiscontinuityType(object):
    name   = ''
    pnames = []
    npar   = len(pnames)
    
    def __init__(self, discontinuity):
        self.discontinuity = self.d = discontinuity
        self.best_fit_pv = None
        self.best_fit_model = None
        self.bic = None
        self._optimization_result = None
            
#    def __str__(self):
#        return '{:s} {:4.1f}  {:4.1f}'.format(self.name, self.position, self.amplitude)

#    def __repr__(self):
#        return '{:s}({:4.1f}, {:4.1f})'.format(self.name, self.position, self.amplitude)

            
    def nlnlike_wn(self, pv):
        if not self.is_inside_bounds(pv):
            return inf
        else:
            return -lnlikelihood(self.d.flux, self.model(pv), self.d.wn_estimate)

        
    def nlnlike_gp(self, pv):
        if not self.is_inside_bounds(pv):
            return inf
        else:
            return -self.d.gp.lnlikelihood(self.d.cadence, self.d.flux-self.model(pv), freeze_k=True)    

        
    def fit(self, use_de=False, de_npop=30, de_niter=100, method='Powell'):
        jamp, jpos, fstd = self.d.amplitude, self.d.position, self.d.flux.std()
        nlnlike = self.nlnlike_gp if self.d.use_gp else self.nlnlike_wn
        if use_de:
            self._de = DiffEvol(nlnlike, self._de_bounds(jamp, jpos, fstd), npop=de_npop)
            self._de.optimize(de_niter)
            pv0 = self._de.minimum_location
        else:
            pv0 = self._pv0(jamp, jpos, fstd)
            
        self._optimization_result = r = minimize(nlnlike, pv0, method=method)
        self.best_fit_pv = r.x
        self.best_fit_model = self.model(r.x)
        self.bic = self.c_bic(r.fun)
        return self.bic

    
    def model(self, pv):
        raise NotImplementedError

    
    def c_bic(self, nln):
        return 2*nln + self.npar*log(self.d.npt)

    
class UnclassifiedDiscontinuity(DiscontinuityType):
    name   = 'Unclassified'
    pnames = []
    npar   = len(pnames)

    
    def fit(self, *nargs, **kwargs):
        raise NotImplementedError

    
class Slope(DiscontinuityType):
    name   = 'slope'
    pnames = 'slope intercept'.split()
    npar   = len(pnames)

    def model(self, pv):
        return np.poly1d(pv)(self.d.cadence)

    def is_inside_bounds(self, pv):
        return True
    
    def fit(self, use_de=False, de_npop=30, de_niter=100, method='Powell'):
        nlnlike = self.nlnlike_gp if self.d.use_gp else self.nlnlike_wn
        pv0 = np.polyfit(self.d.cadence, self.d.flux, 2)
        self._optimization_result = r = minimize(nlnlike, pv0, method=method)
        self.best_fit_pv = r.x
        self.best_fit_model = self.model(r.x)
        self.bic = self.c_bic(r.fun)
        return self.bic

    
class Jump(DiscontinuityType):
    name   = 'jump'
    pnames = 'center width amplitude bl_constant bl_slope'.split()
    npar   = len(pnames)

    def model(self, pv):
        return fm.m_jump(*pv, cadence=self.d.cadence)

    def is_inside_bounds(self, pv):
        return all(pv[:2] > 0) and (0.5 < pv[1] < 3.0) and (self.d.position-3 <= pv[0] <= self.d.position+3)
    
    def _de_bounds(self, jamp, jpos, fstd):
        return [[     jpos-2,     jpos+2],  # 0 - center
                [          1,          3],  # 1 - width
                [  0.75*jamp,  1.25*jamp],  # 2 - amplitude
                [ -0.20*fstd,  0.20*fstd],  # 3 - baseline constant
                [      -1e-3,       1e-3]]  # 4 - baseline slope 

    def _pv0(self, jamp, jpos, fstd):
        return [jpos, 2, jamp, fstd, 0]
            

class Transit(DiscontinuityType):
    name   = 'transit'
    pnames = 'depth center duration bl_constant bl_slope'.split()
    npar   = len(pnames)

    def model(self, pv):
        return fm.m_transit(*pv, cadence=self.d.cadence)
    
    def is_inside_bounds(self, pv):
        return all(pv[:-1] >= 0.) and (self.d.cadence[1]+0.55*pv[2] < pv[1] < self.d.cadence[-2]-0.55*pv[2]) and (1. < pv[2] < 50.)

    def _de_bounds(self, jamp, jpos, fstd):
        return [[  0.8*jamp, 1.2*jamp],  # 0 - transit depth
                [    jpos-5,   jpos+5],  # 1 - center 
                [       1.2,      50.],  # 2 - duration
                [ -0.2*fstd, 0.2*fstd],  # 3 - baseline constant
                [     -1e-3,     1e-3]]  # 4 - baseline slope
                
    def _pv0(self, jamp, jpos, fstd):
        return [jamp, jpos, 10, fstd, 0]
            
    
class Flare(DiscontinuityType):
    name   = 'flare'
    pnames = 'start duration amplitude bl_constant bl_slope'
    npar   = len(pnames)

    def model(self, pv):
        return fm.m_flare(*pv, cadence=self.d.cadence)

    def is_inside_bounds(self, pv):
        return all(pv[:3] >= 0.) and (self.d.cadence[0] < pv[0] < self.d.cadence[-1]) and (pv[1] < 10.)

    def _de_bounds(self, jamp, jpos, fstd):
        return [[   jpos-5,   jpos+5],  # 0 - flare start
                [      1.2,       7.],  # 1 - flare duration
                [ 0.8*jamp, 1.2*jamp],  # 2 - amplitude
                [-0.2*fstd, 0.2*fstd],  # 3 - baseline constant
                [    -1e-3,     1e-3]]  # 4 - baseline slope
               
    def _pv0(self, jamp, jpos, fstd):
        return array([jpos, 2.5, jamp, fstd, 0])

    
class Discontinuity(object):

    _available_models = Slope, Jump, Transit, Flare
    
    def __init__(self, position, amplitude, cadence, flux, window_width=100):
        """
        Parameters
        ----------
        position : float
            Approximate position of the discontinuity.
        amplitude : float
            Approximate amplitude of the discontinuity.
        cadence : 1D array
            Cadence array
        flux : 1D array
            Flux array
        window_width : int (> 25)
            Size of the local window
        """
        assert isinstance(cadence, (tuple, list, ndarray)) 
        assert isinstance(flux,    (tuple, list, ndarray))
        assert isinstance(window_width, int) and window_width > 25
        assert len(flux) == len(cadence)
        
        self.position = float(position)      # Approximate discontinuity position
        self.amplitude = float(amplitude)    # Approximate discontinuity amplitude
        self._cadence_g = cadence            # Global cadence array
        self._flux_g = flux                  # Global flux array

        self.use_gp = False
        self.gp = None
        self.hp = None
        
        # Initialize discontinuity models
        # -------------------------------
        self.type = UnclassifiedDiscontinuity(self)
        self.models = [M(self) for M in self._available_models]
        self.bics = zeros(len(self.models))

        # Create local cadence and flux arrays
        # ------------------------------------
        self._ww = window_width
        self._hw = self._ww//2
        self.npt = self._ww
        
        idx = np.argmin(np.abs(self._cadence_g-self.position))
        self._sl = sl = np.s_[max(0, idx-self._hw) : min(idx+self._hw, self._cadence_g.size)]
        self.cadence  = self._cadence_g[sl].copy()
        self.flux     = self._flux_g[sl].copy()
        self.local_median = np.median(self.flux)
        self.flux[:] = self.flux / self.local_median - 1.
        self.wn_estimate = (self.flux - mf(self.flux, 5)).std()
        
            
    def setup_gp(self, gp, hp):
        self.use_gp = True
        self.gp = gp
        self.hp = hp
        self.gp.set_parameters(hp)
        self.gp.compute(self.cadence)

        
    def classify(self, use_de=True, de_npop=30, de_niter=100, method='Nelder-Mead', wn_estimate=None, gp=None, hp=None):
        if gp is not None and hp is not None:
            self.setup_gp(gp, hp)

        if wn_estimate is not None:
            self.wn_estimate = wn_estimate
            
        self.bics[:] = array([m.fit(use_de, de_npop, de_niter, method) for m in self.models])
        self.bics -= self.bics.min()
        self.type = self.models[self.bics.argmin()]

        
    @property
    def name(self):
        return self.type.name

    @property
    def global_cadence(self):
        return self._cadence_g

    @property
    def local_cadence(self):
        return self.cadence

    @property
    def global_flux(self):
        return self._flux_g

    @property
    def local_flux(self):
        return self.flux

    
dmodels = Slope, Jump, Transit, Flare
