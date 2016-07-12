from .jfinder import JumpFinder
from .jclassifier import JumpClassifier
from .core import Jump, JumpSet, KData

__all__ = ['correct_jumps', 'KData', 'JumpFinder', 'JumpClassifier', 'Jump', 'JumpSet']

def correct_jumps(data, jumps):
    """Correct jumps
    
    Parameters
    ----------
    data  : KData
            Cadence and flux values
            
    jumps : list or JumpSet
            A list of jumps found by JumpFinder and classified
            by JumpClassifier
    """
    kd = KData(data._cadence, data._flux)
    nf = data.normalized_flux.copy()
    for j in jumps:
        if j.type == 'jump':
            nf[data.cadence >= j.pos] -= j.amp
    kd._flux[kd._mask] = (nf + 1.) * data.median
    return kd
