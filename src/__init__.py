from .jfinder import JumpFinder
from .jclassifier import JumpClassifier
from .core import KData
from .models import JumpSet, Jump

__all__ = ['correct_jumps', 'KData', 'JumpFinder', 'JumpClassifier', 'Jump', 'JumpSet']

def correct_jumps(data, jumps, jc):
    """Correct jumps
    
    Parameters
    ----------
    data  : KData
            Cadence and flux values
            
    jumps : list or JumpSet
            A list of jumps found by JumpFinder and classified
            by JumpClassifier

    jc    : JumpClassifier
    """
    kd = KData(data._cadence, data._flux, data._quality)
    for j in jumps:
        if j.type == 'jump':
            pv = j._pv.copy()
            pv[-2:] = 0
            kd._flux[kd._mask] -= j._median * (jc.m_jump(pv, data.cadence))
    return kd
