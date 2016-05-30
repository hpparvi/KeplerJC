Kepler Jump Detection and Classification
========================================

[![Licence](http://img.shields.io/badge/license-GPLv3-blue.svg?style=flat)](http://www.gnu.org/licenses/gpl-3.0.html)
[![DOI](https://zenodo.org/badge/5871/hpparvi/PyTransit.svg)](https://zenodo.org/badge/latestdoi/5871/hpparvi/PyTransit)

`KeplerJC` (Kepler Jump Correction) is a Python package to detect, classify, and remove isolated
discontinuities from individual Kepler light curves.

Installation
------------

    python setup.py install [--user]


Jump detection and correction
-----------------------------

    from keplerjc import JumpFinder, JumpClassifier
    
    jf = JumpFinder(cadence, flux, exclude=[[8883,8938], [10390,10520]])
    jumps = jf.find_jumps()

    jc = JumpClassifier(cadence, flux, jf.hp)
    jc.classify(jumps)

![Example_1](examples/ex1.png)

The figure above is reproduced in the `Example_1` IPython notebook under the `examples` directory. Discontinuities identified as jumps are marked with slashed vertical lines, while transit-like features are marked as thick vertical lines spanning the upper part of the figure. 

Authors
-------

- Hannu Parviainen
- Suzanne Aigrain

