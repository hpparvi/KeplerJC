from numpy.distutils.core import setup, Extension
from numpy.distutils.misc_util import Configuration
import distutils.sysconfig as ds

setup(name='KeplerJC',
      version='0.5',
      description='Jump (discontinuity) detection, classification, and correction for Kepler light curves',
      author='Hannu Parviainen',
      author_email='hpparvi@gmail.com',
      url='https://github.com/hpparvi/KeplerJC',
      package_dir={'keplerjc':'src'},
      packages=['keplerjc']
     )
