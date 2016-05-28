from numpy.distutils.core import setup, Extension
from numpy.distutils.misc_util import Configuration
import distutils.sysconfig as ds

setup(name='KeplerJC',
      version='0.1',
      description='',
      author='Hannu Parviainen',
      author_email='hpparvi@gmail.com',
      url='',
      package_dir={'keplerjc':'src'},
      packages=['keplerjc']
     )
