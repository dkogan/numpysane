#!/usr/bin/python

# using distutils not setuptools because setuptools puts dist_files in the root
# of the host prefix, not the target prefix. Life is too short to fight this
# nonsense
from distutils.core import setup
import re
import glob

version = None
with open("numpysane.py", "r") as f:
    for l in f:
        m = re.match("__version__ *= *'(.*?)' *$", l)
        if m:
            version = m.group(1)
            break

if version is None:
    raise Exception("Couldn't find version in 'numpysane.py'")

pywrap_templates = glob.glob('pywrap-templates/*.c')

setup(name         = 'numpysane',
      version      = version,
      author       = 'Dima Kogan',
      author_email = 'dima@secretsauce.net',
      url          = 'http://github.com/dkogan/numpysane',
      description  = 'more-reasonable core functionality for numpy',

      long_description = """numpysane is a collection of core routines to provide basic numpy
functionality in a more reasonable way""",

      license      = 'LGPL',
      py_modules   = ['numpysane', 'numpysane_pywrap'],
      data_files = [ ('share/python-numpysane/pywrap-templates', pywrap_templates)])
