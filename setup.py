#!/usr/bin/python

from setuptools import setup
import re

version = None
with open("numpysane.py", "r") as f:
    for l in f:
        m = re.match("__version__ *= *'(.*?)' *$", l)
        if m:
            version = m.group(1)
            break

if version is None:
    raise Exception("Couldn't find version in 'numpysane.py'")


setup(name         = 'numpysane',
      version      = version,
      author       = 'Dima Kogan',
      author_email = 'dima@secretsauce.net',
      url          = 'http://github.com/dkogan/numpysane',
      description  = 'more-reasonable core functionality for numpy',

      long_description = """numpysane is a collection of core routines to provide basic numpy
functionality in a more reasonable way""",

      license      = 'LGPL-3+',
      py_modules   = ['numpysane'],
      test_suite   = 'test_numpysane',
      install_requires = 'numpy')
