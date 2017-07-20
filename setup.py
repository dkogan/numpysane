#!/usr/bin/python

from setuptools import setup

setup(name         = 'numpysane',
      version      = '0.9',
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
