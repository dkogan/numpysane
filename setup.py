#!/usr/bin/python

from distutils.core          import setup
from distutils.command.clean import clean
import subprocess

subprocess.call( ['make', 'README', 'README.org'] )


class MoreClean(clean):
    def run(self):
        clean.run(self)
        subprocess.call( ['make', 'clean'] )


setup(name         = 'numpysane',
      version      = '0.1',
      author       = 'Dima Kogan',
      author_email = 'dima@secretsauce.net',
      url          = 'http://github.com/dkogan/numpysane',
      description  = 'more-reasonable core functionality for numpy',
      license      = 'LGPL-3+',
      py_modules   = ['numpysane'],
      cmdclass     = {'clean': MoreClean})
