#!/usr/bin/python

from setuptools import setup
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
      install_requires = ('numpy',),

      # This is REALLY stupid. The simple act of shipping some non-source data
      # with the module is really difficult for Python people. There are
      # multiple methods (package_data, data_files, MANIFEST.in, etc) which all
      # kinda work, with the details changing over time and what's right and
      # wrong over time. Lots of exasperated threads on this topic on the
      # internet. It's easy to get this all mostly working but not really. Just
      # now I was using "data_files", which built stuff correctly, and uploaded
      # stuff to pypi correctly, but the "pip install" wouldn't install the
      # data. Or would install it to some un-findable location. There's an
      # unknowable difference between what "setup.py dist" does and "setup.py
      # bdist" does and what "pip install" does.
      #
      # I think I'm supposed to have all the sources and data in a subdirectory
      # from where setup.py is, but I'm not doing that.
      #
      # What I'm doing here isn't "right", but I think it works. Here I'm saying
      # that there's one "package", in the root, and I have template data. In
      # MANIFEST.in I exclude EVERYTHING, and whitelist all the files that I do
      # want to end up in the package. I think it works?
      #
      # Note that I'm not supposed to add '.' as a package, and it yells at me.
      # But it's just a warning so I move on
      packages = ['.'],
      package_data = {'': pywrap_templates}
)
