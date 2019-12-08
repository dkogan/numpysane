#!/usr/bin/python3

r'''A demo script to show the python wrapping produced with genpywrap.py
'''

import numpy as np
import numpysane as nps
import sys

# The extension module we're testing
import testlibmodule

# Basic 1D arrays
a0 = np.arange(5, dtype=float)
b  = a0+3

# a needs to broadcast; contiguous and strided
a1 = np.arange(10, dtype=float).reshape(2,5)
a2 = nps.transpose(np.arange(10, dtype=float).reshape(5,2))

print(testlibmodule.inner(a0,b))
print(testlibmodule.inner(a1,b))
print(testlibmodule.inner(a2,b))

print(testlibmodule.outer(a0,b))
print(testlibmodule.outer(a1,b))
print(testlibmodule.outer(a2,b))

out = np.zeros((), dtype=float)
testlibmodule.inner(a0,b, out=out)
print(out)
