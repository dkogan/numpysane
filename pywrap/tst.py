#!/usr/bin/python3

import numpy as np
import numpysane as nps
import sys

import testlibmodule

a = np.arange(5, dtype=float)
b = a+3

print(nps.inner(a,b))
print(testlibmodule.inner(a,b))

a = np.arange(10, dtype=float).reshape(2,5)

print(nps.inner(a,b))
print(testlibmodule.inner(a,b))

a = nps.transpose(np.arange(10, dtype=float).reshape(5,2))

print(nps.inner(a,b))
print(testlibmodule.inner(a,b))

# should be ok
print(testlibmodule.inner(np.arange(10, dtype=float).reshape(  2,5),
                        np.arange(15, dtype=float).reshape(3,1,5)))

try:
    testlibmodule.inner(np.arange(10, dtype=float).reshape(2,5),
                      np.arange(15, dtype=float).reshape(3,5))
    print("should have barfed but didn't!")
except:
    # expected barf
    pass


try:
    testlibmodule.inner(np.arange(5), np.arange(6))
    print("should have barfed but didn't!")
except:
    # expected barf
    pass
