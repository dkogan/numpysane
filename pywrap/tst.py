#!/usr/bin/python3

import numpy as np
import numpysane as nps
import sys

import testlibmodule

# pairs of functions that should produce identical results
matching_functions = ( (nps.inner, testlibmodule.inner),
                       (nps.outer, testlibmodule.outer) )

a0 = np.arange(5, dtype=float)
b  = a0+3

a1 = np.arange(10, dtype=float).reshape(2,5)
a2 = nps.transpose(np.arange(10, dtype=float).reshape(5,2))

for f0,f1 in matching_functions:
    print(np.linalg.norm(f0(a0,b) - f1(a0,b)))
    print(np.linalg.norm(f0(a1,b) - f1(a1,b)))
    print(np.linalg.norm(f0(a2,b) - f1(a2,b)))

# should be ok
testlibmodule.inner(np.arange(10, dtype=float).reshape(  2,5),
                    np.arange(15, dtype=float).reshape(3,1,5))

try:
    testlibmodule.inner(np.arange(10, dtype=float).reshape(2,5),
                      np.arange(15, dtype=float).reshape(3,5))
except:
    # expected barf
    pass
else:
    print("should have barfed but didn't!")


try:
    testlibmodule.inner(np.arange(5), np.arange(6))
except:
    # expected barf
    pass
else:
    print("should have barfed but didn't!")

try:
    testlibmodule.outer_only3(np.arange(5), np.arange(5))
except:
    # expected barf
    pass
else:
    print("should have barfed but didn't!")

# should be ok
testlibmodule.outer_only3(np.arange(3), np.arange(3))
