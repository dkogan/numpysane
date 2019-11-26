#!/usr/bin/python3

import numpy as np
import numpysane as nps
import sys

import innermodule

a = np.arange(5, dtype=float)
b = a+3

print(nps.inner(a,b))
print(innermodule.inner(a,b))

a = np.arange(10, dtype=float).reshape(2,5)

print(nps.inner(a,b))
print(innermodule.inner(a,b))

a = nps.transpose(np.arange(10, dtype=float).reshape(5,2))

print(nps.inner(a,b))
print(innermodule.inner(a,b))

# should barf
innermodule.inner(np.arange(5), np.arange(6))
