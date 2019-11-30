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




def check(matching_functions, A, B):
    N = 1
    if type(A) is tuple and len(A) > N:
        N = len(A)
    if type(B) is tuple and len(B) > N:
        N = len(B)

    if type(A) is not tuple: A = (A,) * N
    if type(B) is not tuple: B = (B,) * N

    for f0,f1 in matching_functions:
        for i in range(N):
            result0 = f0(A[i], B[i])
            print(np.linalg.norm(result0 -
                                 f1(A[i], B[i])))

            out0 = np.zeros(result0.shape, dtype=np.array(A[i]).dtype)
            out1 = np.ones (result0.shape, dtype=np.array(A[i]).dtype)
            f0(A[i], B[i], out=out0)
            f1(A[i], B[i], out=out1)
            print(np.linalg.norm(out0 - out1))

check(matching_functions, (a0,a1,a2), b)

check( ((nps.inner, testlibmodule.inner),),
      tuple([a.astype(int) for a in (a0,a1,a2)]),
      b.astype(int))

# # should fail
# check( ((nps.inner, testlibmodule.inner),),
#       (a0,a1,a2),
#       b.astype(int))

# too few argument dimensions
check( ((nps.inner, testlibmodule.inner),),

       6.,

       (5.,
        np.array(5, dtype=float),
        np.array((5,), dtype=float),
        ),)

# too few output dimensions
out = np.zeros((), dtype=float)
testlibmodule.inner( nps.atleast_dims(np.array(6.,dtype=float), -5),
                     nps.atleast_dims(np.array(5.,dtype=float), -2),
                     out=out)
print(out - 30)
sys.exit()



# should be ok
testlibmodule.inner(np.arange(10, dtype=float).reshape(  2,5),
                    np.arange(15, dtype=float).reshape(3,1,5))

try:    testlibmodule.inner(np.arange(10, dtype=float).reshape(2,5),
                            np.arange(15, dtype=float).reshape(3,5))
except: pass # expected barf
else:   print("should have barfed but didn't!")

try:    testlibmodule.inner(np.arange(5), np.arange(6))
except: pass # expected barf
else:   print("should have barfed but didn't!")

try:    testlibmodule.outer_only3(np.arange(5), np.arange(5))
except: pass # expected barf
else:   print("should have barfed but didn't!")

# should be ok
testlibmodule.outer_only3(np.arange(3), np.arange(3))

testlibmodule.outer(a0,b, out=np.zeros((5,5), dtype=float))
# wrong dimensions on out. These all should barf
try:    testlibmodule.outer(a0,b, out=np.zeros((3,3), dtype=float))
except: pass # expected barf
else:   print("should have barfed but didn't!")
try:    testlibmodule.outer(a0,b, out=np.zeros((4,5), dtype=float))
except: pass # expected barf
else:   print("should have barfed but didn't!")
try:    testlibmodule.outer(a0,b, out=np.zeros((5,), dtype=float))
except: pass # expected barf
else:   print("should have barfed but didn't!")
try:    testlibmodule.outer(a0,b, out=np.zeros((), dtype=float))
except: pass # expected barf
else:   print("should have barfed but didn't!")
try:    testlibmodule.outer(a0,b, out=np.zeros((5,5,5), dtype=float))
except: pass # expected barf
else:   print("should have barfed but didn't!")
