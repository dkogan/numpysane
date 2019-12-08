#!/usr/bin/python3

r'''Test the broadcasting in C

Uses the "innerouter" guinea pig C library
'''

import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path[:0] = dir_path + '/..',

import numpy as np
import numpysane as nps

# The extension module we're testing
import innerouter



def check(matching_functions, A, B):
    r'''Compare results of pairs of matching functions

    matching_functions is a list of pairs of functions that are supposed to
    produce identical results (innerouter and numpysane implementations of
    inner and outer products). A and B are lists of arguments that we try out.
    These support broadcasting, so either one is allowed to be a single array,
    which is then used for all the checks. I check both dynamically-created and
    inlined "out" arrays

    '''
    N = 1
    if type(A) is tuple and len(A) > N:
        N = len(A)
    if type(B) is tuple and len(B) > N:
        N = len(B)

    if type(A) is not tuple: A = (A,) * N
    if type(B) is not tuple: B = (B,) * N

    for f0,f1 in matching_functions:
        for i in range(N):
            out0 = f0(A[i], B[i])
            out1 = f1(A[i], B[i])
            if np.linalg.norm(out0 - out1) < 1e-10:
                print("Test passed")
            else:
                print("Dynamically-allocated test failed! {}({}, {}) should equal {}({}, {}), but the second one is {}". \
                      format(f0, A[i], B[i], f1, A[i], B[i], out1))

            outshape = out0.shape
            out0 = np.zeros(outshape, dtype=np.array(A[i]).dtype)
            out1 = np.ones (outshape, dtype=np.array(A[i]).dtype)
            f0(A[i], B[i], out=out0)
            f1(A[i], B[i], out=out1)
            if np.linalg.norm(out0 - out1) < 1e-10:
                print("Test passed")
            else:
                print("Inlined 'out' test failed! {}({}, {}) should equal {}({}, {}), but the second one is {}". \
                      format(f0, A[i], B[i], f1, A[i], B[i], out1))


# pairs of functions that should produce identical results
matching_functions = ( (nps.inner, innerouter.inner),
                       (nps.outer, innerouter.outer) )

# Basic 1D arrays
a0 = np.arange(5, dtype=float)
b  = a0+3

# a needs to broadcast; contiguous and strided
a1 = np.arange(10, dtype=float).reshape(2,5)
a2 = nps.transpose(np.arange(10, dtype=float).reshape(5,2))

# Try it!
check(matching_functions, (a0,a1,a2), b)

# Try it again, but use the floating-point version
check( ((nps.inner, innerouter.inner),),
      tuple([a.astype(int) for a in (a0,a1,a2)]),
      b.astype(int))

try:    check( ((nps.inner, innerouter.inner),),
               (a0,a1,a2),
               b.astype(int))
except: pass # expected barf. Types don't match
else:   print("should have barfed but didn't!")

# Too few input dimensions (passing a scalar where a vector is expected). This
# should be ok. It can be viewed as a length-1 vector
check( ((nps.inner, innerouter.inner),),

       6.,

       (5.,
        np.array(5, dtype=float),
        np.array((5,), dtype=float),
        ),)

# Too few output dimensions. Again, this should be ok
out = np.zeros((), dtype=float)
innerouter.inner( nps.atleast_dims(np.array(6.,dtype=float), -5),
                     nps.atleast_dims(np.array(5.,dtype=float), -2),
                     out=out)
if np.linalg.norm(out - 6*5) < 1e-10:
    print("Test passed")
else:
    print("Inlined 'out' test failed! inner(6,5)=30, but I got {}".format(out))


# Broadcasting. Should be ok. No barf.
innerouter.inner(np.arange(10, dtype=float).reshape(  2,5),
                    np.arange(15, dtype=float).reshape(3,1,5))

try:    innerouter.inner(np.arange(10, dtype=float).reshape(2,5),
                            np.arange(15, dtype=float).reshape(3,5))
except: pass # expected barf
else:   print("should have barfed but didn't!")

try:    innerouter.inner(np.arange(5), np.arange(6))
except: pass # expected barf
else:   print("should have barfed but didn't!")

try:    innerouter.outer_only3(np.arange(5), np.arange(5))
except: pass # expected barf
else:   print("should have barfed but didn't!")

innerouter.outer(a0,b, out=np.zeros((5,5), dtype=float))
# wrong dimensions on out. These all should barf
try:    innerouter.outer(a0,b, out=np.zeros((3,3), dtype=float))
except: pass # expected barf
else:   print("should have barfed but didn't!")
try:    innerouter.outer(a0,b, out=np.zeros((4,5), dtype=float))
except: pass # expected barf
else:   print("should have barfed but didn't!")
try:    innerouter.outer(a0,b, out=np.zeros((5,), dtype=float))
except: pass # expected barf
else:   print("should have barfed but didn't!")
try:    innerouter.outer(a0,b, out=np.zeros((), dtype=float))
except: pass # expected barf
else:   print("should have barfed but didn't!")
try:    innerouter.outer(a0,b, out=np.zeros((5,5,5), dtype=float))
except: pass # expected barf
else:   print("should have barfed but didn't!")
