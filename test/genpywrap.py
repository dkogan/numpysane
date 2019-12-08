#!/usr/bin/python3

r'''generate broadcast-aware python wrapping to innerouter

The test suite runs this script to python-wrap the innerouter C library, then
the test suite builds this python extension module, and then the test suite
validates this module's behavior

'''
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path[:0] = dir_path + '/..',

import numpy as np
import numpysane as nps
import numpysane_pywrap as npsp


m = npsp.module( MODULE_NAME      = "innerouter",
                 MODULE_DOCSTRING = "Inner and outer products module",
                 HEADER           = '#include "innerouter.h"')

m.function( "inner",
            "Inner-product pywrapped with npsp",

            argnames         = ("a", "b"),
            prototype_input  = (('n',), ('n',)),
            prototype_output = (),

            FUNCTION__slice_code = \
                {float:
                 r'''
            ((double*)output.data)[0] =
              inner_double((double*)a.data,
                    (double*)b.data,
                    a.strides[0],
                    b.strides[0],
                    a.dims[0]);
            return true;
''',
                 np.int64:
                 r'''
            ((int64_t*)output.data)[0] =
              inner_int64_t((int64_t*)a.data,
                    (int64_t*)b.data,
                    a.strides[0],
                    b.strides[0],
                    a.dims[0]);
            return true;
''',
                 np.int32:
                 r'''
            ((int32_t*)output.data)[0] =
              inner_int32_t((int32_t*)a.data,
                    (int32_t*)b.data,
                    a.strides[0],
                    b.strides[0],
                    a.dims[0]);
            return true;
'''})

m.function( "outer",
            "Outer-product pywrapped with npsp",

            argnames         = ("a", "b"),
            prototype_input  = (('n',), ('n',)),
            prototype_output = ('n', 'n'),

            FUNCTION__slice_code = \
                {float:
                 r'''
            outer((double*)output.data,
                  output.strides[0],
                  output.strides[1],
                  (double*)a.data,
                  (double*)b.data,
                  a.strides[0],
                  b.strides[0],
                  a.dims[0]);
            return true;
'''})


# Tests. Try to wrap functions using illegal output prototypes. The wrapper code
# should barf
try:
    m.function( "outer2",
                "Outer-product pywrapped with npsp",

                argnames         = ("a", "b"),
                prototype_input  = (('n',), ('n',)),
                prototype_output = ('n', 'fn'),

                FUNCTION__slice_code = '')
except: pass # known error
else:   raise Exception("Expected error didn't happen")

try:
    m.function( "outer3",
                "Outer-product pywrapped with npsp",

                argnames         = ("a", "b"),
                prototype_input  = (('n',), ('n',)),
                prototype_output = ('n', -1),

                FUNCTION__slice_code = '')
except: pass # known error
else:   raise Exception("Expected error didn't happen")

try:
    m.function( "outer4",
                "Outer-product pywrapped with npsp",

                argnames         = ("a", "b"),
                prototype_input  = (('n',), (-1,)),
                prototype_output = ('n', 'n'),

                FUNCTION__slice_code = '')
except: pass # known error
else:   raise Exception("Expected error didn't happen")


m.write()
