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
            ((double*)data__output)[0] =
              inner_double((double*)data__a,
                    (double*)data__b,
                    strides__a[0],
                    strides__b[0],
                    dims__a[0]);
            return true;
''',
                 np.int64:
                 r'''
            ((int64_t*)data__output)[0] =
              inner_int64_t((int64_t*)data__a,
                    (int64_t*)data__b,
                    strides__a[0],
                    strides__b[0],
                    dims__a[0]);
            return true;
''',
                 np.int32:
                 r'''
            ((int32_t*)data__output)[0] =
              inner_int32_t((int32_t*)data__a,
                    (int32_t*)data__b,
                    strides__a[0],
                    strides__b[0],
                    dims__a[0]);
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
            outer((double*)data__output,
                  strides__output[0],
                  strides__output[1],
                  (double*)data__a,
                  (double*)data__b,
                  strides__a[0],
                  strides__b[0],
                  dims__a[0]);
            return true;
'''})

m.function( "innerouter",
            "Inner and outer products pywrapped with npsp",

            argnames         = ("a", "b"),
            prototype_input  = (('n',), ('n',)),
            prototype_output = ((), ('n', 'n')),

            FUNCTION__slice_code = \
                {float:
                 r'''
            *(double*)data__output0 =
                 innerouter((double*)data__output1,
                     (double*)data__a,
                     (double*)data__b,
                     dims__a[0]);
            return true;
'''},
            VALIDATE_code = r'''
            /* output0 is a scalar. It can't be discontiguous */
            /* output1 is NxN. MUST be contiguous */
            int Nelems_slice = 1;
            for(int i=-1; i>=-2; i--)
            {
                if(strides__output1[i+Ndims__output1] != sizeof_element__output1*Nelems_slice)
                    return false;
                Nelems_slice *= dims__output1[i+Ndims__output1];
            }

            /* The two inputs must be contiguous too */
            if(strides__a[Ndims__a-1] != sizeof_element__a)
                return false;
            if(strides__b[Ndims__b-1] != sizeof_element__b)
                return false;

            return true;
''' )


# Tests. Try to wrap functions using illegal output prototypes. The wrapper code
# should barf
try:
    m.function( "outer_broken",
                "Outer-product pywrapped with npsp",
                argnames         = ("a", "b"),
                prototype_input  = (('n',), ('n',)),
                prototype_output = ('n', 'fn'),
                FUNCTION__slice_code = dict(float=''))
except: pass # known error
else:   raise Exception("Expected error didn't happen")

try:
    m.function( "outer_broken",
                "Outer-product pywrapped with npsp",
                argnames         = ("a", "b"),
                prototype_input  = (('n',), ('n',)),
                prototype_output = ('n', -1),
                FUNCTION__slice_code = dict(float=''))
except: pass # known error
else:   raise Exception("Expected error didn't happen")

try:
    m.function( "outer_broken",
                "Outer-product pywrapped with npsp",
                argnames         = ("a", "b"),
                prototype_input  = (('n',), (-1,)),
                prototype_output = ('n', 'n'),
                FUNCTION__slice_code = dict(float=''))
except: pass # known error
else:   raise Exception("Expected error didn't happen")

# first check invalid broadcasting defines
try:
    m.function( "outer_broken",
                "Outer-product pywrapped with npsp",
                argnames         = ("a", "b"),
                prototype_input  = (('n',), ('n',())),
                prototype_output = ('m'),
                FUNCTION__slice_code = dict(float=''))
except: pass # known error
else:   raise Exception("Expected error didn't happen: input dims must be integers or strings")

try:
    m.function( "outer_broken",
                "Outer-product pywrapped with npsp",
                argnames         = ("a", "b"),
                prototype_input  = (('n',), ('n',-1)),
                prototype_output = (),
                FUNCTION__slice_code = dict(float=''))
except: pass # known error
else:   raise Exception("Expected error didn't happen: input dims must >=0")

try:
    m.function( "outer_broken",
                "Outer-product pywrapped with npsp",
                argnames         = ("a", "b"),
                prototype_input  = (('n',), ('n',)),
                prototype_output = (-1,),
                FUNCTION__slice_code = dict(float=''))
except: pass # known error
else:   raise Exception("Expected error didn't happen: output dims must >=0")

try:
    m.function( "outer_broken",
                "Outer-product pywrapped with npsp",
                argnames         = ("a", "b"),
                prototype_input  = (('n',), ('n',)),
                prototype_output = ('m', ()),
                FUNCTION__slice_code = dict(float=''))
except: pass # known error
else:   raise Exception("Expected error didn't happen: output dims must be integers or strings")

try:
    m.function( "outer_broken",
                "Outer-product pywrapped with npsp",
                argnames         = ("a", "b"),
                prototype_input  = (('n',), ('n',)),
                prototype_output = ('m'),
                FUNCTION__slice_code = dict(float=''))
except: pass # known error
else:   raise Exception("Expected error didn't happen: output dims must all be known")

try:
    m.function( "outer_broken",
                "Outer-product pywrapped with npsp",
                argnames         = ("a", "b"),
                prototype_input  = (('n',), ('n',)),
                prototype_output = 'n',
                FUNCTION__slice_code = dict(float=''))
except: pass # known error
else:   raise Exception("Expected error didn't happen: output dims must be a tuple")

try:
    m.function( "outer_broken",
                "Outer-product pywrapped with npsp",
                argnames         = ("a", "b"),
                prototype_input  = (('n',), 'n'),
                prototype_output = 'n',
                FUNCTION__slice_code = dict(float=''))
except: pass # known error
else:   raise Exception("Expected error didn't happen: output dims must be a tuple")


m.write()
