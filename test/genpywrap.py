#!/usr/bin/python3

r'''generate broadcast-aware python wrapping to testlib

The test suite runs this script to python-wrap the testlib C library, then
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


m = npsp.module( name      = "testlib",
                 docstring = "Some functions to test the python wrapping",
                 header    = '''
#include <stdlib.h>
#include "testlib.h"
''')

m.function( "identity3",
            "Generates a 3x3 identity matrix",

            args_input       = (),
            prototype_input  = (),
            prototype_output = (3,3),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            ((double*)(data_slice__output + i*strides_slice__output[0] + j*strides_slice__output[1]))[0] =
                 (i==j) ? 1.0 : 0.0;
    return true;
'''})

m.function( "identity",
            '''Generates an NxN identity matrix. Output matrices must be passed-in to define N''',

            args_input       = (),
            prototype_input  = (),
            prototype_output = ('N', 'N'),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
    int N = dims_slice__output[0];
    for(int i=0; i<N; i++)
        for(int j=0; j<N; j++)
            ((double*)(data_slice__output + i*strides_slice__output[0] + j*strides_slice__output[1]))[0] =
                 (i==j) ? 1.0 : 0.0;
    return true;
'''},)

m.function( "inner",
            "Inner-product pywrapped with npsp",

            args_input       = ('a', 'b'),
            prototype_input  = (('n',), ('n',)),
            prototype_output = (),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
            const int N = dims_slice__a[0];
            ((double*)data_slice__output)[0] =
              inner_double((double*)data_slice__a,
                           (double*)data_slice__b,
                           strides_slice__a[0],
                           strides_slice__b[0],
                           N);
            return true;
''',
                 np.int64:
                 r'''
            const int N = dims_slice__a[0];
            ((int64_t*)data_slice__output)[0] =
              inner_int64_t((int64_t*)data_slice__a,
                            (int64_t*)data_slice__b,
                            strides_slice__a[0],
                            strides_slice__b[0],
                            N);
            return true;
''',
                 np.int32:
                 r'''
            const int N = dims_slice__a[0];
            ((int32_t*)data_slice__output)[0] =
              inner_int32_t((int32_t*)data_slice__a,
                            (int32_t*)data_slice__b,
                            strides_slice__a[0],
                            strides_slice__b[0],
                            N);
            return true;
'''})

m.function( "outer",
            "Outer-product pywrapped with npsp",

            args_input       = ('a', 'b'),
            prototype_input  = (('n',), ('n',)),
            prototype_output = ('n', 'n'),

            Ccode_slice_eval = \
                {np.float64:
                 r'''
            int N = dims_slice__a[0];
            outer((double*)data_slice__output,
                  strides_slice__output[0],
                  strides_slice__output[1],
                  (double*)data_slice__a,
                  (double*)data_slice__b,
                  strides_slice__a[0],
                  strides_slice__b[0],
                  N);
            return true;
'''})

m.function( "innerouter",
            "Inner and outer products pywrapped with npsp",

            args_input       = ('a', 'b'),
            prototype_input  = (('n',), ('n',)),
            prototype_output = ((), ('n', 'n')),

            Ccode_cookie_struct = '''
            double scale; /* from BOTH scale arguments: "scale", "scale_string" */
            char*  ptr;   /* to demo resource allocation/release */
            ''',

            extra_args = (("double",      "scale",          "1",    "d"),
                          ("const char*", "scale_string",   "NULL", "s")),

            Ccode_validate = r'''
            if(! (*scale > 0.0       &&
                  CHECK_CONTIGUOUS_AND_SETERROR_ALL() ) )
                return false;
            cookie->scale = *scale * (scale_string ? atof(scale_string) : 1.0);
            cookie->ptr   = malloc(1000);
            return cookie->ptr != NULL;
''',

            Ccode_slice_eval = \
                {np.float64:
                 r'''
            int N = dims_slice__a[0];
            *(double*)data_slice__output0 =
                 innerouter((double*)data_slice__output1,
                     (double*)data_slice__a,
                     (double*)data_slice__b,
                     cookie->scale,
                     N);
            return true;
'''},
            Ccode_cookie_cleanup = 'if(cookie->ptr != NULL) free(cookie->ptr);'
           )

m.function( "sorted_indices",
            "Return the sorted element indices",

            args_input       = ('x',),
            prototype_input  = (('n',),),
            prototype_output = ('n',),

            Ccode_slice_eval = \
                {(np.float32, np.int32):
                 r'''
                 sorted_indices_float((int*)data_slice__output,
                     (float*)data_slice__x,
                     dims_slice__x[0]);
                 return true;
''',

                 (np.float64, np.int32):
                 r'''
                 sorted_indices_double((int*)data_slice__output,
                     (double*)data_slice__x,
                     dims_slice__x[0]);
                 return true;
'''},
            Ccode_validate = 'return CHECK_CONTIGUOUS_AND_SETERROR__x();' )

# Tests. Try to wrap functions using illegal output prototypes. The wrapper code
# should barf

try:
    m.function( "outer_broken",
                "Outer-product pywrapped with npsp",
                args_input       = ('a', 'b'),
                prototype_input  = (('n',), ('n',)),
                prototype_output = ('n', -1),
                Ccode_slice_eval = {np.float64: 'return true;'})
except: pass # known error
else:   raise Exception("Expected error didn't happen")

try:
    m.function( "outer_broken",
                "Outer-product pywrapped with npsp",
                args_input       = ('a', 'b'),
                prototype_input  = (('n',), (-1,)),
                prototype_output = ('n', 'n'),
                Ccode_slice_eval = {np.float64: 'return true;'})
except: pass # known error
else:   raise Exception("Expected error didn't happen")

try:
    m.function( "outer_broken",
                "Outer-product pywrapped with npsp",
                args_input       = ('a', 'b'),
                prototype_input  = (('n',), ('n',-1)),
                prototype_output = (),
                Ccode_slice_eval = {np.float64: 'return true;'})
except: pass # known error
else:   raise Exception("Expected error didn't happen: input dims must >=0")

try:
    m.function( "outer_broken",
                "Outer-product pywrapped with npsp",
                args_input       = ('a', 'b'),
                prototype_input  = (('n',), ('n',)),
                prototype_output = (-1,),
                Ccode_slice_eval = {np.float64: 'return true;'})
except: pass # known error
else:   raise Exception("Expected error didn't happen: output dims must >=0")

try:
    m.function( "outer_broken",
                "Outer-product pywrapped with npsp",
                args_input       = ('a', 'b'),
                prototype_input  = (('n',), ('n',)),
                prototype_output = ('m', ()),
                Ccode_slice_eval = {np.float64: 'return true;'})
except: pass # known error
else:   raise Exception("Expected error didn't happen: output dims must be integers or strings")

try:
    m.function( "outer_broken",
                "Outer-product pywrapped with npsp",
                args_input       = ('a', 'b'),
                prototype_input  = (('n',), ('n',)),
                prototype_output = 'n',
                Ccode_slice_eval = {np.float64: 'return true;'})
except: pass # known error
else:   raise Exception("Expected error didn't happen: output dims must be a tuple")

try:
    m.function( "outer_broken",
                "Outer-product pywrapped with npsp",
                args_input       = ('a', 'b'),
                prototype_input  = (('n',), 'n'),
                prototype_output = 'n',
                Ccode_slice_eval = {np.float64: 'return true;'})
except: pass # known error
else:   raise Exception("Expected error didn't happen: output dims must be a tuple")

try:
    m.function( "sorted_indices_broken",
                "Return the sorted element indices",
                args_input       = ('x',),
                prototype_input  = (('n',),),
                prototype_output = ('n',),
                Ccode_slice_eval = { np.float64: 'return true;' })
except: raise Exception("Valid usage of Ccode_slice_eval keys failed")

try:
    m.function( "sorted_indices_broken2",
                "Return the sorted element indices",
                args_input       = ('x',),
                prototype_input  = (('n',),),
                prototype_output = ('n',),
                Ccode_slice_eval = { np.float64: 'return true;', np.int32: 'return true;' })
except: raise Exception("Valid usage of Ccode_slice_eval keys failed")

try:
    m.function( "sorted_indices_broken3",
                "Return the sorted element indices",
                args_input       = ('x',),
                prototype_input  = (('n',),),
                prototype_output = ('n',),
                Ccode_slice_eval = { (np.float64, np.int32): 'return true;', np.int32: 'return true;' })
except: raise Exception("Valid usage of Ccode_slice_eval keys failed")

try:
    m.function( "sorted_indices_broken4",
                "Return the sorted element indices",
                args_input       = ('x',),
                prototype_input  = (('n',),),
                prototype_output = ('n',),
                Ccode_slice_eval = { (np.float64, np.int32, np.int32): 'return true;', np.int32: 'return true;' })
except: pass # known error
else:   raise Exception("Expected invalid usage of Ccode_slice_eval keys didn't fail!")


m.write()
