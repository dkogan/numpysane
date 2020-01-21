#!/usr/bin/python3

r'''A demo script to generate broadcast-aware python wrapping to samplelib

samplelib is a tiny demo library that can compute inner and outer products. Here
we wrap each available function. For each one we provide a code snipped that
takes raw data arrays for each slice, and invokes the samplelib library for each
one

'''
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path[:0] = dir_path + '/..',

import numpy as np
import numpysane as nps
import numpysane_pywrap as npsp


m = npsp.module( MODULE_NAME      = "samplelibmodule",
                 MODULE_DOCSTRING = "Test module",
                 HEADER           = '#include "samplelib.h"')

m.function( "inner",
            "Inner-product pywrapped with npsp",

            argnames         = ("a", "b"),
            prototype_input  = (('n',), ('n',)),
            prototype_output = (),

            FUNCTION__slice_code = \
                {np.float64:
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
                {np.float64:
                 r'''
            outer((double*)output.data,
                  (double*)a.data,
                  (double*)b.data,
                  a.strides[0],
                  b.strides[0],
                  a.dims[0]);
            return true;
'''})

m.write()
