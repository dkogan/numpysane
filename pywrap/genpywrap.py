#!/usr/bin/python3

import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path[:0] = dir_path + '/..',

import numpy as np
import numpysane as nps
import numpysane_pywrap as npsp


m = npsp.module( MODULE_NAME      = "testlibmodule",
                 MODULE_DOCSTRING = "Test module",
                 HEADER           = '#include "testlib.h"')

m.function( "inner",
            "Inner-product pywrapped with npsp",

            argnames         = ("a", "b"),
            prototype_input  = (('n',), ('n',)),
            prototype_output = (),

            FUNCTION__slice_code = r'''
            output.data[0] = inner(a.data,
                                   b.data,
                                   a.strides[0],
                                   b.strides[0],
                                   a.dims[0]);
            return true;
''')

m.function( "outer",
            "Outer-product pywrapped with npsp",

            argnames         = ("a", "b"),
            prototype_input  = (('n',), ('n',)),
            prototype_output = ('n', 'n'),

            FUNCTION__slice_code = r'''
            outer(output.data,
                  a.data,
                  b.data,
                  a.strides[0],
                  b.strides[0],
                  a.dims[0]);
            return true;
''')

m.write()

