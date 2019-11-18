#!/usr/bin/python3

import sys
import numpy as np
import numpysane as nps
import numpysane_pywrap as npsp


m = npsp.module( MODULE_NAME      = "innermodule",
                 MODULE_DOCSTRING = "Test module",
                 HEADER           = '#include "inner.h"')

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
                                   a.shape[0]);
            return true;
            ''')

m.write()

