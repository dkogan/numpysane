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
            (('n',), ('n',)),
            (),
            r'''
            output.data[0] = inner(a.data,
                                   b.data,
                                   a.strides[0],
                                   b.strides[0],
                                   a.shape[0]);
            return true;
            ''',
            "a", "b")

m.write()

