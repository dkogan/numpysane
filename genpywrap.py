#!/usr/bin/python2

import sys
import numpy as np
import numpysane as nps
import gnuplotlib as gp


with open( 'pywrap_header.c', 'r') as f:
    header = f.read()

validation = r'''
#include "inner.h"

static
bool VALIDATE(void)
{
    return true;
}
static
bool _inner_one_slice( nps_slice_t output,
                       nps_slice_t a,
                       nps_slice_t b)
{
    assert(a.strides[0] == sizeof(double));
    assert(b.strides[0] == sizeof(double));
    output.data[0] = inner(a.data, b.data, a.shape[0]);
    return true;
}
'''


PROTOTYPE_DEFS = r'''
    const int PROTOTYPE_a[1] = {-1};
    const int PROTOTYPE_b[1] = {-1};
    // Not const. updating the named dimensions in-place
    int PROTOTYPE___output__[0];

    const int PROTOTYPE_LEN_a = (int)sizeof(PROTOTYPE_a)/sizeof(PROTOTYPE_a[0]);
    const int PROTOTYPE_LEN_b = (int)sizeof(PROTOTYPE_b)/sizeof(PROTOTYPE_b[0]);
    const int PROTOTYPE_LEN___output__ = (int)sizeof(PROTOTYPE___output__)/sizeof(PROTOTYPE___output__[0]);

    // compute this at generation time
    int Ndims_named = 1;

    int Ndims_extra_a = PyArray_NDIM(_py_a) - PROTOTYPE_LEN_a;
    int Ndims_extra_b = PyArray_NDIM(_py_b) - PROTOTYPE_LEN_b;
    int Ndims_extra   = 0;
    if(Ndims_extra < Ndims_extra_a) Ndims_extra = Ndims_extra_a;
    if(Ndims_extra < Ndims_extra_b) Ndims_extra = Ndims_extra_b;
'''










prototype = (('n',), ('n',)), ()
