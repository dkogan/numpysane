#!/usr/bin/python2

import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path[:0] = dir_path + '/..',

import numpy as np
import numpysane as nps
from functools import reduce

# Local test harness. The python standard ones all suck
from testutils import *


def arr(*shape, **kwargs):

    dtype = kwargs.get('dtype',float)

    r'''Return an arange() array of the given shape.'''
    if len(shape) == 0:
        return np.array(3, dtype=dtype)
    product = reduce( lambda x,y: x*y, shape)
    return np.arange(product, dtype=dtype).reshape(*shape)


def test_broadcasting():

    # first check invalid broadcasting defines
    def define_f_broken1():
        @nps.broadcast_define( (('n',), ('n',())), ('m') )
        def f_broken(a, b):
            return a.dot(b)
    confirm_raises( define_f_broken1, msg="input dims must be integers or strings" )

    def define_f_broken2():
        @nps.broadcast_define( (('n',), ('n',-1)), () )
        def f_broken(a, b):
            return a.dot(b)
    confirm_raises( define_f_broken2, msg="input dims must >=0" )

    def define_f_broken3():
        @nps.broadcast_define( (('n',), ('n',)), (-1,) )
        def f_broken(a, b):
            return a.dot(b)
    confirm_raises( define_f_broken3, msg="output dims must >=0" )

    def define_f_broken4():
        @nps.broadcast_define( (('n',), ('n',)), ('m', ()) )
        def f_broken(a, b):
            return a.dot(b)
    confirm_raises( define_f_broken4, msg="output dims must be integers or strings" )

    def define_f_broken5():
        @nps.broadcast_define( (('n',), ('n',)), ('m') )
        def f_broken(a, b):
            return a.dot(b)
    confirm_raises( define_f_broken5, msg="output dims must all be known" )

    def define_f_broken6():
        @nps.broadcast_define( (('n',), ('n',)), 'n' )
        def f_broken(a, b):
            return a.dot(b)
    confirm_raises( define_f_broken6, msg="output dims must be a tuple" )

    def define_f_broken7():
        @nps.broadcast_define( (('n',), ('n',)), ('n',), ('n',) )
        def f_broken(a, b):
            return a.dot(b)
    confirm_raises( define_f_broken7, msg="multiple outputs must be specified as a tuple of tuples" )

    def define_f_broken8():
        @nps.broadcast_define( (('n',), ('n',)), (('n',), 'n') )
        def f_broken(a, b):
            return a.dot(b)
    confirm_raises( define_f_broken8, msg="output dims must be a tuple" )

    def define_f_good9():
        @nps.broadcast_define( (('n',), ('n',)), (('n',), ('n',)) )
        def f_broken(a, b):
            return a.dot(b)
        return True
    confirm( define_f_good9, msg="Multiple outputs can be defined" )


    r'''Checking broadcasting rules.'''
    @nps.broadcast_define( (('n',), ('n',)) )
    def f1(a, b):
        r'''Basic inner product.'''
        return a.dot(b)

    assertValueShape( np.array(5),                (),     f1, arr(3),     arr(3))
    assertValueShape( np.array((5,14)),           (2,),   f1, arr(2,3),   arr(3))
    assertValueShape( np.array((5,14)),           (2,),   f1, arr(3),     arr(2,3))
    assertValueShape( np.array(((5,14),)),        (1,2,), f1, arr(1,2,3), arr(3))
    assertValueShape( np.array(((5,),(14,))),     (2,1,), f1, arr(2,1,3), arr(3))
    assertValueShape( np.array((5,14)),           (2,),   f1, arr(2,3),   arr(1,3))
    assertValueShape( np.array((5,14)),           (2,),   f1, arr(1,3),   arr(2,3))
    assertValueShape( np.array(((5,14),)),        (1,2,), f1, arr(1,2,3), arr(1,3))
    assertValueShape( np.array(((5,),(14,))),     (2,1,), f1, arr(2,1,3), arr(1,3))
    assertValueShape( np.array(((5,14),(14,50))), (2,2,), f1, arr(2,1,3), arr(2,3))
    assertValueShape( np.array(((5,14),(14,50))), (2,2,), f1, arr(2,1,3), arr(1,2,3))

    confirm_raises( lambda: f1(arr(3)), msg='right number of args' )

    confirm_raises( lambda: f1(arr(3),arr(5)),         msg='matching args')
    confirm_raises( lambda: f1(arr(2,3),arr(4,3)),     msg='matching args')
    confirm_raises( lambda: f1(arr(3,3,3),arr(2,1,3)), msg='matching args')
    confirm_raises( lambda: f1(arr(1,2,4),arr(2,1,3)), msg='matching args')


    # fancier function, has some preset dimensions
    @nps.broadcast_define( ((3,), ('n',3), ('n',), ('m',)) )
    def f2(a,b,c,d):
        return d

    n=4
    m=6
    d = np.arange(m)

    assertValueShape( d, (m,),
                           f2,
                           arr(  3),
                           arr(n,3),
                           arr(  n),
                           arr(  m))
    assertValueShape( np.array((d,)), (1,m),
                           f2,
                           arr(1,    3),
                           arr(1,  n,3),
                           arr(      n),
                           arr(1,    m))
    assertValueShape( np.array((d,)), (1,m,),
                           f2,
                           arr(1,    3),
                           arr(1,  n,3),
                           arr(      n),
                           arr(      m))
    assertValueShape( np.array((d,d+m,d+2*m,d+3*m,d+4*m)), (5,m),
                           f2,
                           arr(5,    3),
                           arr(5,  n,3),
                           arr(      n),
                           arr(5,    m))
    assertValueShape( np.array(((d,d+m,d+2*m,d+3*m,d+4*m),)), (1,5,m),
                           f2,
                           arr(1,5,    3),
                           arr(  5,  n,3),
                           arr(        n),
                           arr(  5,    m))
    assertValueShape( np.array(((d,d+m,d+2*m,d+3*m,d+4*m), (d,d+m,d+2*m,d+3*m,d+4*m))), (2,5,m),
                           f2,
                           arr(1,5,    3),
                           arr(2,5,  n,3),
                           arr(        n),
                           arr(  5,    m))
    assertValueShape( np.array(((d,d+m,d+2*m,d+3*m,d+4*m), (d,d+m,d+2*m,d+3*m,d+4*m))), (2,5,m),
                           f2,
                           arr(1,5,    3),
                           arr(2,1,  n,3),
                           arr(        n),
                           arr(  5,    m))
    assertValueShape( np.array((((d,d,d,d,d), (d,d,d,d,d)),)), (1,2,5,m),
                           f2,
                           arr(1,1,5,    3),
                           arr(1,2,1,  n,3),
                           arr(1,        n),
                           arr(1,  1,    m))

    confirm_raises( lambda: f2(
                      arr(5,    3),
                      arr(5,  n,3),
                      arr(      m),
                      arr(5,    m)),
                 msg='matching args')
    confirm_raises( lambda: f2(
                      arr(5,    2),
                      arr(5,  n,3),
                      arr(      n),
                      arr(5,    m)),
                 msg='matching args')
    confirm_raises( lambda: f2(
                      arr(5,    2),
                      arr(5,  n,2),
                      arr(      n),
                      arr(5,    m)),
                 msg='matching args')
    confirm_raises( lambda: f2(
                      arr(1,    3),
                      arr(1,  n,3),
                      arr(      5*n),
                      arr(1,    m)),
                 msg='matching args')

    # Make sure extra args and the kwargs are passed through
    @nps.broadcast_define( ((3,), ('n',3), ('n',), ('m',)) )
    def f3(a,b,c,d, e,f, *args, **kwargs):
        def val_or_0(x): return x if x else 0
        return np.array( (a[0], val_or_0(e), val_or_0(f), val_or_0(args[0]), val_or_0( kwargs.get('xxx'))) )
    assertValueShape( np.array( ((0, 1, 2, 3, 6), (3, 1, 2, 3, 6)) ), (2,5),
                           f3,
                           arr(2,    3),
                           arr(1,  n,3),
                           arr(      n),
                           arr(      m),
                           1, 2, 3, 4., dummy=5, xxx=6)

    # Make sure scalars (0-dimensional array) can broadcast
    @nps.broadcast_define( (('n',), ('n','m'), (2,), ()) )
    def f4(a,b,c,d):
        return d
    @nps.broadcast_define( (('n',), ('n','m'), (2,), ()) )
    def f5(a,b,c,d):
        return nps.glue( c, d, axis=-1 )

    assertValueShape( np.array((5,5)), (2,),
                           f4,
                           arr(      3),
                           arr(1,  3,4),
                           arr(2,    2),
                           np.array(5))
    assertValueShape( np.array((5,5)), (2,),
                           f4,
                           arr(      3),
                           arr(1,  3,4),
                           arr(2,    2),
                           5)
    assertValueShape( np.array(((0,1,5),(2,3,5))), (2,3),
                           f5,
                           arr(      3),
                           arr(1,  3,4),
                           arr(2,    2),
                           np.array(5))
    assertValueShape( np.array(((0,1,5),(2,3,5))), (2,3),
                           f5,
                           arr(      3),
                           arr(1,  3,4),
                           arr(2,    2),
                           5)
    confirm_raises( lambda: f5(
                           arr(      3),
                           arr(1,  3,4),
                           arr(2,    2),
                           arr(5)) )

    # Test the generator
    i=0
    for s in nps.broadcast_generate( (('n',), ('n','m'), (2,), ()),
                                     (arr(      3),
                                      arr(1,  3,4),
                                      arr(2,    2),
                                      np.array(5)) ):
        confirm_equal( arr(3),       s[0] )
        confirm_equal( arr(3,4),     s[1] )
        confirm_equal( arr(2) + 2*i, s[2] )
        confirm_equal( np.array(5),  s[3] )
        confirm_equal         ( s[3].shape, ())
        i = i+1

    i=0
    for s in nps.broadcast_generate( (('n',), ('n','m'), (2,), ()),
                                     (arr(      3),
                                      arr(1,  3,4),
                                      arr(2,    2),
                                      5) ):
        confirm_equal( arr(3),       s[0] )
        confirm_equal( arr(3,4),     s[1] )
        confirm_equal( arr(2) + 2*i, s[2] )
        confirm_equal( np.array(5),  s[3] )
        confirm_equal( s[3].shape, ())
        i = i+1

    i=0
    for s in nps.broadcast_generate( (('n',), ('n','m'), (2,), ()),
                                     (arr(      3),
                                      arr(1,  3,4),
                                      arr(2,    2),
                                      arr(2)) ):
        confirm_equal( arr(3),       s[0] )
        confirm_equal( arr(3,4),     s[1] )
        confirm_equal( arr(2) + 2*i, s[2] )
        confirm_equal( np.array(i),  s[3] )
        confirm_equal( s[3].shape, ())
        i = i+1

    # Make sure we add dummy length-1 dimensions
    assertValueShape( None, (3,),
                           nps.matmult,  arr(4), arr(4,3) )
    assertValueShape( None, (3,),
                           nps.matmult2, arr(4), arr(4,3) )
    assertValueShape( None, (1,3,),
                           nps.matmult, arr(1,4), arr(4,3) )
    assertValueShape( None, (1,3,),
                           nps.matmult2, arr(1,4), arr(4,3) )
    assertValueShape( None, (10,3,),
                           nps.matmult, arr(4), arr(10,4,3) )
    assertValueShape( None, (10,3,),
                           nps.matmult2, arr(4), arr(10,4,3) )
    assertValueShape( None, (10,1,3,),
                           nps.matmult, arr(1,4), arr(10,4,3) )
    assertValueShape( None, (10,1,3,),
                           nps.matmult2, arr(1,4), arr(10,4,3) )

    # scalar output shouldn't barf
    @nps.broadcast_define( ((),), )
    def f6(x):
        return 6
    @nps.broadcast_define( ((),), ())
    def f7(x):
        return 7
    assertValueShape( 6, (),
                      f6, 5)
    assertValueShape( 6*np.ones((5,)), (5,),
                      f6, np.arange(5))
    assertValueShape( 7, (),
                      f7, 5)
    assertValueShape( 7*np.ones((5,)), (5,),
                      f7, np.arange(5))

    # make sure the output dimensionality is checked
    @nps.broadcast_define( (('n',), ('n',)), ('n',) )
    def f8(a, b):
        return a.dot(b)
    confirm_raises( lambda: f8(arr(5), arr(  5)), msg='output dimensionality check' )
    confirm_raises( lambda: f8(arr(5), arr(2,5)), msg='output dimensionality check' )

    # make sure the output COUNTS are checked (if I expect 2 outputs, but get
    # only 1, that's an error
    @nps.broadcast_define( (('n',), ('n',)) )
    def f9(a, b):
        return a.dot(b),nps.outer(a,b)
    confirm_raises( lambda: f9(arr(5), arr(  5)), msg='output count check' )
    confirm_raises( lambda: f9(arr(5), arr(2,5)), msg='output count check' )
    @nps.broadcast_define( (('n',), ('n',)), ('n',) )
    def f10(a, b):
        return a.dot(b),nps.outer(a,b)
    confirm_raises( lambda: f10(arr(5), arr(  5)), msg='output count check' )
    confirm_raises( lambda: f10(arr(5), arr(2,5)), msg='output count check' )
    @nps.broadcast_define( (('n',), ('n',)), ('n', 'n') )
    def f11(a, b):
        return a.dot(b),nps.outer(a,b)
    confirm_raises( lambda: f11(arr(5), arr(  5)), msg='output count check' )
    confirm_raises( lambda: f11(arr(5), arr(2,5)), msg='output count check' )
    @nps.broadcast_define( (('n',), ('n',)), (('n', 'n'),) )
    def f11(a, b):
        return a.dot(b),nps.outer(a,b)
    confirm_raises( lambda: f11(arr(5), arr(  5)), msg='output count check' )
    confirm_raises( lambda: f11(arr(5), arr(2,5)), msg='output count check' )
    @nps.broadcast_define( (('n',), ('n',)), (('n', 'n'),('n',)) )
    def f12(a, b):
        return a.dot(b),nps.outer(a,b)
    confirm_raises( lambda: f12(arr(5), arr(  5)), msg='output count check' )
    confirm_raises( lambda: f12(arr(5), arr(2,5)), msg='output count check' )
    @nps.broadcast_define( (('n',), ('n',)), (('n',),('n', 'n')) )
    def f13(a, b):
        return a.dot(b),nps.outer(a,b)
    confirm_raises( lambda: f13(arr(5), arr(  5)), msg='output dimensionality check' )
    confirm_raises( lambda: f13(arr(5), arr(2,5)), msg='output dimensionality check' )
    @nps.broadcast_define( (('n',), ('n',)), ((),('n', 'n',)) )
    def f13(a, b):
        return a.dot(b),nps.outer(a,b)
    confirm( f13(arr(5), arr(  5)) is not None, msg='output count check' )
    confirm( f13(arr(5), arr(2,5)) is not None, msg='output count check' )

    # check output dimensionality with an 'out' kwarg
    @nps.broadcast_define( (('n',), ('n',)), ((),('n', 'n')),
                           out_kwarg = 'out')
    def f14(a, b, out=None):
        if out is None:
            return a.dot(b),nps.outer(a,b)
        if not isinstance(out,tuple) or len(out) != 2:
            raise Exception("'out' must be a tuple")
        nps.inner(a,b,out=out[0])
        nps.outer(a,b,out=out[1])
        return out

    confirm( f14(arr(5), arr(  5)) is not None,
             msg='output dimensionality check with out_kwarg' )


    # Basic out_kwarg tests. More thorough ones later, in
    # test_broadcasting_into_output())
    a5   = arr(   5,         dtype=float)
    a25  = arr(2, 5,         dtype=float)
    o    = np.zeros((),      dtype=float)
    o2   = np.zeros((2,),    dtype=float)
    o5   = np.zeros((5,),    dtype=float)
    o55  = np.zeros((5,5),   dtype=float)
    o25  = np.zeros((2,5),   dtype=float)
    o255 = np.zeros((2,5,5), dtype=float)

    # no broadcasting
    confirm_raises( lambda: f14(a5, a5, out=o),
                                msg='output dimensionality check with out_kwarg' )
    confirm_raises( lambda: f14(a5, a5, out=o2),
                                msg='output dimensionality check with out_kwarg' )
    confirm_raises( lambda: f14(a5, a5, out=(o,)),
                                msg='output dimensionality check with out_kwarg' )
    confirm_raises( lambda: f14(a5, a5, out=(o55,)),
                                msg='output dimensionality check with out_kwarg' )
    confirm_raises( lambda: f14(a5, a5, out=(o55,o)),
                                msg='output dimensionality check with out_kwarg' )
    confirm_raises( lambda: f14(a5, a5, out=(o,o2)),
                                msg='output dimensionality check with out_kwarg' )
    confirm_raises( lambda: f14(a5, a5, out=(o,o5)),
                                msg='output dimensionality check with out_kwarg' )
    confirm_raises( lambda: f14(a5, a5, out=(o2,o55)),
                                msg='output dimensionality check with out_kwarg' )
    confirm_raises( lambda: f14(a5, a5, out=(o,o55,o)),
                                msg='output dimensionality check with out_kwarg' )
    confirm( f14(a5, a5, out=(o,o55)) is not None,
             msg='output dimensionality check with out_kwarg' )
    confirm_equal(o,   a5.dot(a5),      msg='in-place broadcasting computed the right value')
    confirm_equal(o55, np.outer(a5,a5), msg='in-place broadcasting computed the right value')

    # two broadcasted slices
    confirm_raises( lambda: f14(a5, a25, out=o),
                                msg='output dimensionality check with out_kwarg' )
    confirm_raises( lambda: f14(a5, a25, out=o2),
                                msg='output dimensionality check with out_kwarg' )
    confirm_raises( lambda: f14(a5, a25, out=(o,)),
                                msg='output dimensionality check with out_kwarg' )
    confirm_raises( lambda: f14(a5, a25, out=(o55,)),
                                msg='output dimensionality check with out_kwarg' )
    confirm_raises( lambda: f14(a5, a25, out=(o55,o)),
                                msg='output dimensionality check with out_kwarg' )
    confirm_raises( lambda: f14(a5, a25, out=(o,o2)),
                                msg='output dimensionality check with out_kwarg' )
    confirm_raises( lambda: f14(a5, a25, out=(o,o5)),
                                msg='output dimensionality check with out_kwarg' )
    confirm_raises( lambda: f14(a5, a25, out=(o2,o55)),
                                msg='output dimensionality check with out_kwarg' )
    confirm_raises( lambda: f14(a5, a25, out=(o,o55,o)),
                                msg='output dimensionality check with out_kwarg' )
    confirm_raises( lambda: f14(a5, a25, out=(o,o55)),
                                msg='output dimensionality check with out_kwarg' )
    confirm_raises( lambda: f14(a5, a25, out=(o2,o55)),
                                msg='output dimensionality check with out_kwarg' )
    confirm_raises( lambda: f14(a5, a25, out=(o,o255)),
                                msg='output dimensionality check with out_kwarg' )
    confirm( f14(a5, a25, out=(o2,o255)) is not None,
             msg='output dimensionality check with out_kwarg' )
    confirm_equal(o2,   nps.inner(a5,a25), msg='in-place broadcasting computed the right value')
    confirm_equal(o255, nps.outer(a5,a25), msg='in-place broadcasting computed the right value')


def test_broadcasting_into_output():
    r'''Checking broadcasting with the output array defined.'''

    # I think about all 2^5 = 32 combinations:
    #
    # broadcast_define(): yes/no prototype_output, out_kwarg, single/multiple outputs
    # broadcasted call:   yes/no dtype, output

    prototype_input = (('n',), ('n',))
    in1, in2 = arr(3), arr(2,4,3)
    out_inner_ref = np.array([[ 5, 14, 23, 32],
                              [41, 50, 59, 68]])
    out_outer_ref = np.array([[[[ 0.,  0.,  0.],
                                [ 0.,  1.,  2.],
                                [ 0.,  2.,  4.]],
                               [[ 0.,  0.,  0.],
                                [ 3.,  4.,  5.],
                                [ 6.,  8., 10.]],
                               [[ 0.,  0.,  0.],
                                [ 6.,  7.,  8.],
                                [12., 14., 16.]],
                               [[ 0.,  0.,  0.],
                                [ 9., 10., 11.],
                                [18., 20., 22.]]],
                              [[[ 0.,  0.,  0.],
                                [12., 13., 14.],
                                [24., 26., 28.]],
                               [[ 0.,  0.,  0.],
                                [15., 16., 17.],
                                [30., 32., 34.]],
                               [[ 0.,  0.,  0.],
                                [18., 19., 20.],
                                [36., 38., 40.]],
                               [[ 0.,  0.,  0.],
                                [21., 22., 23.],
                                [42., 44., 46.]]]])

    def f_inner(a, b, out=None, dtype=None):
        r'''Basic inner product.'''
        if out is None:
            if dtype is None:
                return a.dot(b)
            else:
                return a.dot(b).astype(dtype)
        if f_inner.do_dtype_check:
            if dtype is not None:
                confirm_equal( out.dtype, dtype )
        if f_inner.do_base_check:
            if f_inner.base is not None:
                confirm_is(out.base, f_inner.base)
                f_inner.base_check_count = f_inner.base_check_count+1
            else:
                f_inner.base_check_count = 0
            f_inner.base = out.base
        if f_inner.do_dim_check:
            if out.shape != ():
                raise nps.NumpysaneError("mismatched lists: {} and {}". \
                                         format(out.shape, ()))
        out.setfield(a.dot(b), out.dtype)
        return out

    def f_inner_outer(a, b, out=None, dtype=None):
        r'''Basic inner AND outer product.'''
        if out is None:
            if dtype is None:
                return a.dot(b),np.outer(a,b)
            else:
                return a.dot(b).astype(dtype),np.outer(a,b).astype(dtype)
        if f_inner_outer.do_dtype_check:
            if dtype is not None:
                confirm_equal( out[0].dtype, dtype )
                confirm_equal( out[1].dtype, dtype )
        if f_inner_outer.do_base_check:
            if f_inner_outer.base is not None:
                confirm_is(out[0].base, f_inner_outer.base[0])
                confirm_is(out[1].base, f_inner_outer.base[1])
                f_inner_outer.base_check_count = f_inner_outer.base_check_count+1
            else:
                f_inner_outer.base_check_count = 0
            f_inner_outer.base = [o.base for o in out]
        if f_inner_outer.do_dim_check:
            if len(out) != 2:      raise nps.NumpysaneError("mismatched Noutput")
            if out[0].shape != (): raise nps.NumpysaneError("mismatched dimensions in output 0")
            if out[1].shape != (a.shape[0],b.shape[0]): raise nps.NumpysaneError("mismatched dimensions in output 1")
        out[0].setfield(a.dot(b),      out[0].dtype)
        out[1].setfield(np.outer(a,b), out[1].dtype)
        return out


    for f, out_ref, prototype_output, prototype_output_bad in \
        ((f_inner,      out_inner_ref, (), (1,)),
         (f_inner_outer, (out_inner_ref, out_outer_ref),
          ((),(3,3)), ((),(3,3),(1,))),
         ):


        def confirm_call_out_values(f, *args, **kwargs):
            try:
                out = f(*args, **kwargs)
                if not isinstance(out_ref, tuple):
                    confirm_equal(out,       out_ref,       "Output matches")
                    confirm_equal(out.shape, out_ref.shape, "Output shape matches")
                else:
                    for i in range(len(out_ref)):
                        confirm_equal(out[i],       out_ref[i],       "Output matches")
                        confirm_equal(out[i].shape, out_ref[i].shape, "Output shape matches")
            except:
                confirm(False, msg='broadcasted function call')



        multiple_outputs = False
        try:
            if isinstance(prototype_output[0], tuple):
                multiple_outputs = True
        except:
            pass
        def confirm_call_out_values(f, *args, **kwargs):
            try:
                out = f(*args, **kwargs)
                if not isinstance(out_ref, tuple):
                    confirm_equal(out,       out_ref,       "Output matches")
                    confirm_equal(out.shape, out_ref.shape, "Output shape matches")
                else:
                    for i in range(len(out_ref)):
                        confirm_equal(out[i],       out_ref[i],       "Output matches")
                        confirm_equal(out[i].shape, out_ref[i].shape, "Output shape matches")
            except:
                confirm(False, msg='broadcasted function call')

        # First we look at the case where broadcast_define() has no out_kwarg.
        # Then the output cannot be specified at all. If prototype_output
        # exists, then it is either used to create the output array, or to
        # validate the dimensions of output slices obtained from elsewhere. The
        # dtype is simply passed through to the inner function, is free to use
        # it, to not use it, or to crash in response (the f() function above
        # will take it; created arrays will be of that type; passed-in arrays
        # will create an error for a wrong type)
        f1 = nps.broadcast_define(prototype_input)                      (f)
        f2 = nps.broadcast_define(prototype_input,
                                  prototype_output=prototype_output)    (f)
        f3 = nps.broadcast_define(prototype_input,
                                  prototype_output=prototype_output_bad)(f)

        f.do_base_check  = False
        f.do_dtype_check = False
        f.do_dim_check   = True

        if not multiple_outputs:
            confirm_call_out_values(f1, in1, in2)
            confirm_call_out_values(f1, in1, in2, dtype=float)
            confirm_call_out_values(f1, in1, in2, dtype=int)
        else:
            confirm_raises(lambda: f1(in1, in2))
            confirm_raises(lambda: f1(in1, in2, dtype=float))
            confirm_raises(lambda: f1(in1, in2, dtype=int))
        confirm_call_out_values(f2, in1, in2)
        confirm_raises     ( lambda: f3(in1, in2) )


        # OK then. Let's now pass in an out_kwarg. Here we do not yet
        # pre-allocate an output. Thus if we don't pass in a prototype_output
        # either, the first slice will dictate the output shape, and we'll have
        # 7 inner calls into an output array (6 base comparisons). If we DO pass
        # in a prototype_output, then we will allocate immediately, and we'll
        # see 8 inner calls into an output array (7 base comparisons)
        f1 = nps.broadcast_define(prototype_input, out_kwarg="out")     (f)
        f2 = nps.broadcast_define(prototype_input, out_kwarg="out",
                                  prototype_output=prototype_output)    (f)
        f3 = nps.broadcast_define(prototype_input, out_kwarg="out",
                                  prototype_output=prototype_output_bad)(f)

        f.do_base_check  = True
        f.do_dtype_check = True
        f.do_dim_check   = True

        if not multiple_outputs:
            f.base = None
            confirm_call_out_values(f1, in1, in2)
            confirm_equal( 6, f.base_check_count )
            f.base = None
            confirm_call_out_values(f1, in1, in2, dtype=float)
            confirm_equal( 6, f.base_check_count )
            f.base = None
            confirm_call_out_values(f1, in1, in2, dtype=int)
            confirm_equal( 6, f.base_check_count )

        f.base = None
        confirm_call_out_values(f2, in1, in2)
        confirm_equal( 7, f.base_check_count )
        f.base = None
        confirm_call_out_values(f2, in1, in2, dtype=float)
        confirm_equal( 7, f.base_check_count )
        f.base = None
        confirm_call_out_values(f2, in1, in2, dtype=int)
        confirm_equal( 7, f.base_check_count )

        # Here the inner function will get an improperly-sized array to fill in.
        # broadcast_define() itself won't see any issues with this, but the
        # inner function is free to detect the error
        f.do_dim_check = False
        f.base = None
        confirm_does_not_raise( lambda: f3(in1, in2),
                                msg='broadcasted function call')
        f.do_dim_check = True
        f.base = None
        confirm_raises( lambda: f3(in1, in2) )


        # Now pre-allocate the full output array ourselves. Any prototype_output
        # we pass in is used for validation. Any dtype passed in does nothing,
        # but assertValueShape() will flag discrepancies. We use the same
        # f1,f2,f3 as above

        f.do_base_check  = True
        f.do_dtype_check = False
        f.do_dim_check   = True

        out_ref_mounted = out_ref if multiple_outputs else (out_ref,)

        # correct shape, varying dtypes
        out0 = tuple( np.empty( o.shape, dtype=float ) for o in out_ref_mounted)
        out1 = tuple( np.empty( o.shape, dtype=int ) for o in out_ref_mounted)

        # shape has too many dimensions
        out2 = tuple( np.empty( o.shape + (1,), dtype=int ) for o in out_ref_mounted)
        out3 = tuple( np.empty( o.shape + (2,), dtype=int ) for o in out_ref_mounted)
        out4 = tuple( np.empty( (1,) + o.shape, dtype=int ) for o in out_ref_mounted)
        out5 = tuple( np.empty( (2,) + o.shape, dtype=int ) for o in out_ref_mounted)

        # shape has the correct number of dimensions, but they aren't right
        out6 = tuple( np.empty( (1,) + o.shape[1:], dtype=int ) for o in out_ref_mounted)
        out7 = tuple( np.empty( o.shape[:1] + (1,), dtype=int ) for o in out_ref_mounted)

        if not multiple_outputs:
            out0 = out0[0]
            out1 = out1[0]
            out2 = out2[0]
            out3 = out3[0]
            out4 = out4[0]
            out5 = out5[0]
            out6 = out6[0]
            out7 = out7[0]

        # f1 and f2 should work exactly the same, since prototype_output is just
        # a validating parameter
        if not multiple_outputs:
            f.base = None
            assertValueShape( out_ref, out_ref.shape, f1, in1, in2, out=out0)
            confirm_equal( 7, f.base_check_count )
            f.base = None
            assertValueShape( out_ref, out_ref.shape, f1, in1, in2, out=out0, dtype=float)
            confirm_equal( 7, f.base_check_count )

            f.base = None
            assertValueShape( out_ref, out_ref.shape, f1, in1, in2, out=out1)
            confirm_equal( 7, f.base_check_count )
            f.base = None
            assertValueShape( out_ref, out_ref.shape, f1, in1, in2, out=out1, dtype=int)
            confirm_equal( 7, f.base_check_count )

        f.base = None
        confirm_call_out_values( f2, in1, in2, out=out0)
        confirm_equal( 7, f.base_check_count )
        f.base = None
        confirm_call_out_values( f2, in1, in2, out=out0, dtype=float)
        confirm_equal( 7, f.base_check_count )

        f.base = None
        confirm_call_out_values( f2, in1, in2, out=out1)
        confirm_equal( 7, f.base_check_count )
        f.base = None
        confirm_call_out_values( f2, in1, in2, out=out1, dtype=int)
        confirm_equal( 7, f.base_check_count )

        # any improperly-sized output matrices WILL be flagged if
        # prototype_output is given, and will likely be flagged if it isn't
        # also, although there are cases where this wouldn't happen. I simply
        # expect all of these to fail
        for out_misshaped in out2,out3,out4,out5,out6,out7:
            f.do_dim_check = False
            f.base = None
            confirm_raises( lambda: f2(in1, in2, out=out_misshaped) )
            f.do_dim_check = True
            f.base = None
            confirm_raises( lambda: f1(in1, in2, out=out_misshaped) )

def test_concatenation():
    r'''Checking the various concatenation functions.'''

    confirm_raises( lambda: nps.glue( arr(2,3), arr(2,3), axis=0), msg='axes are negative' )
    confirm_raises( lambda: nps.glue( arr(2,3), arr(2,3), axis=1), msg='axes are negative' )

    # basic glueing
    assertValueShape( None, (2,6),     nps.glue, arr(2,3), arr(2,3), axis=-1 )
    assertValueShape( None, (4,3),     nps.glue, arr(2,3), arr(2,3), axis=-2 )
    assertValueShape( None, (2,2,3),   nps.glue, arr(2,3), arr(2,3), axis=-3 )
    assertValueShape( None, (2,1,2,3), nps.glue, arr(2,3), arr(2,3), axis=-4 )
    confirm_raises     ( lambda: nps.glue( arr(2,3), arr(2,3)) )
    assertValueShape( None, (2,2,3),   nps.cat,  arr(2,3), arr(2,3) )

    # extra length-1 dims added as needed, data not duplicated as needed
    confirm_raises( lambda: nps.glue( arr(3),   arr(2,3), axis=-1) )
    assertValueShape( None, (3,3),     nps.glue, arr(3),   arr(2,3), axis=-2 )
    confirm_raises( lambda: nps.glue( arr(3),   arr(2,3), axis=-3) )
    confirm_raises( lambda: nps.glue( arr(3),   arr(2,3), axis=-4) )
    confirm_raises( lambda: nps.glue( arr(3),   arr(2,3)) )
    confirm_raises( lambda: nps.cat(  arr(3),   arr(2,3)) )

    confirm_raises( lambda: nps.glue( arr(2,3), arr(3),   axis=-1) )
    assertValueShape( None, (3,3),     nps.glue, arr(2,3), arr(3),   axis=-2 )
    confirm_raises( lambda: nps.glue( arr(2,3), arr(3),   axis=-3) )
    confirm_raises( lambda: nps.glue( arr(2,3), arr(3),   axis=-4) )
    confirm_raises( lambda: nps.cat(  arr(2,3), arr(3)) )

    confirm_raises( lambda: nps.glue( arr(1,3), arr(2,3), axis=-1) )
    assertValueShape( None, (3,3),     nps.glue, arr(1,3), arr(2,3), axis=-2 )
    confirm_raises( lambda: nps.glue( arr(1,3), arr(2,3), axis=-3) )
    confirm_raises( lambda: nps.glue( arr(1,3), arr(2,3), axis=-4) )
    confirm_raises( lambda: nps.cat(  arr(1,3), arr(2,3)) )

    confirm_raises( lambda: nps.glue( arr(2,3), arr(1,3), axis=-1) )
    assertValueShape( None, (3,3),     nps.glue, arr(2,3), arr(1,3), axis=-2 )
    confirm_raises( lambda: nps.glue( arr(2,3), arr(1,3), axis=-3) )
    confirm_raises( lambda: nps.glue( arr(2,3), arr(1,3), axis=-4) )
    confirm_raises( lambda: nps.cat(  arr(2,3), arr(1,3)) )

    confirm_raises( lambda: nps.glue( arr(1,3), arr(2,3), axis=-1) )
    assertValueShape( None, (3,3),     nps.glue, arr(1,3), arr(2,3), axis=-2 )
    confirm_raises( lambda: nps.glue( arr(1,3), arr(2,3), axis=-3) )
    confirm_raises( lambda: nps.glue( arr(1,3), arr(2,3), axis=-4) )
    confirm_raises( lambda: nps.cat(  arr(1,3), arr(2,3)) )

    assertValueShape( None, (3,),      nps.glue, arr(3),       np.array(()), axis=-1 )
    assertValueShape( None, (4,),      nps.glue, arr(3),       np.array(5),  axis=-1 )

    # zero-length arrays do the right thing
    confirm_raises( lambda: nps.glue( arr(0,3), arr(2,3),     axis=-1) )
    assertValueShape( None, (2,3),     nps.glue, arr(0,3), arr(2,3),     axis=-2 )
    confirm_raises( lambda: nps.glue( arr(0,3), arr(2,3),     axis=-3) )
    assertValueShape( None, (2,3),     nps.glue, arr(2,0), arr(2,3),     axis=-1 )
    confirm_raises( lambda: nps.glue( arr(2,0), arr(2,3),     axis=-2) )
    confirm_raises( lambda: nps.glue( arr(2,0), arr(2,3),     axis=-3) )

    assertValueShape( None, (2,3),     nps.glue, arr(2,0), arr(2,3),     axis=-1 )
    confirm_raises( lambda: nps.glue( arr(2,0), arr(2,3),     axis=-2) )
    confirm_raises( lambda: nps.glue( arr(2,0), arr(2,3),     axis=-3) )
    confirm_raises( lambda: nps.glue( arr(2,3), arr(0,3),     axis=-1) )
    assertValueShape( None, (2,3),     nps.glue, arr(2,3), arr(0,3),     axis=-2 )
    confirm_raises( lambda: nps.glue( arr(2,3), arr(0,3),     axis=-3) )

    assertValueShape( None, (0,5),     nps.glue, arr(0,2), arr(0,3),     axis=-1 )
    confirm_raises( lambda: nps.glue( arr(0,2), arr(0,3),     axis=-2) )
    confirm_raises( lambda: nps.glue( arr(0,2), arr(0,3),     axis=-3) )
    confirm_raises( lambda: nps.glue( arr(2,0), arr(0,3),     axis=-1) )
    confirm_raises( lambda: nps.glue( arr(2,0), arr(0,3),     axis=-2) )
    confirm_raises( lambda: nps.glue( arr(2,0), arr(0,3),     axis=-3) )
    confirm_raises( lambda: nps.glue( arr(0,2), arr(3,0),     axis=-1) )
    confirm_raises( lambda: nps.glue( arr(0,2), arr(3,0),     axis=-2) )
    confirm_raises( lambda: nps.glue( arr(0,2), arr(3,0),     axis=-3) )
    confirm_raises( lambda: nps.glue( arr(2,0), arr(3,0),     axis=-1) )
    assertValueShape( None, (5,0),     nps.glue, arr(2,0), arr(3,0),     axis=-2 )
    confirm_raises( lambda: nps.glue( arr(2,0), arr(3,0),     axis=-3) )
    assertValueShape( None, (2,0),     nps.glue, arr(1,0), arr(1,0),     axis=-2 )
    assertValueShape( None, (2,0),     nps.glue, arr(0,),  arr(1,0),     axis=-2 )
    assertValueShape( None, (2,0),     nps.glue, arr(1,0), arr(0,),      axis=-2 )
    assertValueShape( None, (0,),      nps.glue, arr(0,),  arr(0,),      axis=-1 )
    assertValueShape( None, (2,0),     nps.glue, arr(0,),  arr(0,),      axis=-2 )
    assertValueShape( None, (2,1,1,0), nps.glue, arr(0,),  arr(0,),      axis=-4 )

    assertValueShape( None, (0,),      nps.glue, arr(0,),  arr(0,),      axis=-1 )
    assertValueShape( None, (2,),      nps.glue, arr(2,),  arr(0,),      axis=-1 )
    assertValueShape( None, (2,),      nps.glue, arr(0,),  arr(2,),      axis=-1 )
    confirm_raises(   lambda: nps.glue( arr(2,),  arr(0,), axis=-2 ) )
    assertValueShape( None, (2,),      nps.glue, arr(0,),  arr(2,),      axis=-2 )

    # same as before, but np.array(()) instead of np.arange(0)
    assertValueShape( None, (0,),      nps.glue, np.array(()), np.array(()), axis=-1 )
    assertValueShape( None, (2,),      nps.glue, arr(2,),  np.array(()), axis=-1 )
    assertValueShape( None, (2,),      nps.glue, np.array(()),arr(2,),   axis=-1 )

    assertValueShape( None, (0,6),     nps.glue, arr(0,3), arr(0,3),     axis=-1 )
    assertValueShape( None, (0,3),     nps.glue, arr(0,3), arr(0,3),     axis=-2 )
    assertValueShape( None, (2,0,3),   nps.glue, arr(0,3), arr(0,3),     axis=-3 )
    confirm_raises( lambda: nps.glue( arr(3,0), arr(0,3),     axis=-1) )
    confirm_raises( lambda: nps.glue( arr(3,0), arr(0,3),     axis=-2) )
    confirm_raises( lambda: nps.glue( arr(3,0), arr(0,3),     axis=-3) )
    confirm_raises( lambda: nps.glue( arr(0,3), arr(3,0),     axis=-1) )
    confirm_raises( lambda: nps.glue( arr(0,3), arr(3,0),     axis=-2) )
    confirm_raises( lambda: nps.glue( arr(0,3), arr(3,0),     axis=-3) )
    assertValueShape( None, (3,0),     nps.glue, arr(3,0), arr(3,0),     axis=-1 )
    assertValueShape( None, (6,0),     nps.glue, arr(3,0), arr(3,0),     axis=-2 )
    assertValueShape( None, (2,3,0),   nps.glue, arr(3,0), arr(3,0),     axis=-3 )

    # legacy behavior allows one to omit the 'axis' kwarg
    nps.glue.legacy_version = '0.9'
    assertValueShape( None, (2,2,3),   nps.glue, arr(2,3), arr(2,3) )
    delattr(nps.glue, 'legacy_version')

def test_dimension_manipulation():
    r'''Checking the various functions that manipulate dimensions.'''

    assertValueShape( None, (24,),       nps.clump,        arr(2,3,4), n=5 )
    assertValueShape( None, (24,),       nps.clump,        arr(2,3,4), n=4 )
    assertValueShape( None, (24,),       nps.clump,        arr(2,3,4), n=3 )
    assertValueShape( None, (6,4),       nps.clump,        arr(2,3,4), n=2 )
    assertValueShape( None, (2,3,4),     nps.clump,        arr(2,3,4), n=1 )
    assertValueShape( None, (2,3,4),     nps.clump,        arr(2,3,4), n=0 )
    assertValueShape( None, (2,3,4),     nps.clump,        arr(2,3,4), n=1 )
    assertValueShape( None, (2,12),      nps.clump,        arr(2,3,4), n=-2 )
    assertValueShape( None, (24,),       nps.clump,        arr(2,3,4), n=-3 )
    assertValueShape( None, (24,),       nps.clump,        arr(2,3,4), n=-4 )
    assertValueShape( None, (24,),       nps.clump,        arr(2,3,4), n=-5 )

    # legacy behavior: n>0 required, and always clumps the trailing dimensions
    nps.clump.legacy_version = '0.9'
    confirm_raises     ( lambda: nps.clump(        arr(2,3,4), n=-1) )
    assertValueShape( None, (2,3,4),     nps.clump,        arr(2,3,4), n=0 )
    assertValueShape( None, (2,3,4),     nps.clump,        arr(2,3,4), n=1 )
    assertValueShape( None, (2,12),      nps.clump,        arr(2,3,4), n=2 )
    assertValueShape( None, (24,),       nps.clump,        arr(2,3,4), n=3 )
    assertValueShape( None, (24,),       nps.clump,        arr(2,3,4), n=4 )
    delattr(nps.clump, 'legacy_version')

    assertValueShape( None, (2,3,4),     nps.atleast_dims, arr(2,3,4), -1, 1 )
    assertValueShape( None, (2,3,4),     nps.atleast_dims, arr(2,3,4), -2, 1 )
    assertValueShape( None, (2,3,4),     nps.atleast_dims, arr(2,3,4), -3, 1 )
    assertValueShape( None, (1,2,3,4),   nps.atleast_dims, arr(2,3,4), -4, 1 )
    assertValueShape( None, (2,3,4),     nps.atleast_dims, arr(2,3,4), -2, 0 )
    assertValueShape( None, (2,3,4),     nps.atleast_dims, arr(2,3,4), -2, 1 )
    assertValueShape( None, (2,3,4),     nps.atleast_dims, arr(2,3,4), -2, 2 )
    confirm_raises     ( lambda: nps.atleast_dims( arr(2,3,4), -2, 3) )

    assertValueShape( None, (3,),        nps.atleast_dims, arr(3), 0 )
    confirm_raises     ( lambda: nps.atleast_dims( arr(3), 1) )
    assertValueShape( None, (3,),        nps.atleast_dims, arr(3), -1 )
    assertValueShape( None, (1,3,),      nps.atleast_dims, arr(3), -2 )

    confirm_raises     ( lambda: nps.atleast_dims( arr(), 0) )
    confirm_raises     ( lambda: nps.atleast_dims( arr(), 1) )
    assertValueShape( None, (1,),        nps.atleast_dims, arr(), -1 )
    assertValueShape( None, (1,1),       nps.atleast_dims, arr(), -2 )

    l = (-4,1)
    confirm_raises     ( lambda: nps.atleast_dims( arr(2,3,4), l) )
    l = [-4,1]
    confirm_raises     ( lambda: nps.atleast_dims( arr(2,3,4), l, -1) )
    assertValueShape( None, (1,2,3,4),   nps.atleast_dims, arr(2,3,4), l )
    confirm_equal   ( l, [-4, 2])

    assertValueShape( None, (3,4,2),     nps.mv,           arr(2,3,4), -3, -1 )
    assertValueShape( None, (3,2,4),     nps.mv,           arr(2,3,4), -3,  1 )
    assertValueShape( None, (2,1,1,3,4), nps.mv,           arr(2,3,4), -3, -5 )
    assertValueShape( None, (2,1,1,3,4), nps.mv,           arr(2,3,4),  0, -5 )

    assertValueShape( None, (4,3,2),     nps.xchg,         arr(2,3,4), -3, -1 )
    assertValueShape( None, (3,2,4),     nps.xchg,         arr(2,3,4), -3,  1 )
    assertValueShape( None, (2,1,1,3,4), nps.xchg,         arr(2,3,4), -3, -5 )
    assertValueShape( None, (2,1,1,3,4), nps.xchg,         arr(2,3,4),  0, -5 )

    assertValueShape( None, (2,4,3),     nps.transpose,    arr(2,3,4) )
    assertValueShape( None, (4,3),       nps.transpose,    arr(3,4) )
    assertValueShape( None, (4,1),       nps.transpose,    arr(4) )

    assertValueShape( None, (1,2,3,4),   nps.dummy,        arr(2,3,4),  0 )
    assertValueShape( None, (2,1,3,4),   nps.dummy,        arr(2,3,4),  1 )
    assertValueShape( None, (2,3,4,1),   nps.dummy,        arr(2,3,4), -1 )
    assertValueShape( None, (2,3,1,4),   nps.dummy,        arr(2,3,4), -2 )
    assertValueShape( None, (2,3,1,1,4), nps.dummy,        arr(2,3,4), -2, -2 )
    assertValueShape( None, (2,1,3,4),   nps.dummy,        arr(2,3,4), -3 )
    assertValueShape( None, (1,2,3,4),   nps.dummy,        arr(2,3,4), -4 )
    assertValueShape( None, (1,1,2,3,4), nps.dummy,        arr(2,3,4), -5 )
    assertValueShape( None, (2,3,1,4),   nps.dummy,        arr(2,3,4),  2 )
    confirm_raises     ( lambda: nps.dummy(        arr(2,3,4),  3) )

    assertValueShape( None, (2,4,3),     nps.reorder,      arr(2,3,4),  0, -1, 1 )
    assertValueShape( None, (3,4,2),     nps.reorder,      arr(2,3,4),  -2, -1, 0 )
    assertValueShape( None, (1,3,1,4,2), nps.reorder,      arr(2,3,4),  -4, -2, -5, -1, 0 )
    confirm_raises     ( lambda: nps.reorder(      arr(2,3,4),  -4, -2, -5, -1, 0, 5),
                         msg='reorder barfs on out-of-bounds dimensions' )

def test_inner():
    r'''Testing the broadcasted inner product'''

    assertResult_inoutplace(  np.array([[[  30,  255,  730],
                                          [ 180,  780, 1630]],

                                         [[ 180,  780, 1630],
                                          [1455, 2430, 3655]],

                                         [[ 330, 1305, 2530],
                                          [2730, 4080, 5680]],

                                         [[ 480, 1830, 3430],
                                          [4005, 5730, 7705.0]]]),
                  nps.inner, arr(2,3,5), arr(4,1,3,5), out_inplace_dtype=float )

    assertResult_inoutplace(  np.array([[[  30,  255,  730],
                                          [ 180,  780, 1630]],

                                         [[ 180,  780, 1630],
                                          [1455, 2430, 3655]],

                                         [[ 330, 1305, 2530],
                                          [2730, 4080, 5680]],

                                         [[ 480, 1830, 3430],
                                          [4005, 5730, 7705.0]]]),
                              nps.inner, arr(2,3,5), arr(4,1,3,5), dtype=float,
                              out_inplace_dtype=float )

    output = np.empty((4,2,3), dtype=float)
    confirm_raises( lambda: nps.inner( arr(2,3,5), arr(4,1,3,5), dtype=int, out=output ),
                    "inner(out=out, dtype=dtype) have out=dtype==dtype" )

    assertResult_inoutplace( np.array((24+148j)),
                              nps.dot,
                              np.array(( 1 + 2j, 3 + 4j, 5 + 6j)),
                              np.array(( 1 + 2j, 3 + 4j, 5 + 6j)) + 5,
                              out_inplace_dtype=np.complex)

    assertResult_inoutplace( np.array((136-60j)),
                              nps.vdot,
                              np.array(( 1 + 2j, 3 + 4j, 5 + 6j)),
                              np.array(( 1 + 2j, 3 + 4j, 5 + 6j)) + 5,
                              out_inplace_dtype=np.complex)

    # complex values AND non-trivial dimensions
    a = arr(  2,3,5).astype(np.complex)
    b = arr(4,1,3,5).astype(np.complex)
    a += a*a * 1j
    b -= b * 1j

    dot_ref = np.array([[[  130.0   +70.0j,    2180.0   +1670.0j,   9730.0   +8270.0j],
                        [  3430.0  +3070.0j,  18230.0  +16670.0j,  46030.0  +42770.0j]],
                       [[  730.0   +370.0j,   6530.0   +4970.0j,   21580.0  +18320.0j],
                        [  26530.0 +23620.0j, 56330.0  +51470.0j,  102880.0 +95570.0j]],
                       [[  1330.0  +670.0j,   10880.0  +8270.0j,   33430.0  +28370.0j],
                        [  49630.0 +44170.0j, 94430.0  +86270.0j,  159730.0 +148370.0j]],
                       [[  1930.0  +970.0j,   15230.0  +11570.0j,  45280.0  +38420.0j],
                        [  72730.0 +64720.0j, 132530.0 +121070.0j, 216580.0 +201170.0j]]])
    vdot_ref = np.array([[[ -70.0    -130.0j,   -1670.0   -2180.0j,   -8270.0   -9730.0j],
                         [  -3070.0  -3430.0j,  -16670.0  -18230.0j,  -42770.0  -46030.0j]],
                        [[  -370.0   -730.0j,   -4970.0   -6530.0j,   -18320.0  -21580.0j],
                         [  -23620.0 -26530.0j, -51470.0  -56330.0j,  -95570.0  -102880.0j]],
                        [[  -670.0   -1330.0j,  -8270.0   -10880.0j,  -28370.0  -33430.0j],
                         [  -44170.0 -49630.0j, -86270.0  -94430.0j,  -148370.0 -159730.0j]],
                        [[  -970.0   -1930.0j,  -11570.0  -15230.0j,  -38420.0  -45280.0j],
                         [  -64720.0 -72730.0j, -121070.0 -132530.0j, -201170.0 -216580.0j]]])

    assertResult_inoutplace( dot_ref,
                              nps.dot,
                              a, b,
                              out_inplace_dtype=np.complex)

    assertResult_inoutplace( vdot_ref,
                              nps.vdot,
                              a, b,
                              out_inplace_dtype=np.complex)


def test_mag():
    r'''Testing the broadcasted magnitude product'''

    # input is a 1D array of integers, no output dtype specified. Output
    # should be a floating-point scalar
    assertResult_inoutplace(  np.sqrt(nps.norm2(np.arange(5))),
                              nps.mag, arr(5, dtype=int) )

    # input is a 1D array of floats, no output dtype specified. Output
    # should be a floating-point scalar
    assertResult_inoutplace(  np.sqrt(nps.norm2(np.arange(5))),
                              nps.mag, arr(5, dtype=float) )

    # input is a 1D array of integers, output dtype=int. Output should be an
    # integer scalar
    output = np.empty((), dtype=int)
    nps.mag( np.arange(5, dtype=int),
             out = output )
    confirm_equal(int(np.sqrt(nps.norm2(np.arange(5)))), output)

    # input is a 1D array of integers, output dtype=float. Output should be an
    # float scalar
    output = np.empty((), dtype=float)
    nps.mag( np.arange(5, dtype=int),
             out = output )
    confirm_equal(np.sqrt(nps.norm2(np.arange(5))), output)

    # input is a 2D array of integers, no output dtype specified. Output
    # should be a floating-point 1D vector
    assertResult_inoutplace(  np.sqrt(np.array(( nps.norm2(np.arange(5)),
                                                 nps.norm2(np.arange(5,10))))),
                              nps.mag, arr(2,5, dtype=int) )

    # input is a 2D array of floats, no output dtype specified. Output
    # should be a floating-point 1D vector
    assertResult_inoutplace(  np.sqrt(np.array(( nps.norm2(np.arange(5)),
                                                 nps.norm2(np.arange(5,10))))),
                              nps.mag, arr(2,5, dtype=float) )

    # input is a 2D array of integers, output dtype=int. Output should be an
    # array of integers
    output = np.empty((2,), dtype=int)
    nps.mag( arr(2,5, dtype=int),
             out = output )
    confirm_equal(np.sqrt(np.array(( nps.norm2(np.arange(5)),
                                     nps.norm2(np.arange(5,10))))).astype(int),
                  output)

    # input is a 2D array of integers, output dtype=float. Output should be an
    # array of floats
    output = np.empty((2,), dtype=float)
    nps.mag( arr(2,5, dtype=int),
             out = output )
    confirm_equal(np.sqrt(np.array(( nps.norm2(np.arange(5)),
                                     nps.norm2(np.arange(5,10))))),
                  output)


def test_outer():
    r'''Testing the broadcasted outer product'''

    # comes from PDL. numpy has a reversed axis ordering convention from
    # PDL, so I transpose the array before comparing
    ref = nps.transpose(
        np.array([[[[[0,0,0,0,0],[0,1,2,3,4],[0,2,4,6,8],[0,3,6,9,12],[0,4,8,12,16]],
                    [[25,30,35,40,45],[30,36,42,48,54],[35,42,49,56,63],[40,48,56,64,72],[45,54,63,72,81]],
                    [[100,110,120,130,140],[110,121,132,143,154],[120,132,144,156,168],[130,143,156,169,182],[140,154,168,182,196]]],
                   [[[0,0,0,0,0],[15,16,17,18,19],[30,32,34,36,38],[45,48,51,54,57],[60,64,68,72,76]],
                    [[100,105,110,115,120],[120,126,132,138,144],[140,147,154,161,168],[160,168,176,184,192],[180,189,198,207,216]],
                    [[250,260,270,280,290],[275,286,297,308,319],[300,312,324,336,348],[325,338,351,364,377],[350,364,378,392,406]]]],
                  [[[[0,15,30,45,60],[0,16,32,48,64],[0,17,34,51,68],[0,18,36,54,72],[0,19,38,57,76]],
                    [[100,120,140,160,180],[105,126,147,168,189],[110,132,154,176,198],[115,138,161,184,207],[120,144,168,192,216]],
                    [[250,275,300,325,350],[260,286,312,338,364],[270,297,324,351,378],[280,308,336,364,392],[290,319,348,377,406]]],
                   [[[225,240,255,270,285],[240,256,272,288,304],[255,272,289,306,323],[270,288,306,324,342],[285,304,323,342,361]],
                    [[400,420,440,460,480],[420,441,462,483,504],[440,462,484,506,528],[460,483,506,529,552],[480,504,528,552,576]],
                    [[625,650,675,700,725],[650,676,702,728,754],[675,702,729,756,783],[700,728,756,784,812],[725,754,783,812,841]]]],
                  [[[[0,30,60,90,120],[0,31,62,93,124],[0,32,64,96,128],[0,33,66,99,132],[0,34,68,102,136]],
                    [[175,210,245,280,315],[180,216,252,288,324],[185,222,259,296,333],[190,228,266,304,342],[195,234,273,312,351]],
                    [[400,440,480,520,560],[410,451,492,533,574],[420,462,504,546,588],[430,473,516,559,602],[440,484,528,572,616]]],
                   [[[450,480,510,540,570],[465,496,527,558,589],[480,512,544,576,608],[495,528,561,594,627],[510,544,578,612,646]],
                    [[700,735,770,805,840],[720,756,792,828,864],[740,777,814,851,888],[760,798,836,874,912],[780,819,858,897,936]],
                    [[1000,1040,1080,1120,1160],[1025,1066,1107,1148,1189],[1050,1092,1134,1176,1218],[1075,1118,1161,1204,1247],[1100,1144,1188,1232,1276]]]],
                  [[[[0,45,90,135,180],[0,46,92,138,184],[0,47,94,141,188],[0,48,96,144,192],[0,49,98,147,196]],
                    [[250,300,350,400,450],[255,306,357,408,459],[260,312,364,416,468],[265,318,371,424,477],[270,324,378,432,486]],
                    [[550,605,660,715,770],[560,616,672,728,784],[570,627,684,741,798],[580,638,696,754,812],[590,649,708,767,826]]],
                   [[[675,720,765,810,855],[690,736,782,828,874],[705,752,799,846,893],[720,768,816,864,912],[735,784,833,882,931]],
                    [[1000,1050,1100,1150,1200],[1020,1071,1122,1173,1224],[1040,1092,1144,1196,1248],[1060,1113,1166,1219,1272],[1080,1134,1188,1242,1296]],
                    [[1375,1430,1485,1540,1595],[1400,1456,1512,1568,1624],[1425,1482,1539,1596,1653],[1450,1508,1566,1624,1682],[1475,1534,1593,1652,1711]]]]]))

    assertResult_inoutplace( ref,
                              nps.outer, arr(2,3,5), arr(4,1,3,5),
                              out_inplace_dtype=float )

    # unequal dimensions.
    a = arr(1,3,1,4)
    b = arr(  3,7,3)
    ref = nps.matmult( nps.dummy(a, -1),
                       nps.dummy(b, -2))
    assertResult_inoutplace( ref,
                             nps.outer, a, b,
                             out_inplace_dtype=float )


def test_matmult():
    r'''Testing the broadcasted matrix multiplication'''
    assertValueShape( None, (4,2,3,5), nps.matmult, arr(2,3,7), arr(4,1,7,5) )

    ref = np.array([[[[  42,   48,   54],
                      [ 114,  136,  158]],

                     [[ 114,  120,  126],
                      [ 378,  400,  422]]],


                    [[[ 186,  224,  262],
                      [ 258,  312,  366]],

                     [[ 642,  680,  718],
                      [ 906,  960, 1014]]]])

    assertResult_inoutplace( ref,
                              nps.matmult2, arr(2,1,2,4), arr(2,4,3),
                              out_inplace_dtype=float )

    ref2 = np.array([[[[  156.], [  452.]],
                      [[  372.], [ 1244.]]],
                     [[[  748.], [ 1044.]],
                      [[ 2116.], [ 2988.]]]])

    assertResult_inoutplace(ref2,
                             nps.matmult2, arr(2,1,2,4), nps.matmult2(arr(2,4,3), arr(3,1)))

    # not doing assertResult_inoutplace() because matmult() doesn't take an
    # 'out' kwarg
    confirm_equal(ref2,
                  nps.matmult(arr(2,1,2,4), arr(2,4,3), arr(3,1)))

    # checking the null-dimensionality logic
    A = arr(2,3)
    assertResult_inoutplace( nps.inner(nps.transpose(A), np.arange(2)),
                             nps.matmult2,
                             np.arange(2), A )

    A = arr(3)
    assertResult_inoutplace( A*2,
                             nps.matmult2,
                             np.array([2]), A )

    A = arr(3)
    assertResult_inoutplace( A*2,
                             nps.matmult2,
                             np.array(2), A )




test_broadcasting()
test_broadcasting_into_output()
test_concatenation()
test_dimension_manipulation()
test_inner()
test_mag()
test_outer()
test_matmult()

finish()
