#!/usr/bin/python2

import unittest

import numpy as np
import numpysane as nps
from functools import reduce

def arr(*shape):
    r'''Return an arange() array of the given shape.'''
    if len(shape) == 0:
        return np.array(3)
    product = reduce( lambda x,y: x*y, shape)
    return np.arange(product).reshape(*shape)

class TestNumpysane(unittest.TestCase):

    def assertListEqual(self, s1, s2):
        r'''This is unittest.TestCase.assertListEqual(), but not retarded.

        That function barfs when fed (), and this one does not.

        '''
        self.assertEqual(len(s1), len(s2), msg="Lists {} and {} do not match".format(s1,s2))
        for v1,v2 in zip(s1,s2):
            self.assertEqual(v1,v2, msg="Lists {} and {} do not match".format(s1,s2))

    def assertNumpyAlmostEqual(self, first, second):
        self.assertListEqual(first.shape, second.shape)
        diff = first - second
        diff = diff.ravel().astype(float)
        rms = np.sqrt(diff.dot(diff) / diff.size)
        self.assertLess( rms, 1e-6, msg='matrix discrepancy:\n{} vs\n{}. Diff:\n{}'.format(first,second,diff) )

    def assertError(self, f, *args, **kwargs):
        r'''Convenience wrapper for my use of assertRaises()'''
        return self.assertRaises((nps.NumpysaneError, ValueError), f, *args, **kwargs)

    def assertValueShape(self, value, s, f, *args, **kwargs):
        r'''Makes sure a given call produces a given value and shape.

        It is redundant to specify both, but it makes it clear I'm asking for
        what I think I'm asking. The value check can be skipped by passing None.

        '''
        res = f(*args, **kwargs)
        if s is not None:
            self.assertListEqual(res.shape, s)
        if value is not None:
            self.assertNumpyAlmostEqual(res, value)
        if 'dtype' in kwargs:
            self.assertEqual(res.dtype, kwargs['dtype'])

    def test_broadcasting(self):
        r'''Checking broadcasting rules.'''
        @nps.broadcast_define( (('n',), ('n',)) )
        def f1(a, b):
            r'''Basic inner product.'''
            return a.dot(b)

        self.assertValueShape( np.array(5),                (),     f1, arr(3),     arr(3))
        self.assertValueShape( np.array((5,14)),           (2,),   f1, arr(2,3),   arr(3))
        self.assertValueShape( np.array((5,14)),           (2,),   f1, arr(3),     arr(2,3))
        self.assertValueShape( np.array(((5,14),)),        (1,2,), f1, arr(1,2,3), arr(3))
        self.assertValueShape( np.array(((5,),(14,))),     (2,1,), f1, arr(2,1,3), arr(3))
        self.assertValueShape( np.array((5,14)),           (2,),   f1, arr(2,3),   arr(1,3))
        self.assertValueShape( np.array((5,14)),           (2,),   f1, arr(1,3),   arr(2,3))
        self.assertValueShape( np.array(((5,14),)),        (1,2,), f1, arr(1,2,3), arr(1,3))
        self.assertValueShape( np.array(((5,),(14,))),     (2,1,), f1, arr(2,1,3), arr(1,3))
        self.assertValueShape( np.array(((5,14),(14,50))), (2,2,), f1, arr(2,1,3), arr(2,3))
        self.assertValueShape( np.array(((5,14),(14,50))), (2,2,), f1, arr(2,1,3), arr(1,2,3))

        # wrong number of args
        self.assertError( f1, arr(3) )

        # mismatched args
        self.assertError( f1, arr(3),arr(5) )
        self.assertError( f1, arr(2,3),arr(4,3) )
        self.assertError( f1, arr(3,3,3),arr(2,1,3) )
        self.assertError( f1, arr(1,2,4),arr(2,1,3) )


        # fancier function, has some preset dimensions
        @nps.broadcast_define( ((3,), ('n',3), ('n',), ('m',)) )
        def f2(a,b,c,d):
            return d

        n=4
        m=6
        d = np.arange(m)

        self.assertValueShape( d, (m,),
                               f2,
                               arr(  3),
                               arr(n,3),
                               arr(  n),
                               arr(  m))
        self.assertValueShape( np.array((d,)), (1,m),
                               f2,
                               arr(1,    3),
                               arr(1,  n,3),
                               arr(      n),
                               arr(1,    m))
        self.assertValueShape( np.array((d,)), (1,m,),
                               f2,
                               arr(1,    3),
                               arr(1,  n,3),
                               arr(      n),
                               arr(      m))
        self.assertValueShape( np.array((d,d+m,d+2*m,d+3*m,d+4*m)), (5,m),
                               f2,
                               arr(5,    3),
                               arr(5,  n,3),
                               arr(      n),
                               arr(5,    m))
        self.assertValueShape( np.array(((d,d+m,d+2*m,d+3*m,d+4*m),)), (1,5,m),
                               f2,
                               arr(1,5,    3),
                               arr(  5,  n,3),
                               arr(        n),
                               arr(  5,    m))
        self.assertValueShape( np.array(((d,d+m,d+2*m,d+3*m,d+4*m), (d,d+m,d+2*m,d+3*m,d+4*m))), (2,5,m),
                               f2,
                               arr(1,5,    3),
                               arr(2,5,  n,3),
                               arr(        n),
                               arr(  5,    m))
        self.assertValueShape( np.array(((d,d+m,d+2*m,d+3*m,d+4*m), (d,d+m,d+2*m,d+3*m,d+4*m))), (2,5,m),
                               f2,
                               arr(1,5,    3),
                               arr(2,1,  n,3),
                               arr(        n),
                               arr(  5,    m))
        self.assertValueShape( np.array((((d,d,d,d,d), (d,d,d,d,d)),)), (1,2,5,m),
                               f2,
                               arr(1,1,5,    3),
                               arr(1,2,1,  n,3),
                               arr(1,        n),
                               arr(1,  1,    m))

        # mismatched args
        self.assertError( f2,
                          arr(5,    3),
                          arr(5,  n,3),
                          arr(      m),
                          arr(5,    m))
        self.assertError( f2,
                          arr(5,    2),
                          arr(5,  n,3),
                          arr(      n),
                          arr(5,    m))
        self.assertError( f2,
                          arr(5,    2),
                          arr(5,  n,2),
                          arr(      n),
                          arr(5,    m))
        self.assertError( f2,
                          arr(1,    3),
                          arr(1,  n,3),
                          arr(      5*n),
                          arr(1,    m))

        # Make sure extra args and the kwargs are passed through
        @nps.broadcast_define( ((3,), ('n',3), ('n',), ('m',)) )
        def f3(a,b,c,d, e,f, *args, **kwargs):
            def val_or_0(x): return x if x else 0
            return np.array( (a[0], val_or_0(e), val_or_0(f), val_or_0(args[0]), val_or_0( kwargs.get('xxx'))) )
        self.assertValueShape( np.array( ((0, 1, 2, 3, 6), (3, 1, 2, 3, 6)) ), (2,5),
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

        self.assertValueShape( np.array((5,5)), (2,),
                               f4,
                               arr(      3),
                               arr(1,  3,4),
                               arr(2,    2),
                               np.array(5))
        self.assertValueShape( np.array((5,5)), (2,),
                               f4,
                               arr(      3),
                               arr(1,  3,4),
                               arr(2,    2),
                               5)
        self.assertValueShape( np.array(((0,1,5),(2,3,5))), (2,3),
                               f5,
                               arr(      3),
                               arr(1,  3,4),
                               arr(2,    2),
                               np.array(5))
        self.assertValueShape( np.array(((0,1,5),(2,3,5))), (2,3),
                               f5,
                               arr(      3),
                               arr(1,  3,4),
                               arr(2,    2),
                               5)
        self.assertError(      f5,
                               arr(      3),
                               arr(1,  3,4),
                               arr(2,    2),
                               arr(5))

        # Test the generator
        i=0
        for s in nps.broadcast_generate( (('n',), ('n','m'), (2,), ()),
                                         (arr(      3),
                                          arr(1,  3,4),
                                          arr(2,    2),
                                          np.array(5)) ):
            self.assertNumpyAlmostEqual( arr(3),       s[0] )
            self.assertNumpyAlmostEqual( arr(3,4),     s[1] )
            self.assertNumpyAlmostEqual( arr(2) + 2*i, s[2] )
            self.assertNumpyAlmostEqual( np.array(5),  s[3] )
            self.assertListEqual( s[3].shape, ())
            i = i+1

        i=0
        for s in nps.broadcast_generate( (('n',), ('n','m'), (2,), ()),
                                         (arr(      3),
                                          arr(1,  3,4),
                                          arr(2,    2),
                                          5) ):
            self.assertNumpyAlmostEqual( arr(3),       s[0] )
            self.assertNumpyAlmostEqual( arr(3,4),     s[1] )
            self.assertNumpyAlmostEqual( arr(2) + 2*i, s[2] )
            self.assertNumpyAlmostEqual( np.array(5),  s[3] )
            self.assertListEqual( s[3].shape, ())
            i = i+1

        i=0
        for s in nps.broadcast_generate( (('n',), ('n','m'), (2,), ()),
                                         (arr(      3),
                                          arr(1,  3,4),
                                          arr(2,    2),
                                          arr(2)) ):
            self.assertNumpyAlmostEqual( arr(3),       s[0] )
            self.assertNumpyAlmostEqual( arr(3,4),     s[1] )
            self.assertNumpyAlmostEqual( arr(2) + 2*i, s[2] )
            self.assertNumpyAlmostEqual( np.array(i),  s[3] )
            self.assertListEqual( s[3].shape, ())
            i = i+1

    def test_broadcasting_into_output(self):
        r'''Checking broadcasting with the output array defined.'''

        # I think about all 2^4 = 16 combinations:
        #
        # broadcast_define(): yes/no prototype_output, out_kwarg
        # broadcasted call:   yes/no dtype, output

        prototype = (('n',), ('n',))
        in1, in2 = arr(3), arr(2,4,3)
        out_ref = np.array([[ 5, 14, 23, 32],
                            [41, 50, 59, 68]])
        outshape_ref = (2,4)

        def f(a, b, out=None, dtype=None):
            r'''Basic inner product.'''

            if out is None:
                if dtype is None:
                    return a.dot(b)
                else:
                    return a.dot(b).astype(dtype)

            if f.do_dtype_check:
                if dtype is not None:
                    self.assertEqual( out.dtype, dtype )

            if f.do_base_check:
                if f.base is not None:
                    self.assertIs(out.base, f.base)
                    f.base_check_count = f.base_check_count+1
                else:
                    f.base_check_count = 0

                f.base = out.base

            if f.do_dim_check:
                if out.shape != ():
                    raise nps.NumpysaneError("mismatched lists")

            out.setfield(a.dot(b), out.dtype)
            return out

        # First we look at the case where broadcast_define() has no out_kwarg.
        # Then the output cannot be specified at all. If prototype_output
        # exists, then it is either used to create the output array, or to
        # validate the dimensions of output slices obtained from elsewhere. The
        # dtype is simply passed through to the inner function, is free to use
        # it, to not use it, or to crash in response (the f() function above
        # will take it; created arrays will be of that type; passed-in arrays
        # will create an error for a wrong type)
        f1 = nps.broadcast_define(prototype)                        (f)
        f2 = nps.broadcast_define(prototype, prototype_output=()   )(f)
        f3 = nps.broadcast_define(prototype, prototype_output=(1,) )(f)

        f.do_base_check  = False
        f.do_dtype_check = False
        f.do_dim_check   = True
        self.assertValueShape( out_ref, outshape_ref, f1, in1, in2)
        self.assertValueShape( out_ref, outshape_ref, f1, in1, in2, dtype=float)
        self.assertValueShape( out_ref, outshape_ref, f1, in1, in2, dtype=int)
        self.assertValueShape( out_ref, outshape_ref, f2, in1, in2)
        self.assertError     (                        f3, in1, in2)


        # OK then. Let's now pass in an out_kwarg. Here we do not yet
        # pre-allocate an output. Thus if we don't pass in a prototype_output
        # either, the first slice will dictate the output shape, and we'll have
        # 7 inner calls into an output array (6 base comparisons). If we DO pass
        # in a prototype_output, then we will allocate immediately, and we'll
        # see 8 inner calls into an output array (7 base comparisons)
        f1 = nps.broadcast_define(prototype, out_kwarg="out")                        (f)
        f2 = nps.broadcast_define(prototype, out_kwarg="out", prototype_output=()   )(f)
        f3 = nps.broadcast_define(prototype, out_kwarg="out", prototype_output=(1,) )(f)

        f.do_base_check  = True
        f.do_dtype_check = True
        f.do_dim_check   = True

        f.base = None
        self.assertValueShape( out_ref, outshape_ref, f1, in1, in2)
        self.assertEqual( 6, f.base_check_count )
        f.base = None
        self.assertValueShape( out_ref, outshape_ref, f1, in1, in2, dtype=float)
        self.assertEqual( 6, f.base_check_count )
        f.base = None
        self.assertValueShape( out_ref, outshape_ref, f1, in1, in2, dtype=int)
        self.assertEqual( 6, f.base_check_count )

        f.base = None
        self.assertValueShape( out_ref, outshape_ref, f2, in1, in2)
        self.assertEqual( 7, f.base_check_count )
        f.base = None
        self.assertValueShape( out_ref, outshape_ref, f2, in1, in2, dtype=float)
        self.assertEqual( 7, f.base_check_count )
        f.base = None
        self.assertValueShape( out_ref, outshape_ref, f2, in1, in2, dtype=int)
        self.assertEqual( 7, f.base_check_count )

        # Here the inner function will get an improperly-sized array to fill in.
        # broadcast_define() itself won't see any issues with this, but the
        # inner function is free to detect the error
        f.do_dim_check = False
        f.base = None
        self.assertValueShape( None, None, f3, in1, in2)
        f.do_dim_check = True
        f.base = None
        self.assertError( f3, in1, in2)



        # Now pre-allocate the full output array ourselves. Any prototype_output
        # we pass in is used for validation. Any dtype passed in does nothing,
        # but assertValueShape() will flag discrepancies. We use the same
        # f1,f2,f3 as above

        f.do_base_check  = True
        f.do_dtype_check = False
        f.do_dim_check   = True

        # correct shape, varying dtypes
        out0 = np.empty( outshape_ref, dtype=float )
        out1 = np.empty( outshape_ref, dtype=int )

        # shape has too many dimensions
        out2 = np.empty( outshape_ref + (1,), dtype=int )
        out3 = np.empty( outshape_ref + (2,), dtype=int )
        out4 = np.empty( (1,) + outshape_ref, dtype=int )
        out5 = np.empty( (2,) + outshape_ref, dtype=int )

        # shape has the correct number of dimensions, but they aren't right
        out6 = np.empty( (1,) + outshape_ref[1:], dtype=int )
        out7 = np.empty( outshape_ref[:1] + (1,), dtype=int )


        # f1 and f2 should work exactly the same, since prototype_output is just
        # a validating parameter
        for f12 in f1,f2:
            f.base = None
            self.assertValueShape( out_ref, outshape_ref, f12, in1, in2, out=out0)
            self.assertEqual( 7, f.base_check_count )
            f.base = None
            self.assertValueShape( out_ref, outshape_ref, f12, in1, in2, out=out0, dtype=float)
            self.assertEqual( 7, f.base_check_count )

            f.base = None
            self.assertValueShape( out_ref, outshape_ref, f12, in1, in2, out=out1)
            self.assertEqual( 7, f.base_check_count )
            f.base = None
            self.assertValueShape( out_ref, outshape_ref, f12, in1, in2, out=out1, dtype=int)
            self.assertEqual( 7, f.base_check_count )

        # any improperly-sized output matrices WILL be flagged if
        # prototype_output is given, and will likely be flagged if it isn't
        # also, although there are cases where this wouldn't happen. I simply
        # expect all of these to fail
        for out_misshaped in out2,out3,out4,out5,out6,out7:
            f.do_dim_check = False
            f.base = None
            self.assertError( f2, in1, in2, out=out_misshaped)
            f.do_dim_check = True
            f.base = None
            self.assertError( f1, in1, in2, out=out_misshaped)



    def test_concatenation(self):
        r'''Checking the various concatenation functions.'''

        # axes must be negative
        self.assertError( nps.glue, arr(2,3), arr(2,3), axis=0 )
        self.assertError( nps.glue, arr(2,3), arr(2,3), axis=1 )

        # basic glueing
        self.assertValueShape( None, (2,6),     nps.glue, arr(2,3), arr(2,3), axis=-1 )
        self.assertValueShape( None, (4,3),     nps.glue, arr(2,3), arr(2,3), axis=-2 )
        self.assertValueShape( None, (2,2,3),   nps.glue, arr(2,3), arr(2,3), axis=-3 )
        self.assertValueShape( None, (2,1,2,3), nps.glue, arr(2,3), arr(2,3), axis=-4 )
        self.assertValueShape( None, (2,2,3),   nps.glue, arr(2,3), arr(2,3) )
        self.assertValueShape( None, (2,2,3),   nps.cat,  arr(2,3), arr(2,3) )

        # extra length-1 dims added as needed, data not duplicated as needed
        self.assertError(                       nps.glue, arr(3),   arr(2,3), axis=-1 )
        self.assertValueShape( None, (3,3),     nps.glue, arr(3),   arr(2,3), axis=-2 )
        self.assertError(                       nps.glue, arr(3),   arr(2,3), axis=-3 )
        self.assertError(                       nps.glue, arr(3),   arr(2,3), axis=-4 )
        self.assertError(                       nps.glue, arr(3),   arr(2,3) )
        self.assertError(                       nps.cat,  arr(3),   arr(2,3) )

        self.assertError(                       nps.glue, arr(2,3), arr(3),   axis=-1 )
        self.assertValueShape( None, (3,3),     nps.glue, arr(2,3), arr(3),   axis=-2 )
        self.assertError(                       nps.glue, arr(2,3), arr(3),   axis=-3 )
        self.assertError(                       nps.glue, arr(2,3), arr(3),   axis=-4 )
        self.assertError(                       nps.cat,  arr(2,3), arr(3) )

        self.assertError(                       nps.glue, arr(1,3), arr(2,3), axis=-1 )
        self.assertValueShape( None, (3,3),     nps.glue, arr(1,3), arr(2,3), axis=-2 )
        self.assertError(                       nps.glue, arr(1,3), arr(2,3), axis=-3 )
        self.assertError(                       nps.glue, arr(1,3), arr(2,3), axis=-4 )
        self.assertError(                       nps.cat,  arr(1,3), arr(2,3) )

        self.assertError(                       nps.glue, arr(2,3), arr(1,3), axis=-1 )
        self.assertValueShape( None, (3,3),     nps.glue, arr(2,3), arr(1,3), axis=-2 )
        self.assertError(                       nps.glue, arr(2,3), arr(1,3), axis=-3 )
        self.assertError(                       nps.glue, arr(2,3), arr(1,3), axis=-4 )
        self.assertError(                       nps.cat,  arr(2,3), arr(1,3) )

        self.assertError(                       nps.glue, arr(1,3), arr(2,3), axis=-1 )
        self.assertValueShape( None, (3,3),     nps.glue, arr(1,3), arr(2,3), axis=-2 )
        self.assertError(                       nps.glue, arr(1,3), arr(2,3), axis=-3 )
        self.assertError(                       nps.glue, arr(1,3), arr(2,3), axis=-4 )
        self.assertError(                       nps.cat,  arr(1,3), arr(2,3) )

    def test_dimension_manipulation(self):
        r'''Checking the various functions that manipulate dimensions.'''

        self.assertError     (                    nps.clump,        arr(2,3,4), n=-1 )
        self.assertValueShape( None, (2,3,4),     nps.clump,        arr(2,3,4), n=0 )
        self.assertValueShape( None, (2,3,4),     nps.clump,        arr(2,3,4), n=1 )
        self.assertValueShape( None, (2,12),      nps.clump,        arr(2,3,4), n=2 )
        self.assertValueShape( None, (24,),       nps.clump,        arr(2,3,4), n=3 )
        self.assertValueShape( None, (24,),       nps.clump,        arr(2,3,4), n=4 )
        self.assertValueShape( None, (24,),       nps.clump,        arr(2,3,4), n=5 )

        self.assertValueShape( None, (2,3,4),     nps.atleast_dims, arr(2,3,4), -1, 1 )
        self.assertValueShape( None, (2,3,4),     nps.atleast_dims, arr(2,3,4), -2, 1 )
        self.assertValueShape( None, (2,3,4),     nps.atleast_dims, arr(2,3,4), -3, 1 )
        self.assertValueShape( None, (1,2,3,4),   nps.atleast_dims, arr(2,3,4), -4, 1 )
        self.assertValueShape( None, (2,3,4),     nps.atleast_dims, arr(2,3,4), -2, 0 )
        self.assertValueShape( None, (2,3,4),     nps.atleast_dims, arr(2,3,4), -2, 1 )
        self.assertValueShape( None, (2,3,4),     nps.atleast_dims, arr(2,3,4), -2, 2 )
        self.assertError     (                    nps.atleast_dims, arr(2,3,4), -2, 3 )

        self.assertValueShape( None, (3,),        nps.atleast_dims, arr(3), 0 )
        self.assertError     (                    nps.atleast_dims, arr(3), 1 )
        self.assertValueShape( None, (3,),        nps.atleast_dims, arr(3), -1 )
        self.assertValueShape( None, (1,3,),      nps.atleast_dims, arr(3), -2 )

        self.assertError     (                    nps.atleast_dims, arr(), 0 )
        self.assertError     (                    nps.atleast_dims, arr(), 1 )
        self.assertValueShape( None, (1,),        nps.atleast_dims, arr(), -1 )
        self.assertValueShape( None, (1,1),       nps.atleast_dims, arr(), -2 )

        l = (-4,1)
        self.assertError     (                    nps.atleast_dims, arr(2,3,4), l )
        l = [-4,1]
        self.assertError     (                    nps.atleast_dims, arr(2,3,4), l, -1 )
        self.assertValueShape( None, (1,2,3,4),   nps.atleast_dims, arr(2,3,4), l )
        self.assertListEqual ( l, [-4, 2])

        self.assertValueShape( None, (3,4,2),     nps.mv,           arr(2,3,4), -3, -1 )
        self.assertValueShape( None, (3,2,4),     nps.mv,           arr(2,3,4), -3,  1 )
        self.assertValueShape( None, (2,1,1,3,4), nps.mv,           arr(2,3,4), -3, -5 )
        self.assertValueShape( None, (2,1,1,3,4), nps.mv,           arr(2,3,4),  0, -5 )

        self.assertValueShape( None, (4,3,2),     nps.xchg,         arr(2,3,4), -3, -1 )
        self.assertValueShape( None, (3,2,4),     nps.xchg,         arr(2,3,4), -3,  1 )
        self.assertValueShape( None, (2,1,1,3,4), nps.xchg,         arr(2,3,4), -3, -5 )
        self.assertValueShape( None, (2,1,1,3,4), nps.xchg,         arr(2,3,4),  0, -5 )

        self.assertValueShape( None, (2,4,3),     nps.transpose,    arr(2,3,4) )
        self.assertValueShape( None, (4,3),       nps.transpose,    arr(3,4) )
        self.assertValueShape( None, (4,1),       nps.transpose,    arr(4) )

        self.assertValueShape( None, (1,2,3,4),   nps.dummy,        arr(2,3,4),  0 )
        self.assertValueShape( None, (2,1,3,4),   nps.dummy,        arr(2,3,4),  1 )
        self.assertValueShape( None, (2,3,4,1),   nps.dummy,        arr(2,3,4), -1 )
        self.assertValueShape( None, (2,3,1,4),   nps.dummy,        arr(2,3,4), -2 )
        self.assertValueShape( None, (2,1,3,4),   nps.dummy,        arr(2,3,4), -3 )
        self.assertValueShape( None, (1,2,3,4),   nps.dummy,        arr(2,3,4), -4 )
        self.assertValueShape( None, (1,1,2,3,4), nps.dummy,        arr(2,3,4), -5 )
        self.assertValueShape( None, (2,3,1,4),   nps.dummy,        arr(2,3,4),  2 )
        self.assertError     (                    nps.dummy,        arr(2,3,4),  3 )

        self.assertValueShape( None, (2,4,3),     nps.reorder,      arr(2,3,4),  0, -1, 1 )
        self.assertValueShape( None, (3,4,2),     nps.reorder,      arr(2,3,4),  -2, -1, 0 )
        self.assertValueShape( None, (1,3,1,4,2), nps.reorder,      arr(2,3,4),  -4, -2, -5, -1, 0 )
        self.assertError     (                    nps.reorder,      arr(2,3,4),  -4, -2, -5, -1, 0, 5 )

    def test_inner(self):
        r'''Testing the broadcasted inner product'''

        ref = np.array([[[  30,  255,  730],
                         [ 180,  780, 1630]],

                        [[ 180,  780, 1630],
                         [1455, 2430, 3655]],

                        [[ 330, 1305, 2530],
                         [2730, 4080, 5680]],

                        [[ 480, 1830, 3430],
                         [4005, 5730, 7705.0]]])
        self.assertValueShape( ref, (4,2,3), nps.inner, arr(2,3,5), arr(4,1,3,5) )

    def test_outer(self):
        r'''Testing the broadcasted outer product'''
        self.assertValueShape( None, (4,2,3,5,5), nps.outer, arr(2,3,5), arr(4,1,3,5) )

    def test_matmult(self):
        r'''Testing the broadcasted matrix multiplication'''
        self.assertValueShape( None, (4,2,3,5), nps.matmult, arr(2,3,7), arr(4,1,7,5) )

        ref = np.array([[[[  42,   48,   54],
                          [ 114,  136,  158]],

                         [[ 114,  120,  126],
                          [ 378,  400,  422]]],


                        [[[ 186,  224,  262],
                          [ 258,  312,  366]],

                         [[ 642,  680,  718],
                          [ 906,  960, 1014.0]]]])
        self.assertValueShape( ref, (2,2,2,3), nps.matmult, arr(2,1,2,4), arr(2,4,3) )

if __name__ == '__main__':
    unittest.main(verbosity=2)
