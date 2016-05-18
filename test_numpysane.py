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
        self.assertListEqual(res.shape, s)
        if value is not None:
            self.assertNumpyAlmostEqual(res, value)

    def test_broadcasting(self):
        r'''Checking broadcasting rules.'''
        @nps.broadcast_define( ('n',), ('n',) )
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
        @nps.broadcast_define( (3,), ('n',3), ('n',), ('m',) )
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
        @nps.broadcast_define( (3,), ('n',3), ('n',), ('m',) )
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
        @nps.broadcast_define( ('n',), ('n','m'), (2,), () )
        def f4(a,b,c,d):
            return d
        @nps.broadcast_define( ('n',), ('n','m'), (2,), () )
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
