#!/usr/bin/python2

import unittest

import numpy as np
import numpysane as nps


def arr(*shape):
    """Return an arange() array of the given shape."""
    product = reduce( lambda x,y: x*y, shape)
    return np.arange(product).reshape(*shape)

class TestNumpysane(unittest.TestCase):

    def assertListEqual(self, s1, s2):
        """This is unittest.TestCase.assertListEqual(), but not retarded.

That function barfs when fed (), and this one does not.

        """
        self.assertEqual(len(s1), len(s2), msg="Lists {} and {} do not match".format(s1,s2))
        for v1,v2 in zip(s1,s2):
            self.assertEqual(v1,v2, msg="Lists {} and {} do not match".format(s1,s2))

    def assertNumpyAlmostEqual(self, first, second):
        self.assertListEqual(first.shape, second.shape)
        diff = first - second
        diff = diff.ravel()
        rms = np.sqrt(diff.dot(diff) / diff.size)
        self.assertLess( rms, 1e-6, msg='matrix discrepancy:\n{} vs\n{}. Diff:\n{}'.format(first,second,diff) )

    def assertError(self, f, *args, **kwargs):
        """Convenience wrapper for my use of assertRaises()"""
        return self.assertRaises(nps.NumpysaneError, f, *args, **kwargs)

    def assertValueShape(self, value, s, f, *args, **kwargs):
        """Makes sure a given call produces a given value and shape.

It is redundant to specify both, but it makes it clear I'm asking for what I
think I'm asking. The value check can be skipped by passing None.

        """
        res = f(*args, **kwargs)
        self.assertListEqual(res.shape, s)
        if value is not None:
            self.assertNumpyAlmostEqual(res, value)

    def test_broadcasting(self):
        """Checking broadcasting rules."""
        @nps.broadcast_define( ('n',), ('n') )
        def f1(a, b):
            """Basic inner product."""
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
        self.assertError( f1, arr(3),arr(3),arr(3) )

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
                               arr(  m));
        self.assertValueShape( np.array((d,)), (1,m),
                               f2,
                               arr(1,    3),
                               arr(1,  n,3),
                               arr(      n),
                               arr(1,    m));
        self.assertValueShape( np.array((d,)), (1,m,),
                               f2,
                               arr(1,    3),
                               arr(1,  n,3),
                               arr(      n),
                               arr(      m));
        self.assertValueShape( np.array((d,d+m,d+2*m,d+3*m,d+4*m)), (5,m),
                               f2,
                               arr(5,    3),
                               arr(5,  n,3),
                               arr(      n),
                               arr(5,    m));
        self.assertValueShape( np.array(((d,d+m,d+2*m,d+3*m,d+4*m),)), (1,5,m),
                               f2,
                               arr(1,5,    3),
                               arr(  5,  n,3),
                               arr(        n),
                               arr(  5,    m));
        self.assertValueShape( np.array(((d,d+m,d+2*m,d+3*m,d+4*m), (d,d+m,d+2*m,d+3*m,d+4*m))), (2,5,m),
                               f2,
                               arr(1,5,    3),
                               arr(2,5,  n,3),
                               arr(        n),
                               arr(  5,    m));
        self.assertValueShape( np.array(((d,d+m,d+2*m,d+3*m,d+4*m), (d,d+m,d+2*m,d+3*m,d+4*m))), (2,5,m),
                               f2,
                               arr(1,5,    3),
                               arr(2,1,  n,3),
                               arr(        n),
                               arr(  5,    m));
        self.assertValueShape( np.array((((d,d,d,d,d), (d,d,d,d,d)),)), (1,2,5,m),
                               f2,
                               arr(1,1,5,    3),
                               arr(1,2,1,  n,3),
                               arr(1,        n),
                               arr(1,  1,    m));

        # mismatched args
        self.assertError( f2,
                          arr(5,    3),
                          arr(5,  n,3),
                          arr(      m),
                          arr(5,    m));
        self.assertError( f2,
                          arr(5,    2),
                          arr(5,  n,3),
                          arr(      n),
                          arr(5,    m));
        self.assertError( f2,
                          arr(5,    2),
                          arr(5,  n,2),
                          arr(      n),
                          arr(5,    m));
        self.assertError( f2,
                          arr(1,    3),
                          arr(1,  n,3),
                          arr(      5*n),
                          arr(1,    m));



    def test_concatenation(self):
        """Checking the various concatenation functions."""

        # axes must be negative
        self.assertError( nps.glue, arr(2,3), arr(2,3), axis=0 )
        self.assertError( nps.glue, arr(2,3), arr(2,3), axis=1 )

        # basic glueing
        self.assertValueShape( None, (2,6),     nps.glue, arr(2,3), arr(2,3), axis=-1 )
        self.assertValueShape( None, (4,3),     nps.glue, arr(2,3), arr(2,3), axis=-2 )
        self.assertValueShape( None, (2,2,3),   nps.glue, arr(2,3), arr(2,3), axis=-3 )
        self.assertValueShape( None, (2,1,2,3), nps.glue, arr(2,3), arr(2,3), axis=-4 )

        # self.assertListEqual( (2,6), nps.glue( arr(1,3), arr(2,3), axis=-1).shape )
        # self.assertListEqual( (4,3), nps.glue( arr(2,3), arr(2,3), axis=-2).shape )


if __name__ == '__main__':
    unittest.main(verbosity=2)
