#!/usr/bin/python2

import unittest

import numpy as np
import numpysane as nps


def arr(*shape):
    """Return an arange() array of the given shape."""
    product = reduce( lambda x,y: x*y, shape)
    return np.arange(product).reshape(*shape)

class TestBroadcasting(unittest.TestCase):

    def assertListEqual(self, s1, s2):
        """This is unittest.TestCase.assertListEqual(), but not retarded. That function
barfs when fed ()."""
        self.assertEqual(len(s1), len(s2))
        for v1,v2 in zip(s1,s2):
            self.assertEqual(v1,v2)

    def assertNumpyAlmostEqual(self, first, second):
        self.assertListEqual(first.shape, second.shape)
        diff = first - second
        diff = diff.ravel()
        rms = np.sqrt(diff.dot(diff) / diff.size)
        self.assertLess( rms, 1e-6 )

    def test_broadcasting(self):
        """Checking broadcasting rules."""
        @nps.broadcast_define( ('n',), ('n') )
        def f1(a, b):
            """Basic inner product."""
            return a.dot(b)

        res = f1(arr(3), arr(3))
        self.assertListEqual       (res.shape, ())
        self.assertNumpyAlmostEqual(res, np.array(5))

        res = f1(arr(2,3), arr(3))
        self.assertListEqual       (res.shape, (2,))
        self.assertNumpyAlmostEqual(res, np.array((5,14)))

        res = f1(arr(3), arr(2,3))
        self.assertListEqual       (res.shape, (2,))
        self.assertNumpyAlmostEqual(res, np.array((5,14)))

        res = f1(arr(1,2,3), arr(3))
        self.assertListEqual       (res.shape, (1,2,))
        self.assertNumpyAlmostEqual(res, np.array(((5,14),)))

        res = f1(arr(2,1,3), arr(3))
        self.assertListEqual       (res.shape, (2,1,))
        self.assertNumpyAlmostEqual(res, np.array(((5,),(14,))))

        res = f1(arr(2,3), arr(1,3))
        self.assertListEqual       (res.shape, (2,))
        self.assertNumpyAlmostEqual(res, np.array((5,14)))

        res = f1(arr(1,3), arr(2,3))
        self.assertListEqual       (res.shape, (2,))
        self.assertNumpyAlmostEqual(res, np.array((5,14)))

        res = f1(arr(1,2,3), arr(1,3))
        self.assertListEqual       (res.shape, (1,2,))
        self.assertNumpyAlmostEqual(res, np.array(((5,14),)))

        res = f1(arr(2,1,3), arr(1,3))
        self.assertListEqual       (res.shape, (2,1,))
        self.assertNumpyAlmostEqual(res, np.array(((5,),(14,))))

        res = f1(arr(2,1,3), arr(2,3))
        self.assertListEqual       (res.shape, (2,2,))
        self.assertNumpyAlmostEqual(res, np.array(((5,14),(14,50))))

        res = f1(arr(2,1,3), arr(1,2,3))
        self.assertListEqual       (res.shape, (2,2,))
        self.assertNumpyAlmostEqual(res, np.array(((5,14),(14,50))))


        # wrong number of args
        self.assertRaises( nps.NumpySaneError, f1, arr(3) )
        self.assertRaises( nps.NumpySaneError, f1, arr(3),arr(3),arr(3) )

        # mismatched args
        self.assertRaises( nps.NumpySaneError, f1, arr(3),arr(5) )
        self.assertRaises( nps.NumpySaneError, f1, arr(2,3),arr(4,3) )
        self.assertRaises( nps.NumpySaneError, f1, arr(3,3,3),arr(2,1,3) )
        self.assertRaises( nps.NumpySaneError, f1, arr(1,2,4),arr(2,1,3) )


if __name__ == '__main__':
    unittest.main(verbosity=2)
