#!/usr/bin/python3

r'''Test the broadcasting in C

Uses the "testlib" guinea pig C library
'''

import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path[:0] = dir_path + '/..',

import numpy as np
import numpysane as nps

# Local test harness. The python standard ones all suck
from testutils import *

# The extension module we're testing
import testlib


def check(matching_functions, A, B):
    r'''Compare results of pairs of matching functions

    matching_functions is a list of pairs of functions that are supposed to
    produce identical results (testlib and numpysane implementations of
    inner and outer products). A and B are lists of arguments that we try out.
    These support broadcasting, so either one is allowed to be a single array,
    which is then used for all the checks. I check both dynamically-created and
    inlined "out" arrays

    '''
    N = 1
    if type(A) is tuple and len(A) > N:
        N = len(A)
    if type(B) is tuple and len(B) > N:
        N = len(B)

    if type(A) is not tuple: A = (A,) * N
    if type(B) is not tuple: B = (B,) * N

    for what,f0,f1 in matching_functions:
        for i in range(N):
            out0 = f0(A[i], B[i])
            out1 = f1(A[i], B[i])

            confirm_equal( out0, out1,
                           msg = what + ' matches. Dynamically-allocated output' )

            outshape = out1.shape
            out0 = np.zeros(outshape, dtype=np.array(A[i]).dtype)
            out1 = np.ones (outshape, dtype=np.array(A[i]).dtype)
            f0(A[i], B[i], out=out0)
            f1(A[i], B[i], out=out1)
            confirm_equal( out0, out1,
                           msg = what + ' matches. Pre-allocated output' )


# pairs of functions that should produce identical results
matching_functions = ( ("inner", testlib.inner, nps.inner),
                       ("outer", testlib.outer, nps.outer) )

# Basic 1D arrays
a0 = np.arange(5, dtype=float)
b  = a0+3

# a needs to broadcast; contiguous and strided
a1 = np.arange(10, dtype=float).reshape(2,5)
a2 = nps.transpose(np.arange(10, dtype=float).reshape(5,2))

# Try it!
check(matching_functions, (a0,a1,a2), b)

# Try it again, but use the floating-point version
check( (("inner", nps.inner, testlib.inner),),
      tuple([a.astype(int) for a in (a0,a1,a2)]),
      b.astype(int))

confirm_raises( lambda: check( (("inner", nps.inner, testlib.inner),),
                               (a0,a1,a2),
                               b.astype(int)),
                msg = "types must match" )

# Too few input dimensions (passing a scalar where a vector is expected). This
# should be ok. It can be viewed as a length-1 vector
check( (("inner", nps.inner, testlib.inner),),

       6.,

       (5.,
        np.array(5, dtype=float),
        np.array((5,), dtype=float),
        ),)

# Too few output dimensions. No. This is accepted only for inputs
out = np.zeros((), dtype=float)
confirm_raises(lambda: testlib.inner( nps.atleast_dims(np.array(6.,dtype=float), -5),
                                      nps.atleast_dims(np.array(5.,dtype=float), -2),
                                      out=out))

# Broadcasting. Should be ok. No barf.
confirm_does_not_raise(lambda: testlib.inner(np.arange(10, dtype=float).reshape(  2,5),
                                                np.arange(15, dtype=float).reshape(3,1,5)),
                       msg='Aligned dimensions')

confirm_raises( lambda: testlib.inner(np.arange(10, dtype=float).reshape(2,5),
                                         np.arange(15, dtype=float).reshape(3,5)) )
confirm_raises( lambda: testlib.inner(np.arange(5), np.arange(6)) )

confirm_does_not_raise( lambda: testlib.outer(a0,b, out=np.zeros((5,5), dtype=float)),
                        msg = "Basic in-place broadcasting")
confirm_raises(lambda: testlib.outer(a0,b, out=np.zeros((5,5), dtype=int)),
               msg = "Output type must match")
confirm_raises(lambda: testlib.outer(a0.astype(int),b.astype(int), out=np.zeros((5,5), dtype=float)),
               msg = "Output type must match")
confirm_does_not_raise( lambda: testlib.outer(a0.astype(float),b.astype(float), out=np.zeros((5,5), dtype=float)),
                        msg = "Output type must match")
confirm_does_not_raise( lambda: testlib.inner(a0.astype(int),b.astype(int), out=np.zeros((), dtype=int)),
                        msg = "Output type must match")
confirm_raises( lambda: testlib.outer(a0,b, out=np.zeros((3,3), dtype=float)),
                msg = "Wrong dimensions on out" )
confirm_raises( lambda: testlib.outer(a0,b, out=np.zeros((4,5), dtype=float)),
                msg = "Wrong dimensions on out" )
confirm_raises( lambda: testlib.outer(a0,b, out=np.zeros((5,), dtype=float)),
                msg = "Wrong dimensions on out" )
confirm_raises( lambda: testlib.outer(a0,b, out=np.zeros((), dtype=float)),
                msg = "Wrong dimensions on out" )
confirm_raises( lambda: testlib.outer(a0,b, out=np.zeros((5,5,5), dtype=float)),
                msg = "Wrong dimensions on out" )





from functools import reduce
def arr(*shape, **kwargs):

    dtype = kwargs.get('dtype',float)

    r'''Return an arange() array of the given shape.'''
    if len(shape) == 0:
        return np.array(3, dtype=dtype)
    product = reduce( lambda x,y: x*y, shape)
    return np.arange(product, dtype=dtype).reshape(*shape)


def test_identity3():
    r'''Testing identity3()'''

    ref = np.array(((1,0,0),
                    (0,1,0),
                    (0,0,1)),dtype=float)
    ref_int = np.array(((1,0,0),
                        (0,1,0),
                        (0,0,1)),dtype=int)

    out     = ref*0
    out_int = ref_int*0

    assertResult_inoutplace( ref,
                             testlib.identity3 )
    confirm_does_not_raise( lambda: testlib.identity3(out = out))
    confirm_raises(         lambda: testlib.identity3(out = out_int))

    out_discontiguous = np.zeros((4,5,6), dtype=float)[:3,:3,0]
    confirm(not out_discontiguous.flags['C_CONTIGUOUS'])
    testlib.identity3(out = out_discontiguous)
    confirm_equal(ref, out_discontiguous)


def test_identity():
    r'''Testing identity()

    This tests much of the named-dimensions-in-output-only logic

    '''

    # This i
    ref     = np.eye(2, dtype=float)
    ref_int = np.eye(2, dtype=int)

    out     = np.zeros((2,2), dtype=float)
    out_int = np.zeros((2,2), dtype=int)

    out32   = np.zeros((3,2), dtype=float)
    out23   = np.zeros((3,2), dtype=float)

    confirm_raises(lambda: testlib.identity(),
                   msg='output-only named dimensions MUST be given in the in-place array')
    confirm_raises(lambda: testlib.identity(out=out_int),
                   msg='types must match')
    confirm_equal(ref, testlib.identity(out=out),
                  msg='basic output-only named dimensions work')
    confirm_raises(lambda: testlib.identity(out=out23),
                   msg='output-only named dimensions must still be self-consistent')
    confirm_raises(lambda: testlib.identity(out=out32),
                   msg='output-only named dimensions must still be self-consistent')

    out_discontiguous = np.zeros((4,5,6), dtype=float)[:3,:3,0]
    confirm(not out_discontiguous.flags['C_CONTIGUOUS'])
    testlib.identity(out = out_discontiguous)
    confirm_equal(np.eye(3, dtype=float),
                  out_discontiguous)


def test_inner():
    r'''Testing the broadcasted inner product'''

    ref = np.array([[[  30,  255,  730],
                     [ 180,  780, 1630]],
                    [[ 180,  780, 1630],
                     [1455, 2430, 3655]],
                    [[ 330, 1305, 2530],
                     [2730, 4080, 5680]],
                    [[ 480, 1830, 3430],
                     [4005, 5730, 7705.0]]])
    assertResult_inoutplace(  ref,
                              testlib.inner, arr(2,3,5), arr(4,1,3,5) )

    output = np.empty((4,2,3), dtype=int)
    confirm_raises( lambda: testlib.inner( arr(  2,3,5, dtype=float),
                                              arr(4,1,3,5, dtype=float),
                                              out=output ),
                    "inner(out=out, dtype=dtype) have out=dtype==dtype" )

    # make sure non-contiguous output works properly
    output = np.empty((4,2,3), dtype=float)
    confirm(output.flags['C_CONTIGUOUS'])
    output = nps.reorder( np.empty((2,3,4), dtype=float),
                          2,0,1 )
    confirm(not output.flags['C_CONTIGUOUS'])
    confirm_equal( testlib.inner( arr(  2,3,5, dtype=float),
                                     arr(4,1,3,5, dtype=float),
                                     out=output ),
                   ref,
                   msg = 'Noncontiguous output' )
    confirm(not output.flags['C_CONTIGUOUS'])
    confirm_equal( output,
                   ref,
                   msg = 'Noncontiguous output' )


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
                             testlib.outer, arr(2,3,5, dtype=float), arr(4,1,3,5, dtype=float) )

    # make sure non-contiguous output (in both the broadcasting AND within each
    # slice) works properly
    output = np.empty((4,2,3,5,5), dtype=float)
    confirm(output.flags['C_CONTIGUOUS'])
    output = nps.reorder( np.empty((2,3,4,5,5), dtype=float),
                          2,0,1, 4,3)
    confirm(not output.flags['C_CONTIGUOUS'])
    confirm_equal( testlib.outer( arr(  2,3,5, dtype=float),
                                     arr(4,1,3,5, dtype=float),
                                     out=output ),
                   ref,
                   msg = 'Noncontiguous output (broadcasting and within each slice)' )
    confirm(not output.flags['C_CONTIGUOUS'])
    confirm_equal( output,
                   ref,
                   msg = 'Noncontiguous output (broadcasting and within each slice)' )


def test_innerouter():
    r'''Testing the broadcasted inner product'''

    ref_inner = np.array([[[  30,  255,  730],
                           [ 180,  780, 1630]],
                          [[ 180,  780, 1630],
                           [1455, 2430, 3655]],
                          [[ 330, 1305, 2530],
                           [2730, 4080, 5680]],
                          [[ 480, 1830, 3430],
                           [4005, 5730, 7705.0]]])

    # comes from PDL. numpy has a reversed axis ordering convention from
    # PDL, so I transpose the array before comparing
    ref_outer = nps.transpose(
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

    # not in-place
    try:
        i,o = testlib.innerouter(arr(2,3,5, dtype=float), arr(4,1,3,5, dtype=float))
    except:
        confirm(False, msg="broadcasted innerouter succeeded")
    else:
        confirm_equal(i.shape, ref_inner.shape, msg="broadcasted innerouter produced correct inner.shape")
        confirm_equal(i,       ref_inner,       msg="broadcasted innerouter produced correct inner")
        confirm_equal(o.shape, ref_outer.shape, msg="broadcasted innerouter produced correct outer.shape")
        confirm_equal(o,       ref_outer,       msg="broadcasted innerouter produced correct outer")

    # in-place
    try:
        i = np.empty(ref_inner.shape, dtype=float)
        o = np.empty(ref_outer.shape, dtype=float)
        testlib.innerouter(arr(2,3,5, dtype=float), arr(4,1,3,5, dtype=float), out=(i,o))
    except:
        confirm(False, msg="broadcasted in-place innerouter succeeded")
    else:
        confirm(True, msg="broadcasted in-place innerouter succeeded")
        confirm_equal(i.shape, ref_inner.shape, msg="broadcasted in-place innerouter produced correct inner.shape")
        confirm_equal(i,       ref_inner,       msg="broadcasted in-place innerouter produced correct inner")
        confirm_equal(o.shape, ref_outer.shape, msg="broadcasted in-place innerouter produced correct outer.shape")
        confirm_equal(o,       ref_outer,       msg="broadcasted in-place innerouter produced correct outer")

    # in-place with float scaling
    try:
        i = np.empty(ref_inner.shape, dtype=float)
        o = np.empty(ref_outer.shape, dtype=float)
        testlib.innerouter(arr(2,3,5, dtype=float), arr(4,1,3,5, dtype=float), out=(i,o), scale=3.5)
    except:
        confirm(False, msg="broadcasted in-place innerouter succeeded")
    else:
        confirm(True, msg="broadcasted in-place innerouter succeeded")
        confirm_equal(i.shape, ref_inner.shape, msg="broadcasted in-place innerouter with scaling produced correct inner.shape")
        confirm_equal(i,       ref_inner * 3.5, msg="broadcasted in-place innerouter with scaling produced correct inner")
        confirm_equal(o.shape, ref_outer.shape, msg="broadcasted in-place innerouter with scaling produced correct outer.shape")
        confirm_equal(o,       ref_outer * 3.5, msg="broadcasted in-place innerouter with scaling produced correct outer")

    # in-place with float scaling and string scaling
    try:
        i = np.empty(ref_inner.shape, dtype=float)
        o = np.empty(ref_outer.shape, dtype=float)
        testlib.innerouter(arr(2,3,5, dtype=float), arr(4,1,3,5, dtype=float), out=(i,o), scale=3.5, scale_string="10.0")
    except:
        confirm(False, msg="broadcasted in-place innerouter succeeded")
    else:
        confirm(True, msg="broadcasted in-place innerouter succeeded")
        confirm_equal(i.shape, ref_inner.shape, msg="broadcasted in-place innerouter with float and string scaling produced correct inner.shape")
        confirm_equal(i,       ref_inner * 35., msg="broadcasted in-place innerouter with float and string scaling produced correct inner")
        confirm_equal(o.shape, ref_outer.shape, msg="broadcasted in-place innerouter with float and string scaling produced correct outer.shape")
        confirm_equal(o,       ref_outer * 35., msg="broadcasted in-place innerouter with float and string scaling produced correct outer")

    # in-place, with some extra dummy dimensions in the output. Not allowed
    i = np.empty((1,) + ref_inner.shape, dtype=float)
    o = np.empty(ref_outer.shape, dtype=float)
    confirm_raises( lambda: testlib.innerouter(arr(2,3,5, dtype=float), arr(4,1,3,5, dtype=float), out=(i,o)),
                    msg="Extra broadcasted dimensions in the output not allowed")

    # in-place, with some extra dummy dimensions in the output. Not allowed
    i = np.empty(ref_inner.shape, dtype=float)
    o = np.empty((1,) + ref_outer.shape, dtype=float)
    confirm_raises( lambda: testlib.innerouter(arr(2,3,5, dtype=float), arr(4,1,3,5, dtype=float), out=(i,o)),
                    msg="Extra broadcasted dimensions in the output not allowed")

    # now some bogus shapes and types that should fail
    i = np.empty(ref_inner.shape, dtype=float)
    o = np.empty(ref_outer.shape, dtype=float)
    confirm_does_not_raise( lambda: testlib.innerouter(arr(2,3,5, dtype=float), arr(4,1,3,5, dtype=float), out=(i,o)),
                            msg = "basic broadcasted innerouter works")
    confirm_raises(lambda: testlib.innerouter(arr(2,3,5, dtype=float), arr(4,1,3,5, dtype=float), out=(i,i)),
                   msg = "in-place broadcasting output dimensions match")
    confirm_raises(lambda: testlib.innerouter(arr(2,3,5, dtype=float), arr(4,1,3,5, dtype=float), out=(o,o)),
                   msg = "in-place broadcasting output dimensions match")
    confirm_raises(lambda: testlib.innerouter(arr(2,3,5, dtype=float), arr(4,1,3,5, dtype=float), out=(o,i)),
                   msg = "in-place broadcasting output dimensions match")
    confirm_raises(lambda: testlib.innerouter(arr(2,3,5, dtype=float), arr(4,1,3,5, dtype=float), out=(i,o,i)),
                   msg = "in-place broadcasting output dimensions match")
    confirm_raises(lambda: testlib.innerouter(arr(2,3,5, dtype=float), arr(4,1,3,5, dtype=float), out=(i,i,o)),
                   msg = "in-place broadcasting output dimensions match")
    confirm_raises(lambda: testlib.innerouter(arr(2,3,5, dtype=float), arr(4,1,3,5, dtype=int), out=(i,o)),
                   msg = "in-place broadcasting output dimensions match")
    confirm_raises(lambda: testlib.innerouter(arr(2,3,5, dtype=float), arr(4,1,3,5, dtype=float), out=o),
                   msg = "in-place broadcasting output dimensions match")
    confirm_raises(lambda: testlib.innerouter(arr(2,3,5, dtype=float), arr(4,1,3,5, dtype=float), out=i),
                   msg = "in-place broadcasting output dimensions match")
    confirm_raises(lambda: testlib.innerouter(arr(2,3,5, dtype=float), arr(4,1,3,5, dtype=float), out=(i,None)),
                   msg = "in-place broadcasting output dimensions match")
    confirm_raises(lambda: testlib.innerouter(arr(2,3,5, dtype=float), arr(4,1,3,5, dtype=float), out=(None,i)),
                   msg = "in-place broadcasting output dimensions match")
    iint  = np.empty(ref_inner.shape, dtype=int)
    i1    = np.empty((1,) + ref_inner.shape, dtype=float)
    i2    = np.empty((2,) + ref_inner.shape, dtype=float)
    confirm_raises(lambda: testlib.innerouter(arr(2,3,5, dtype=float), arr(4,1,3,5, dtype=float), out=(iint,o)),
                   msg = "in-place broadcasting output types match")
    confirm_raises( lambda: testlib.innerouter(arr(2,3,5, dtype=float), arr(4,1,3,5, dtype=float), out=(i1,o)),
                    msg = "broadcasted innerouter: extra output dims are forbidden")
    confirm_raises(lambda: testlib.innerouter(arr(  2,3,5, dtype=float),
                                              arr(4,1,3,5, dtype=float),
                                              out=(i2,o)),
                   msg = "in-place broadcasting output dimensions match")

    confirm_does_not_raise(lambda: testlib.innerouter(arr(2,3,5, dtype=float), arr(4,1,3,5, dtype=float), scale=3.5),
                           msg = 'Validation looks at the cookie')
    confirm_raises(lambda: testlib.innerouter(arr(2,3,5, dtype=float), arr(4,1,3,5, dtype=float), scale=-3.5),
                   msg = 'Validation looks at the cookie')


def test_sorted_indices():
    x64  = np.array((1., 5., 3, 2.5, 3.5, 2.9), dtype=float)
    x32  = np.array((1., 5., 3, 2.5, 3.5, 2.9), dtype=np.float32)
    iref = np.array((0, 3, 5, 2, 4, 1), dtype=int)

    confirm_raises(lambda: testlib.sorted_indices(np.arange(5, dtype=int)))
    confirm_does_not_raise(lambda: testlib.sorted_indices(np.arange(5, dtype=np.float32)))
    confirm_does_not_raise(lambda: testlib.sorted_indices(np.arange(5, dtype=np.float32),
                                                             out=np.arange(5, dtype=np.int32)))
    confirm_raises(lambda: testlib.sorted_indices(np.arange(5, dtype=np.float32),
                                                     out=np.arange(5, dtype=int)))
    confirm_raises(lambda: testlib.sorted_indices(np.arange(5, dtype=np.float32),
                                                     out=np.arange(5, dtype=float)))
    assertResult_inoutplace( iref,
                             testlib.sorted_indices, x64, out_inplace_dtype=np.int32)
    assertResult_inoutplace( iref,
                             testlib.sorted_indices, x32, out_inplace_dtype=np.int32)


def test_broadcasting():

    assertValueShape( np.array(5),                (),     testlib.inner, arr(3),     arr(3))
    assertValueShape( np.array((5,14)),           (2,),   testlib.inner, arr(2,3),   arr(3))
    assertValueShape( np.array((5,14)),           (2,),   testlib.inner, arr(3),     arr(2,3))
    assertValueShape( np.array(((5,14),)),        (1,2,), testlib.inner, arr(1,2,3), arr(3))
    assertValueShape( np.array(((5,),(14,))),     (2,1,), testlib.inner, arr(2,1,3), arr(3))
    assertValueShape( np.array((5,14)),           (2,),   testlib.inner, arr(2,3),   arr(1,3))
    assertValueShape( np.array((5,14)),           (2,),   testlib.inner, arr(1,3),   arr(2,3))
    assertValueShape( np.array(((5,14),)),        (1,2,), testlib.inner, arr(1,2,3), arr(1,3))
    assertValueShape( np.array(((5,),(14,))),     (2,1,), testlib.inner, arr(2,1,3), arr(1,3))
    assertValueShape( np.array(((5,14),(14,50))), (2,2,), testlib.inner, arr(2,1,3), arr(2,3))
    assertValueShape( np.array(((5,14),(14,50))), (2,2,), testlib.inner, arr(2,1,3), arr(1,2,3))

    confirm_raises( lambda: testlib.inner(arr(3)), msg='right number of args' )

    confirm_raises( lambda: testlib.inner(arr(3),arr(5)),         msg='matching args')
    confirm_raises( lambda: testlib.inner(arr(2,3),arr(4,3)),     msg='matching args')
    confirm_raises( lambda: testlib.inner(arr(3,3,3),arr(2,1,3)), msg='matching args')
    confirm_raises( lambda: testlib.inner(arr(1,2,4),arr(2,1,3)), msg='matching args')

    # make sure the output COUNTS are checked (if I expect 2 outputs, but get
    # only 1, that's an error
    confirm( testlib.innerouter(arr(5), arr(  5)) is not None, msg='output count check' )
    confirm( testlib.innerouter(arr(5), arr(2,5)) is not None, msg='output count check' )

    confirm( testlib.innerouter(arr(5), arr(  5)) is not None,
             msg='output dimensionality check with given out' )


    # Basic out_kwarg tests. More thorough ones later, in
    # test_broadcasting_into_output())
    a5   = arr(   5,          dtype=float)
    a25  = arr(2, 5,          dtype=float)
    a125 = arr(1, 2, 5,       dtype=float)
    o    = np.zeros((),       dtype=float)
    o2   = np.zeros((2,),     dtype=float)
    o5   = np.zeros((5,),     dtype=float)
    o12  = np.zeros((1,2),    dtype=float)
    o22  = np.zeros((2,2),    dtype=float)
    o55  = np.zeros((5,5),    dtype=float)
    o25  = np.zeros((2,5),    dtype=float)
    o255 = np.zeros((2,5,5),  dtype=float)
    o1255= np.zeros((1,2,5,5),dtype=float)

    # no broadcasting
    confirm_raises( lambda: \
                    testlib.innerouter(a5, a5, out=o), \
                    msg='output dimensionality check with given out' )
    confirm_raises( lambda: \
                    testlib.innerouter(a5, a5, out=o2), \
                    msg='output dimensionality check with given out' )
    confirm_raises( lambda: \
                    testlib.innerouter(a5, a5, out=(o,)), \
                    msg='output dimensionality check with given out' )
    confirm_raises( lambda: \
                    testlib.innerouter(a5, a5, out=(o55,)), \
                    msg='output dimensionality check with given out' )
    confirm_raises( lambda: \
                    testlib.innerouter(a5, a5, out=(o55,o)), \
                    msg='output dimensionality check with given out' )
    confirm_raises( lambda: \
                    testlib.innerouter(a5, a5, out=(o,o2)), \
                    msg='output dimensionality check with given out' )
    confirm_raises( lambda: \
                    testlib.innerouter(a5, a5, out=(o,o5)), \
                    msg='output dimensionality check with given out' )
    confirm_raises( lambda: \
                    testlib.innerouter(a5, a5, out=(o2,o55)), \
                    msg='output dimensionality check with given out' )
    confirm_raises( lambda: \
                    testlib.innerouter(a5, a5, out=(o,o55,o)), \
                    msg='output dimensionality check with given out' )
    confirm( testlib.innerouter(a5, a5, out=(o,o55)) is not None,
             msg='output dimensionality check with given out' )
    confirm_equal(o,   a5.dot(a5),      msg='in-place broadcasting computed the right value')
    confirm_equal(o55, np.outer(a5,a5), msg='in-place broadcasting computed the right value')

    # two broadcasted slices
    confirm_raises( lambda: \
                    testlib.innerouter(a5, a25, out=o),
                    msg='output dimensionality check with given out' )
    confirm_raises( lambda: \
                    testlib.innerouter(a5, a25, out=o2),
                    msg='output dimensionality check with given out' )
    confirm_raises( lambda: \
                    testlib.innerouter(a5, a25, out=(o,)),
                    msg='output dimensionality check with given out' )
    confirm_raises( lambda: \
                    testlib.innerouter(a5, a25, out=(o55,)),
                    msg='output dimensionality check with given out' )
    confirm_raises( lambda: \
                    testlib.innerouter(a5, a25, out=(o55,o)),
                    msg='output dimensionality check with given out' )
    confirm_raises( lambda: \
                    testlib.innerouter(a5, a25, out=(o,o2)),
                    msg='output dimensionality check with given out' )
    confirm_raises( lambda: \
                    testlib.innerouter(a5, a25, out=(o,o5)),
                    msg='output dimensionality check with given out' )
    confirm_raises( lambda: \
                    testlib.innerouter(a5, a25, out=(o2,o55)),
                    msg='output dimensionality check with given out' )
    confirm_raises( lambda: \
                    testlib.innerouter(a5, a25, out=(o,o55,o)),
                    msg='output dimensionality check with given out' )
    confirm_raises( lambda: \
                    testlib.innerouter(a5, a25, out=(o,o55)),
                    msg='output dimensionality check with given out' )
    confirm_raises( lambda: \
                    testlib.innerouter(a5, a25, out=(o,o255)),
                    msg='output dimensionality check with given out' )
    confirm( testlib.innerouter(a5, a25, out=(o2,o255)) is not None,
             msg='output dimensionality check with given out' )
    confirm_equal(o2,   nps.inner(a5,a25), msg='in-place broadcasting computed the right value')
    confirm_equal(o255, nps.outer(a5,a25), msg='in-place broadcasting computed the right value')

    # Non-contiguous data should work with inner and outer, but not innerouter
    # (that's what the underlying C library does/does not support)
    a2                = arr(2, dtype=float)
    a25_noncontiguous = arr(5, 2, dtype=float).T
    o255_noncontiguous = nps.transpose(np.zeros((2,5,5), dtype=float))
    o255_noncontiguous_in_broadcast = np.zeros((2,2,5,5), dtype=float)[:,0,:,:]
    confirm_does_not_raise(lambda: testlib.inner     (a25_noncontiguous, a5),
                           msg='Validation: noncontiguous in the function slice')
    confirm_does_not_raise(lambda: testlib.outer     (a25_noncontiguous, a5),
                           msg='Validation: noncontiguous in the function slice')
    confirm_does_not_raise(lambda: testlib.outer     (a25_noncontiguous, a5, out=o255_noncontiguous),
                           msg='Validation: noncontiguous in the function slice')
    confirm_raises        (lambda: testlib.innerouter(a25_noncontiguous, a5),
                           msg='Validation: noncontiguous in the function slice')
    confirm_does_not_raise(lambda: testlib.innerouter(a25, a5, out=(a2, o255)),
                           msg='Validation: noncontiguous in the function slice')
    confirm_raises        (lambda: testlib.innerouter(a25, a5, out=(a2, o255_noncontiguous)),
                           msg='Validation: noncontiguous in the function slice')

    confirm_does_not_raise(lambda: testlib.innerouter(a25, a5, out=(a2, o255_noncontiguous_in_broadcast)),
                           msg='Validation: noncontiguous array that are noncontiguous ONLY in the broadcasted dimensions (i.e. each slice IS contiguous)')

    # Extra slices in the output not allowed
    confirm_does_not_raise( lambda: \
                            testlib.innerouter(a5, a25, out=(o2,o255)),
                            msg='output dimensionality check with given out' )
    confirm_raises( lambda: \
                    testlib.innerouter(a5, a125, out=(o2,o255)),
                    msg='output dimensionality check with given out' )
    confirm_does_not_raise( lambda: \
                            testlib.innerouter(a5, a125, out=(o12,o1255)),
                            msg='output dimensionality check with given out' )
    confirm_raises( lambda: \
                    testlib.innerouter(a5, a125, out=(o12,o2255)),
                    msg='output dimensionality check with given out' )


test_identity3()
test_identity()
test_inner()
test_outer()
test_innerouter()
test_broadcasting()
test_sorted_indices()

finish()
