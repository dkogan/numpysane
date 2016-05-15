#!/usr/bin/python

r'''* SYNOPSIS

    >>> import numpy as np
    >>> import numpysane as nps

    >>> a   = np.arange(6).reshape(2,3)
    >>> b   = a + 100
    >>> row = a[0,:]

    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])

    >>> b
    array([[100, 101, 102],
           [103, 104, 105]])

    >>> row
    array([1000, 1001, 1002])

    >>> nps.glue(a,b, axis=-1)
    array([[  0,   1,   2, 100, 101, 102],
           [  3,   4,   5, 103, 104, 105]])

    >>> nps.glue(a,b,row, axis=-2)
    array([[   0,    1,    2],
           [   3,    4,    5],
           [ 100,  101,  102],
           [ 103,  104,  105],
           [1000, 1001, 1002]])

    >>> nps.cat(a,b)
    array([[[  0,   1,   2],
            [  3,   4,   5]],

           [[100, 101, 102],
            [103, 104, 105]]])

    >>> @nps.broadcast_define( ('n',), ('n',) )
    ... def inner_product(a, b):
    ...     return a.dot(b)

    >>> inner_product(a,b)
    array([ 305, 1250])

* DESCRIPTION
Numpy is widely used, relatively polished, and has a wide range of libraries
available. At the same time, some of its very core functionality is strange,
confusing and just plain wrong. This is in contrast with PDL
(http://pdl.perl.org), which has a much more reasonable core, but a number of
higher-level warts, and a relative dearth of library support. This module
intends to improve the developer experience by providing alternate APIs to some
core numpy functionality that is much more reasonable, especially for those who
have used PDL in the past.

The issues addressed by this module fall into two categories:

1. Incomplete broadcasting support
2. Strange, special-case-ridden rules for basic array manipulation, especially
   dealing with dimensionality

** Broadcasting
*** Problem
Numpy has a limited support for broadcasting
(http://docs.scipy.org/doc/numpy-1.10.1/user/basics.broadcasting.html), a
generic way to vectorize functions. When making a broadcasted call to a
function, you pass in arguments with the inputs to vectorize available in new
dimensions, and the broadcasting mechanism automatically calls the function
multiple times as needed, and reports the output as an array collecting all the
results.

A basic example is an inner product: a function that takes in two
identically-sized vectors (1-dimensional arrays), and returns a scalar
(0-dimensional array). As an example, a broadcasted inner product function could
take in two arrays of shape (2,3,4), compute the 6 inner products of length-4
each, and report the output in an array of shape (2,3). Numpy puts the
most-significant dimension at the end, which is why this isn't 12 inner products
of length-2 each.

The user doesn't choose whether to use broadcasting or not; some functions
support it, and some do not. In PDL, broadcasting (called "threading" in that
system) is a pervasive concept throughout. A PDL user has an expectation that
every function can broadcast, and the documentation for every function is very
explicit about the dimensionality of the inputs and outputs. Any data above the
expected input dimensions is broadcast.

By contrast, in numpy very few functions know how to broadcast. On top of that,
the documentation is usually silent about the broadcasting status of a function
in question. This results in a messy situation where the user is often not sure
of the exact effect of the functions they're calling.

*** Solution
This module contains functionality to make any arbitrary function broadcastable.
This is invoked as a decorator, applied to the arbitrary user function. An
example:

    >>> import numpysane as nps

    >>> @nps.broadcast_define( ('n',), ('n',) )
    ... def inner_product(a, b):
    ...     return a.dot(b)

Here we have a simple inner product function. We call 'broadcast_define' to add
a broadcasting-aware wrapper that takes two 1D vectors of length 'n' each (same
'n' for the two inputs). This new 'inner_product' function applies broadcasting,
as needed:

    >>> import numpy as np

    >>> a = np.arange(6).reshape(2,3)
    >>> b = a + 100

    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])

    >>> b
    array([[100, 101, 102],
           [103, 104, 105]])

    >>> a = np.arange(6).reshape(2,3)
    >>> b = a + 100

    >>> inner_product(a,b)
    array([ 305, 1250])

A detailed description of broadcasting rules is available in the numpy
documentation: http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

In short:

- The most significant dimension in a numpy array is the LAST one, so the
  prototype of an input argument must exactly match a given input's trailing
  shape. So a prototype shape of (a,b,c) accepts an argument shape of (......,
  a,b,c), with as many or as few leading dimensions as desired. This design
  choice is backwards from how PDL does it, where the most significant dimension
  is the first one.
- The extra leading dimensions must be compatible across all the inputs. This
  means that each leading dimension must either
  - be 1
  - be missing (thus assumed to be 1)
  - be some positive integer >1, consistent across all arguments
- The output is collected into an array that's sized as a superset of the
  above-prototype shape of each argument

More involved example: A function with input prototype ( (3,), ('n',3), ('n',),
('m',) ) given inputs of shape

    (1,5,    3)
    (2,1,  8,3)
    (        8)
    (  5,    9)

will return an output array of shape (2,5, ...), where ... is the shape of each
output slice. Note again that the prototype dictates the TRAILING shape of the
inputs.

Stock numpy as some rudimentary support for this with its vectorize() function,
but it assumes only scalar inputs and outputs, which severaly limits its
usefulness.

*** New planned functionality

In addition to this basic broadcasting support, I'm planning the following:

- Output memory should be used more efficiently. This means that the output
  array should be allocated once, and each slice output should be written into
  the correct place in the array. To make this possible, the output dimensions
  need to be a part of the prototype, and the output array should be passable to
  the function being wrapped.

- A C-level broadcast_define(). This would be the analogue of PDL::PP
  (http://pdl.perl.org/PDLdocs/PP.html). This flavor of broadcast_define() would
  be invoked by the build system to wrap C functions. It would implement
  broadcasting awareness in generated C code. This should work more effectively
  for performance-sensitive inner loops.

- Automatic parallelization for broadcasted slices. Since each broadcasting loop
  is independent, this is a very natural place to add parallelization.

- Dimensions should support a symbolic declaration. For instance, one could want
  a function to accept an input of shape (n) and another of shape (n*n). There's
  no way to declare this currently, but there should be.

** Strangeness in core routines
*** Problem
There are some core numpy functions whose behavior is very confusing (at least
to me) and full of special cases, that make it even more difficult to know
what's going on. A prime example (but not the only one) is the array
concatenation routines. Numpy has a number of functions to do this, each being
strange and confusing. In the below examples, I use a function "arr" that
returns a numpy array of with given dimensions:

    >>> def arr(*shape): return np.zeros(shape)

    >>> arr(1,2,3).shape
    (1, 2, 3)

Existing concatenation functions:

**** hstack()
hstack() performs a "horizontal" concatenation. When numpy prints an array, this
is the last dimension (remember, the most significant dimensions in numpy are at
the end). So one would expect that this function concatenates arrays along this
last dimension. In the special case of 1D and 2D arrays, one would be right:

    >>> np.hstack( (arr(3), arr(3))).shape
    (6,)

    >>> np.hstack( (arr(2,3), arr(2,3))).shape
    (2, 6)

but in any other case, one would be wrong:

    >>> np.hstack( (arr(1,2,3), arr(1,2,3))).shape
    (1, 4, 3)     <------ I expect (1, 2, 6)

    >>> np.hstack( (arr(1,2,3), arr(1,2,4))).shape
    [exception]   <------ I expect (1, 2, 7)

    >>> np.hstack( (arr(3), arr(1,3))).shape
    [exception]   <------ I expect (1, 6)

    >>> np.hstack( (arr(1,3), arr(3))).shape
    [exception]   <------ I expect (1, 6)

I think the above should all succeed, and should produce the shapes as
indicated. Cases such as "np.hstack( (arr(3), arr(1,3)))" are maybe up for
debate, but broadcasting rules allow adding as many extra length-1 dimensions as
we want without changing the meaning of the object, so I claim this should work.
Either way, if you print out the operands for any of the above, you too would
expect a "horizontal" stack() to work as stated above.

It turns out that normally hstack() concatenates along the SECOND dimension,
unless the first argument only has one dimension, in which case the FIRST
dimension is used. This is 100% wrong in a system where the most significant
dimension is the last one, unless you assume that everyone has only 2D arrays,
where the last dimension and the second dimension are the same.

**** vstack()
Similarly, vstack() performs a "vertical" concatenation. When numpy prints an
array, this is the second-to-last dimension (remember, the most significant
dimensions in numpy are at the end). So one would expect that this function
concatenates arrays along this second-to-last dimension. Again, in the special
case of 1D and 2D arrays, one would be right:

    >>> np.vstack( (arr(2,3), arr(2,3))).shape
    (4, 3)

    >>> np.vstack( (arr(1,3), arr(3))).shape
    (2, 3)

    >>> np.vstack( (arr(3), arr(1,3))).shape
    (2, 3)

    >>> np.vstack( (arr(2,3), arr(3))).shape
    (3, 3)

Note that this function appears to tolerate some amount of shape mismatches. It
does it in a form one would expect, but given the weird state of the rest of
this system, I found it surprising. For instance "np.hstack( (arr(1,3),
arr(3)))" fails, so one would think that "np.vstack( (arr(1,3), arr(3)))" would
fail too.

And once again, adding more dimensions make it confused, for the same reason:

    >>> np.vstack( (arr(1,2,3), arr(2,3))).shape
    [exception]   <------ I expect (1, 4, 3)

    >>> np.vstack( (arr(1,2,3), arr(1,2,3))).shape
    (2, 2, 3)     <------ I expect (1, 4, 3)

Similarly to hstack(), vstack() concatenates along the first dimension, which is
"vertical" only for 2D arrays, but for no others. And similarly to hstack(), the
1D case is special-cased to work properly.

**** dstack()
I'll skip the detailed description, since this is similar to hstack() and
vstack(). The intent was to concatenate across the third-to-last dimension, but
the implementation takes dimension 2 instead. This is wrong, as before. And I
find it strange that these 3 functions even exist, since they are all
special-cases: the concatenation should be an argument, and at most the edge
special case (hstack()) should exist. This brings us to the next function:

**** concatenate()
This is a more general function, and unlike hstack(), vstack() and dstack(), it
takes as input a list of arrays AND the concatenation dimension. It accepts
negative concatenation dimensions to allow us to count from the end, so things
should work better. And in many ways that failed previously, they do:

    >>> np.concatenate( (arr(1,2,3), arr(1,2,3)), axis=-1).shape
    (1, 2, 6)

    >>> np.concatenate( (arr(1,2,3), arr(1,2,4)), axis=-1).shape
    (1, 2, 7)

    >>> np.concatenate( (arr(1,2,3), arr(1,2,3)), axis=-2).shape
    (1, 4, 3)

But many things still don't work as I would expect:

    >>> np.concatenate( (arr(1,3), arr(3)), axis=-1).shape
    [exception]   <------ I expect (1, 6)

    >>> np.concatenate( (arr(3), arr(1,3)), axis=-1).shape
    [exception]   <------ I expect (1, 6)

    >>> np.concatenate( (arr(1,3), arr(3)), axis=-2).shape
    [exception]   <------ I expect (3, 3)

    >>> np.concatenate( (arr(3), arr(1,3)), axis=-2).shape
    [exception]   <------ I expect (2, 3)

    >>> np.concatenate( (arr(2,3), arr(2,3)), axis=-3).shape
    [exception]   <------ I expect (2, 2, 3)

This function works as expected only if

- All inputs have the same number of dimensions
- All inputs have a matching shape, except for the dimension along which we're
  concatenating
- All inputs have the dimension along which we're concatenating

A legitimate use case that violates these conditions: I have an object that
contains N 3d vectors, and I want to add another 3d vector to it. This is
essentially the first example above.

**** stack()
The name makes it sound exactly like concatenate(), and it takes the same
arguments, but it is very different. stack() requires that all inputs have
EXACTLY the same shape. It then concatenates all the inputs along a new
dimension, and places that dimension in the location given by the 'axis' input.
If this is the exact type of concatenation you want, this function works fine.
But it's one of many things a user may want to do.

*** Solution
This module introduces new functions to concatenate arrays in various ways that
(I think) are more intuitive and more reasonable. They do not refer to anything
being "horizontal" or "vertical", nor do they talk about "rows" or "columns".
These concepts simply don't apply to a generic N-dimensional system.
Furthermore, these functions come directly from PDL, and have the same names and
semantics.

These functions assume that broadcasting is an important concept in the system,
so all dimensions are counted from the most significant dimension: the last
dimension in numpy. This means that only negative dimension indices are
accepted.

Example for further justification: an array containing N 3D vectors would have
shape (N,3). Another array containing a single 3D vector would have shape (3).
Counting the dimensions from the end, each vector is indexed in dimension -1.
However, counting from the front the vector is indexed in dimension 0 or 1,
depending on which of the two arrays we're looking at. If we want to add the
single vector to the array containing the N vectors, and we mistakenly try to
concatenate along the first dimension, it would fail if N != 3. But if we're
unlucky, and N=3, then we'd get a nonsensical output array of shape (3,4). Why
would an array of N 3D vectors have shape (N,3) and not (3,N)? Because if we
apply python iteration to it, we'd expect to get N iterates of arrays with shape
(3,) each, and numpy iterates from the first dimension:

    >>> a = np.arange(2*3).reshape(2,3)

    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])

    >>> [x for x in a]
    [array([0, 1, 2]), array([3, 4, 5])]

New concatenation, dimension-manipulation functions this module provides
(documented fully in the next section):

- glue: concatenate a given list of arrays along the given axis
- cat:  concatenate a given list of arrays along a new least-significant
  (leading) axis

'''

import numpy as np
from functools import reduce

# object needed for fancy slices. m[:] is exactly the same as
# m[_colon], but '_colon' can be manipulated in ways that ':' can't
_colon = slice(None, None, None)


class NumpysaneError(Exception):
    def __init__(self, err): self.err = err
    def __str__(self):       return self.err

def glue(*args, **kwargs):
    r'''Concatenates a given list of arrays along the given 'axis' keyword argument.

    Synopsis:

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(6).reshape(2,3)
        >>> b = a + 100
        >>> row = a[0,:] + 1000

        >>> a
        array([[0, 1, 2],
               [3, 4, 5]])

        >>> b
        array([[100, 101, 102],
               [103, 104, 105]])

        >>> row
        array([1000, 1001, 1002])

        >>> nps.glue(a,b, axis=-1)
        array([[  0,   1,   2, 100, 101, 102],
               [  3,   4,   5, 103, 104, 105]])

        >>> nps.glue(a,b,row, axis=-2)
        array([[   0,    1,    2],
               [   3,    4,    5],
               [ 100,  101,  102],
               [ 103,  104,  105],
               [1000, 1001, 1002]])

        >>> nps.glue(a,b, axis=-3)
        array([[[  0,   1,   2],
                [  3,   4,   5]],

               [[100, 101, 102],
                [103, 104, 105]]])

    If no such keyword argument is given, a new dimension is added at the front, and
    we concatenate along that new dimension. This case is equivalent to the cat()
    function.

    In order to count dimensions from the inner-most outwards, this function accepts
    only negative axis arguments. This is because numpy broadcasts from the last
    dimension, and the last dimension is the inner-most in the (usual) internal
    storage scheme. Allowing glue() to look at dimensions at the start would allow
    it to unalign the broadcasting dimensions, which is never what you want.

    To glue along the last dimension, pass axis=-1; to glue along the second-to-last
    dimension, pass axis=-2, and so on.

    Unlike in PDL, this function refuses to create duplicated data to make the
    shapes fit. In my experience, this isn't what you want, and can create bugs. For
    instance:

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(6).reshape(2,3)
        >>> b = a[0:1,:]


        >>> a
        array([[0, 1, 2],
               [3, 4, 5]])

        >>> b
        array([[0, 1, 2]])

        >>> nps.glue(a,b,axis=-1)
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          File "numpysane.py", line 140, in glue
            return np.concatenate( args, axis=axis )
        ValueError: all the input array dimensions except for the concatenation axis must match exactly


    Finally, this function adds as many length-1 dimensions at the front as
    required. Note that this does not create new data, just new degenerate
    dimensions. Example:

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(6).reshape(2,3)
        >>> b = a + 100

        >>> a
        array([[0, 1, 2],
               [3, 4, 5]])

        >>> b
        array([[100, 101, 102],
               [103, 104, 105]])

        >>> res = nps.glue(a,b, axis=-5)
        >>> res
        array([[[[[  0,   1,   2],
                  [  3,   4,   5]]]],



               [[[[100, 101, 102],
                  [103, 104, 105]]]]])

        >>> res.shape
        (2, 1, 1, 2, 3)

    '''

    axis = kwargs.get('axis')
    if axis is not None and axis >= 0:
        raise NumpysaneError("axis >= 0 can make broadcasting dimensions inconsistent, and is thus not allowed")

    # deal with scalar (non-ndarray) args
    args = [ np.asarray(x) for x in args ]

    # If no axis is given, add a new axis at the front, and glue along it
    max_ndim = max( x.ndim for x in args )
    if axis is None:
        axis = -1 - max_ndim

    # if we're glueing along a dimension beyond what we already have, expand the
    # target dimension count
    if max_ndim < -axis:
        max_ndim = -axis

    # Now I add dummy dimensions at the front of each array, to bring the source
    # arrays to the same dimensionality. After this is done, ndims for all the
    # matrices will be the same, and np.concatenate() should know what to do.
    args = [ x[(np.newaxis,)*(max_ndim - x.ndim) + (Ellipsis,)] for x in args ]

    return np.concatenate( args, axis=axis )


def cat(*args):
    r'''Concatenates a given list of arrays along a new first (outer) dimension.

    Synopsis:

        >>> import numpy as np
        >>> import numpysane as nps

        >>> a = np.arange(6).reshape(2,3)
        >>> b = a + 100
        >>> c = a - 100

        >>> a
        array([[0, 1, 2],
               [3, 4, 5]])

        >>> b
        array([[100, 101, 102],
               [103, 104, 105]])

        >>> c
        array([[-100,  -99,  -98],
               [ -97,  -96,  -95]])

        >>> res = nps.cat(a,b,c)
        >>> res
        array([[[   0,    1,    2],
                [   3,    4,    5]],

               [[ 100,  101,  102],
                [ 103,  104,  105]],

               [[-100,  -99,  -98],
                [ -97,  -96,  -95]]])

        >>> res.shape
        (3, 2, 3)

        >>> [x for x in res]
        [array([[0, 1, 2],
               [3, 4, 5]]),
         array([[100, 101, 102],
               [103, 104, 105]]),
         array([[-100,  -99,  -98],
               [ -97,  -96,  -95]])]

    This function creates a new outer dimension (at the start) that is one-larger
    than the highest-dimension array in the input, and glues the input arrays along
    that dimension. The dimensions are aligned along the last one, so broadcasting
    will continue to work as expected. Note that this is the opposite operation from
    iterating a numpy array; see the example above.

    '''
    return glue(*args) # axis is unspecified


def clump(x, **kwargs):
    r'''Groups the given n most significant dimensions together.

    Synopsis:

        >>> nps.clump( arr(2,3,4), n=2).shape
        (2, 12)
    '''
    n = kwargs.get('n')
    if n is None:
        raise NumpysaneError("clump() requires a dimension count in the 'n' kwarg")
    if n < 0:
        raise NumpysaneError("clump() requires n > 0")
    if n <= 1:
        return x

    if x.ndim < n:
        n = x.ndim

    s = list(x.shape[:-n]) + [ reduce( lambda a,b: a*b, x.shape[-n:]) ]
    return x.reshape(s)

def atleast_dims(x, *dims):
    r'''Returns an array with extra length-1 dimensions to contain all given axes.

    If the given axes already exist in the given array, the given array itself
    is returned. Otherwise length-1 dimensions are added to the front until all
    the requested dimensions exist. Only <0 out-bounds axes are allowed to keep
    the last dimension anchored

    '''
    if max(dims) >= x.ndim:
        raise NumpysaneError("Axis {} out of bounds because x.ndim={}.\n"
                             "To keep the last dimension anchored, "
                             "only <0 out-of-bounds axes are allowed".format(max(dims), x.ndim))

    need_ndim = -min(d if d<0 else -1 for d in dims)
    if x.ndim >= need_ndim:
        return x
    return x[ (np.newaxis,)*(need_ndim-x.ndim) + (_colon,)*x.ndim ]

def mv(x, axis_from, axis_to):
    r'''Moves a given axis to a new position. Same as numpy.moveaxis().

    New length-1 dimensions are added at the front, as required.

    '''
    x = atleast_dims(x, axis_from, axis_to)
    return np.moveaxis( x, axis_from, axis_to )

def xchg(x, axis_a, axis_b):
    r'''Exchanges the positions of the two given axes. Same as numpy.swapaxes()

    New length-1 dimensions are added at the front, as required.

    '''
    x = atleast_dims(x, axis_a, axis_b)
    return np.swapaxes( x, axis_a, axis_b )

def transpose(x):
    r'''Reverses the order of the last two dimensions.

    A "matrix" is generally seen as a 2D array that we can transpose by looking
    at the 2 dimensions in the opposite order. Here we treat an n-dimensional
    array as an n-2 dimensional object containing 2D matrices. As usual, the
    last two dimensions contain the matrix.

    New length-1 dimensions are added at the front, as required.

    '''
    return xchg( atleast_dims(x, -2), -1, -2)

def dummy(x, axis=None):
    r'''Adds a single length-1 dimension at the given position.

    This is very similar to numpy.expand_dims(), but handles out-of-bounds
    dimensions much better

    '''
    need_ndim = axis+1 if axis >= 0 else -axis
    if x.ndim >= need_ndim:
        # referring to an axis that already exists. expand_dims() thus works
        return np.expand_dims(x, axis)

    # referring to a non-existing axis. I simply add sufficient new axes, and
    # I'm done
    return atleast_dims(x, axis)

def reorder(x, *dims):
    r'''Reorders the dimensions of an array.

    This is very similar to numpy.transpose(), but handles out-of-bounds
    dimensions much better

    '''
    x = atleast_dims(x, *dims)
    return np.transpose(x, dims)

def broadcast_define(*prototype):
    r'''Vectorizes an arbitrary function, expecting input as in the given prototype.

    Synopsis:

        >>> import numpy as np
        >>> import numpysane as nps

        >>> @nps.broadcast_define( ('n',), ('n',) )
        ... def inner_product(a, b):
        ...     return a.dot(b)

        >>> a = np.arange(6).reshape(2,3)
        >>> b = a + 100

        >>> a
        array([[0, 1, 2],
               [3, 4, 5]])

        >>> b
        array([[100, 101, 102],
               [103, 104, 105]])

        >>> inner_product(a,b)
        array([ 305, 1250])


    The prototype defines the dimensionality of the inputs. In the basic inner
    product example above, the input is two 1D n-dimensional vectors. In
    particular, the 'n' is the same for the two inputs. This function is
    intended to be used as a decorator, applied to a function defining the
    operation to be vectorized. Each element of the prototype list refers to
    each input, in order. In turn, each prototype element is a list that
    describes the shape of that input. Each of these shape descriptors can be
    any of

    - a positive integer, indicating an input dimension of exactly that length
    - a string, indicating an arbitrary, but internally consistent dimension

    The normal numpy broadcasting rules (as described elsewhere) apply. In
    summary:

    - Dimensions are aligned at the end of the shape list, and must match the
      prototype

    - Extra dimensions left over at the front must be consistent for all the
      input arguments, meaning:

      - All dimensions !=1 must be identical
      - Dimensions that are =1 are implicitly extended to the lengths implied by
        other arguments

      - The output has a shape where

        - The trailing dimensions are whatever the function being broadcasted
          outputs
        - The leading dimensions come from the extra dimensions in the inputs


    Let's look at a more involved example. Let's say we have an function that
    takes a set of points in R^2 and a single center point in R^2, and finds a
    best-fit least-squares line that passes through the given center point. Let
    it return a 3D vector containing the slope, y-intercept and the RMS residual
    of the fit. This broadcasting-enabled function can be defined like this:

        import numpy as np
        import numpysane as nps

        @nps.broadcast_define( ('n',2), (2,) )
        def fit(xy, c):
            # line-through-origin-model: y = m*x
            # E = sum( (m*x - y)**2 )
            # dE/dm = 2*sum( (m*x-y)*x ) = 0
            # ----> m = sum(x*y)/sum(x*x)
            x,y = (xy - c).transpose()
            m = np.sum(x*y) / np.sum(x*x)
            err = m*x - y
            err **= 2
            rms = np.sqrt(err.mean())
            # I return m,b because I need to translate the line back
            b = c[1] - m*c[0]

            return np.array((m,b,rms))

    And I can use broadcasting to compute a number of these fits at once. Let's
    say I want to compute 4 different fits of 5 points each. I can do this:

        n = 5
        m = 4
        c = np.array((20,300))
        xy = np.arange(m*n*2, dtype=np.float64).reshape(m,n,2) + c
        xy += np.random.rand(*xy.shape)*5

        res = fit( xy, c )
        mb  = res[..., 0:2]
        rms = res[..., 2]
        print "RMS residuals: {}".format(rms)

    Here I had 4 different sets of points, but a single center point c. If I
    wanted 4 different center points, I could pass c as an array of shape (4,2).
    I can use broadcasting to plot all the results (the points and the fitted
    lines):

        import gnuplotlib as gp

        gp.plot( xy[..., 0], xy[..., 1], _with='linespoints',
                 equation=['{}*x + {}'.format(mb_single[0],
                                              mb_single[1]) for mb_single in mb],
                 unset='grid', square=1)

    This function is analogous to thread_define() in PDL.

    '''
    def inner_decorator_for_some_reason(func):
        def range_rev(n):
            r'''Returns a range from -1 to -n.

            Useful to index variable-sized lists while aligning their ends.'''
            return [-i-1 for i in range(n)]

        def parse_dims( name_arg,
                        shape_prototype, shape_arg, dims_extra, dims_named ):
            # first, I make sure the input is at least as dimension-ful as the
            # prototype. I make this a hard requirement. Even if the missing
            # dimensions have length 1, it is likely a mistake on the part of the
            # user
            if len(shape_prototype) > len(shape_arg):
                raise NumpysaneError("Argument {} has {} dimensions, but the prototype {} demands at least {}".format(name_arg, len(shape_arg), shape_prototype, len(shape_prototype)))

            # Loop through the dimensions. Set the dimensionality of any new named
            # argument to whatever the current argument has. Any already-known
            # argument must match
            for i_dim in range_rev(len(shape_prototype)):
                if type(shape_prototype[i_dim]) is not int and \
                   shape_prototype[i_dim] not in dims_named:
                    dims_named[shape_prototype[i_dim]] = shape_arg[i_dim]

                # The prototype dimension (named or otherwise) now has a numeric
                # value. Make sure it matches what I have
                dim_prototype = dims_named[shape_prototype[i_dim]] \
                                if type(shape_prototype[i_dim]) is not int \
                                   else shape_prototype[i_dim]

                if dim_prototype != shape_arg[i_dim]:
                    raise NumpysaneError("Argument {} dimension '{}': expected {} but got {}".
                        format(name_arg,
                               shape_prototype[i_dim],
                               dim_prototype,
                               shape_arg[i_dim]))

            # I now know that this argument matches the prototype. I look at the
            # extra dimensions to broadcast, and make sure they match with the
            # dimensions I saw previously
            ndims_extra_here = len(shape_arg) - len(shape_prototype)

            # This argument has ndims_extra_here dimensions to broadcast. The
            # current shape to broadcast must be at least as large, and must match
            if ndims_extra_here > len(dims_extra):
                dims_extra[:0] = [1] * (ndims_extra_here - len(dims_extra))
            for i_dim in range_rev(ndims_extra_here):
                dim_arg = shape_arg[i_dim - len(shape_prototype)]
                if dim_arg != 1:
                    if dims_extra[i_dim] == 1:
                        dims_extra[i_dim] = dim_arg
                    elif dims_extra[i_dim] != dim_arg:
                        raise NumpysaneError("Argument {} prototype {} extra broadcast dim {} mismatch: previous arg set this to {}, but this arg wants {}".
                            format(name_arg,
                                   shape_prototype,
                                   i_dim,
                                   dims_extra[i_dim],
                                   dim_arg))


        # args broadcast, kwargs do not. All auxillary data should go into the
        # kwargs
        def broadcast_loop(*args, **kwargs):

            if len(prototype) != len(args):
                raise NumpysaneError("Mismatched number of input arguments. Wanted {} but got {}". \
                                      format(len(prototype), len(args)))

            dims_extra = [] # extra dimensions to broadcast through
            dims_named = {} # named dimension lengths
            for i_arg in range(len(args)):
                parse_dims( i_arg,
                            prototype[i_arg], args[i_arg].shape,
                            dims_extra, dims_named )


            # I checked all the dimensions and aligned everything. I have my
            # to-broadcast dimension counts. Iterate through all the broadcasting
            # output, and gather the results

            def accum_dim( i_dims_extra, idx_slices, idx_extra ):
                r'''Recursive function to iterate through all the broadcasting slices.

                Each recursive call loops through a single dimension. I can do
                some of this with itertools.product(), and maybe using that
                would be a better choice.

                i_dims_extra is an integer indexing the current extra dimension
                we're looking at.

                idx_slices is an array of indices for each argument that is
                filled in by this function. This may vary for each argument
                because of varying prototypes and varying broadcasting shapes.

                '''

                if i_dims_extra < len(dims_extra):
                    # more dimensions remaining. recurse

                    # idx_slices contains the indices for each argument.
                    # Two notes:
                    #
                    # 1. Any indices indexing above the first dimension should be
                    #    omitted
                    # 2. Indices into a higher dimension of length 1 should be left at 0

                    for dim in range(dims_extra[i_dims_extra]):
                        for x,p,idx_slice in zip(args,prototype,idx_slices):
                            x_dim = x.ndim - (len(p) + len(dims_extra) - i_dims_extra)
                            if x_dim >= 0 and x.shape[x_dim] > 1:
                                idx_slice[x_dim] = dim

                        idx_extra[i_dims_extra] = dim
                        accum_dim(i_dims_extra+1, idx_slices, idx_extra)
                    return


                # This is the last dimension. Evaluate this slice.
                #
                sliced_args = [ x[idx] for idx,x in zip(idx_slices, args) ]
                result = func( *sliced_args, **kwargs )
                if accum_dim.output is None:
                    accum_dim.output = np.zeros( dims_extra + list(result.shape),
                                                 dtype = result.dtype)
                accum_dim.output[idx_extra + [Ellipsis]] = result


            accum_dim.output = None

            idx_slices = [[0]*(x.ndim-len(p)) + [_colon]*len(p) for p,x in zip(prototype,args)]
            accum_dim( 0, idx_slices, [0] * len(dims_extra) )
            return accum_dim.output


        return broadcast_loop
    return inner_decorator_for_some_reason
