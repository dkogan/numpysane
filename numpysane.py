#!/usr/bin/python


import numpy as np

debugprint = None


# object needed for fancy slices. m[:] is exactly the same as
# m[colon], but 'colon' can be manipulated in ways that ':' can't
colon = slice(None, None, None)


class NumpySaneError(Exception):
    def __init__(self, err): self.err = err
    def __str__(self):       return self.err

def glue(*args, **kwargs):
    """Concatenates a given list of arrays along the dimension given by the 'axis'
keyword argument. If no such keyword argument is given, a new dimension is added
at the front, and we concatenate along that new dimension. This case is
equivalent to the cat() function.

It is STRONGLY recommended to use only negative axis indices in order to count
dimensions relative to the most-significant broadcasting axis, which in numpy is
the last one (axis = -1). You WILL write bugs if you don't do this."""

    axis = kwargs.get('axis')
    if axis is not None and axis >= 0:
        raise NumpySaneError("axis >= 0 makes broadcasting dimensions inconsistent, and is thus not allowed")

    # deal with scalar (non-ndarray) args
    args = [ np.asarray(x) for x in args ]
    if debugprint:
        print args

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
    """Concatenates a given list of arrays along a new first dimension. The
dimensions are aligned along the last one, so broadcasting will continue to work
as expected"""
    return glue(*args) # axis is unspecified



def broadcast_define(*prototype):
    """Vectorizes an arbitrary function 'func', expecting input as in the prototype

    an integer (assumed >0) means "exactly this many". Otherwise, it's a name
    of a variable whose value must be consistent.

    So this means inputs are ( 3-vector, n,3-matrix, n-vector, m-vector)
    prototype = ( (3,), ('n',3), ('n',), ('m',))
"""

    def inner_decorator_for_some_reason(func):
        # a "reversed" range iterator. Does this:
        #   In [3]: range_rev(10)
        #   Out[3]: [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
        #
        def range_rev(n):
            """Useful to index variable-sized lists while aligning their ends."""
            return [-i-1 for i in range(n)]

        def parse_dims( name_arg,
                        shape_prototype, shape_arg, dims_extra, dims_named ):
            # first, I make sure the input is at least as dimension-ful as the
            # prototype. I make this a hard requirement. Even if the missing
            # dimensions have length 1, it is likely a mistake on the part of the
            # user
            if len(shape_prototype) > len(shape_arg):
                raise NumpySaneError("Argument {} has {} dimensions, but the prototype {} demands at least {}".format(name_arg, len(shape_arg), shape_prototype, len(shape_prototype)))

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
                    raise NumpySaneError("Argument {} dimension '{}': expected {} but got {}".
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
                        raise NumpySaneError("Argument {} prototype {} extra broadcast dim {} mismatch: previous arg set this to {}, but this arg wants {}".
                            format(name_arg,
                                   shape_prototype,
                                   i_dim,
                                   dims_extra[i_dim],
                                   dim_arg))


        # args broadcast, kwargs do not. All auxillary data should go into the
        # kwargs
        def broadcast_loop(*args, **kwargs):

            if len(prototype) != len(args):
                raise NumpySaneError("Mismatched number of input arguments. Wanted {} but got {}". \
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
                """Recursive function to iterate through all the broadcasting slices. Each
recursive call loops through a single dimension. I can do some of this with
itertools.product(), and maybe using that would be a better choice.

i_dims_extra is an integer indexing the current extra dimension we're looking at.

idx_slices is an array of indices for each argument that is filled in by this
function. This may vary for each argument because of varying prototypes and
varying broadcasting shapes."""

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
                result = func( *sliced_args )
                if accum_dim.output is None:
                    accum_dim.output = np.zeros( dims_extra + list(result.shape),
                                                 dtype = result.dtype)
                accum_dim.output[idx_extra + [Ellipsis]] = result


            accum_dim.output = None

            idx_slices = [[0]*(x.ndim-len(p)) + [colon]*len(p) for p,x in zip(prototype,args)]
            accum_dim( 0, idx_slices, [0] * len(dims_extra) )
            return accum_dim.output


        return broadcast_loop
    return inner_decorator_for_some_reason





if __name__ == '__main__':
    # test concatenation routines

    # test broadcast_define

    def mkarr(*shape):
        """Return an arange() array of the given shape."""
        product = reduce( lambda x,y: x*y, shape)
        return np.arange(product).reshape(*shape)


    @broadcast_define( ('n',), ('n') )
    def f1(a, b):
        """Basic inner product."""
        return a.dot(b)

    print f1(mkarr(3), mkarr(3))



    #@broadcast_define( (3,), ('n',3), ('n',), ('m',) )
    n=4
    m=5

    a = mkarr(1,5,    3)
    b = mkarr(2,5,  n,3)
    c = mkarr(        n)
    d = mkarr(  5,    m)
