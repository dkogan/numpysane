r'''Python-wrap C code with broadcasting awareness

* SYNOPSIS

Let's implement a broadcastable and type-checked inner product that is

- Written in C (i.e. it is fast)
- Callable from python using numpy arrays (i.e. it is convenient)

We write a bit of python to generate the wrapping code. "genpywrap.py":

    import numpy     as np
    import numpysane as nps
    import numpysane_pywrap as npsp

    m = npsp.module( name      = "innerlib",
                     docstring = "An inner product module in C")
    m.function( "inner",
                "Inner product pywrapped with npsp",

                args_input       = ('a', 'b'),
                prototype_input  = (('n',), ('n',)),
                prototype_output = (),

                Ccode_slice_eval = \
                    {np.float64:
                     r"""
                       double* out = (double*)data_slice__output;
                       const int N = dims_slice__a[0];

                       *out = 0.0;

                       for(int i=0; i<N; i++)
                         *out += *(const double*)(data_slice__a +
                                                  i*strides_slice__a[0]) *
                                 *(const double*)(data_slice__b +
                                                  i*strides_slice__b[0]);
                       return true;""" })
    m.write()

We run this, and save the output to "inner_pywrap.c":

    python3 genpywrap.py > inner_pywrap.c

We build this into a python module:

    COMPILE=(`python3 -c "
    import sysconfig
    conf = sysconfig.get_config_vars()
    print('{} {} {} -I{}'.format(*[conf[x] for x in ('CC',
                                                     'CFLAGS',
                                                     'CCSHARED',
                                                     'INCLUDEPY')]))"`)
    LINK=(`python3 -c "
    import sysconfig
    conf = sysconfig.get_config_vars()
    print('{} {} {}'.format(*[conf[x] for x in ('BLDSHARED',
                                                'BLDLIBRARY',
                                                'LDFLAGS')]))"`)
    EXT_SUFFIX=`python3 -c "
    import sysconfig
    print(sysconfig.get_config_vars('EXT_SUFFIX')[0])"`

    ${COMPILE[@]} -c -o inner_pywrap.o inner_pywrap.c
    ${LINK[@]} -o innerlib$EXT_SUFFIX inner_pywrap.o

Here we used the build commands directly. This could be done with
setuptools/distutils instead; it's a normal extension module. And now we can
compute broadcasted inner products from a python script "tst.py":

    import numpy as np
    import innerlib
    print(innerlib.inner( np.arange(4, dtype=float),
                          np.arange(8, dtype=float).reshape( 2,4)))

Running it to compute inner([0,1,2,3],[0,1,2,3]) and inner([0,1,2,3],[4,5,6,7]):

    $ python3 tst.py
    [14. 38.]

* DESCRIPTION
This module provides routines to python-wrap existing C code by generating C
sources that define the wrapper python extension module.

To create the wrappers we

1. Instantiate a new numpysane_pywrap.module class
2. Call module.function() for each wrapper function we want to add to this
   module
3. Call module.write() to write the C sources defining this module to standard
   output

The sources can then be built and executed normally, as any other python
extension module. The resulting functions are called as one would expect:

    output                  = f_one_output      (input0, input1, ...)
    (output0, output1, ...) = f_multiple_outputs(input0, input1, ...)

depending on whether we declared a single output, or multiple outputs (see
below). It is also possible to pre-allocate the output array(s), and call the
functions like this (see below):

    output = np.zeros(...)
    f_one_output      (input0, input1, ..., out = output)

    output0 = np.zeros(...)
    output1 = np.zeros(...)
    f_multiple_outputs(input0, input1, ..., out = (output0, output1))

Each wrapped function is broadcasting-aware. The normal numpy broadcasting rules
(as described in 'broadcast_define' and on the numpy website:
http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) apply. In
summary:

- Dimensions are aligned at the end of the shape list, and must match the
  prototype

- Extra dimensions left over at the front must be consistent for all the
  input arguments, meaning:

  - All dimensions of length != 1 must match
  - Dimensions of length 1 match corresponding dimensions of any length in
    other arrays
  - Missing leading dimensions are implicitly set to length 1

- The output(s) have a shape where
  - The trailing dimensions match the prototype
  - The leading dimensions come from the extra dimensions in the inputs

When we create a wrapper function, we only define how to compute a single
broadcasted slice. If the generated function is called with higher-dimensional
inputs, this slice code will be called multiple times. This broadcast loop is
produced by the numpysane_pywrap generator automatically. The generated code
also

- parses the python arguments
- generates python return values
- validates the inputs (and any pre-allocated outputs) to make sure the given
  shapes and types all match the declared shapes and types. For instance,
  computing an inner product of a 5-vector and a 3-vector is illegal
- creates the output arrays as necessary

This code-generator module does NOT produce any code to implicitly make copies
of the input. If the inputs fail validation (unknown types given, contiguity
checks failed, etc) then an exception is raised. Copying the input is
potentially slow, so we require the user to do that, if necessary.

** Explicated example

In the synopsis we declared the wrapper module like this:

    m = npsp.module( name      = "innerlib",
                     docstring = "An inner product module in C")

This produces a module named "innerlib". Note that the python importer will look
for this module in a file called "innerlib$EXT_SUFFIX" where EXT_SUFFIX comes
from the python configuration. This is normal behavior for python extension
modules.

A module can contain many wrapper functions. Each one is added by calling
'm.function()'. We did this:

    m.function( "inner",
                "Inner product pywrapped with numpysane_pywrap",

                args_input       = ('a', 'b'),
                prototype_input  = (('n',), ('n',)),
                prototype_output = (),

                Ccode_slice_eval = \
                    {np.float64:
                     r"""
                       double* out = (double*)data_slice__output;
                       const int N = dims_slice__a[0];

                       *out = 0.0;

                       for(int i=0; i<N; i++)
                         *out += *(const double*)(data_slice__a +
                                                  i*strides_slice__a[0]) *
                                 *(const double*)(data_slice__b +
                                                  i*strides_slice__b[0]);
                       return true;""" })

We declared:

- A function "inner" with the given docstring
- two inputs to this function: named 'a' and 'b'. Each is a 1-dimensional array
  of length 'n', same 'n' for both arrays
- one output: a scalar
- how to compute a single inner product where all inputs and outputs are 64-bit
  floating-point values: this snippet of C is included in the generated sources
  verbatim

It is possible to support multiple sets of types by passing more key/value
combinations in 'Ccode_slice_eval'. Each set of types requires a different C
snippet. If the input doesn't match any known type set, an exception will be
thrown. More on the type matching below.

The length of the inner product is defined by the length of the input, in this
case 'dims_slice__a[0]'. I could have looked at 'dims_slice__b[0]' instead, but
I know it's identical: the 'prototype_input' says that both 'a' and 'b' have
length 'n', and if we're running the slice code snippet, we know that the inputs
have already been checked, and have compatible dimensionality. More on this
below.

I did not assume the data is contiguous, so I use 'strides_slice__a' and
'strides_slice__b' to index the input arrays. We could add a validation function
that accepts only contiguous input; if we did that, the slice code snippet could
assume contiguous data and ignore the strides. More on that below.

Once all the functions have been added, we write out the generated code to
standard output by invoking

    m.write()

** Dimension specification
The shapes of the inputs and outputs are given in the 'prototype_input' and
'prototype_output' arguments respectively. This is similar to how this is done
in 'numpysane.broadcast_define()': each prototype is a tuple of shapes, one for
each argument. Each shape is given as a tuple of sizes for each expected
dimension. Each size can be either

- a positive integer if we know the expected dimension size beforehand, and only
  those sizes are accepted

- a string that names the dimension. Any size could be accepted for a named
  dimension, but for any given named dimension, the sizes must match across all
  inputs and outputs

Unlike 'numpysane.broadcast_define()', the shapes of both inputs and outputs
must be defined here: the output shape may not be omitted.

The common special case of a single output is supported: this one output is
specified in 'prototype_output' as a single shape, instead of a tuple of shapes.
This also affects whether the resulting python function returns the one output
or a tuple of outputs.

Examples:

A function taking in some 2D vectors and the same number of 3D vectors:

    prototype_input  = (('n',2), ('n',3))

A function producing a single 2D vector:

    prototype_output = (2,)

A function producing 3 outputs: some number of 2D vectors, a single 3D vector
and a scalar:

    prototype_output = (('n',2), (3,), ())

Note that when creating new output arrays, all the dimensions must be known from
the inputs. For instance, given this, we cannot create the output:

    prototype_input  = ((2,), ('n',))
    prototype_output = (('m',), ('m', 'm'))

I have the inputs, so I know 'n', but I don't know 'm'. When calling a function
like this, it is required to pass in pre-allocated output arrays instead of
asking the wrapper code to create new ones. See below.

** In-place outputs
As with 'numpysane.broadcast_define()', the caller of the generated python
function may pre-allocate the output and pass it in the 'out' kwarg to be
filled-in. Sometimes this is required if we want to avoid extra copying of data.
This is also required if the output prototypes have any named dimensions not
present in the input prototypes: in this case we dont know how large the output
arrays should be, so we can't create them.

If a wrapped function is called this way, we check that the dimensions and types
in the outputs match the prototype. Otherwise, we create a new output array with
the correct type and shape.

If we have multiple outputs, the in-place arrays are given as a tuple of arrays
in the 'out' kwarg. If any outputs are pre-allocated, all of them must be.

Example. Let's use the inner-product we defined earlier. We compute two sets of
inner products. We make two calls to inner(), each one broadcasted to produce
two inner products into a non-contiguous slice of an output array:

    import numpy as np
    import innerlib

    out=np.zeros((2,2), dtype=float)
    innerlib.inner( np.arange(4, dtype=float),
                    np.arange(8, dtype=float).reshape( 2,4),
                    out=out[:,0] )
    innerlib.inner( 1+np.arange(4, dtype=float),
                    np.arange(8, dtype=float).reshape( 2,4),
                    out=out[:,1] )
    print(out)

The first two inner products end up in the first column of the output, and the
next two inner products in the second column:

    $ python3 tst.py

    [[14. 20.]
     [38. 60.]]

If we have a function "f" that produces two outputs, we'd do this:

    output0 = np.zeros(...)
    output1 = np.zeros(...)
    f( ..., out = (output0, output1) )

** Type checking
Since C code is involved, we must be very explicit about the types of our
arrays. These types are specified in the keys of the 'Ccode_slice_eval'
argument to 'function()'. For each type specification in a key, the
corresponding value is a C code snippet to use for that type spec. The type
specs can be either

- A type known by python and acceptable to numpy as a valid dtype. In this usage
  ALL inputs and ALL outputs must have this type
- A tuple of types. The elements of this tuple correspond to each input, in
  order, followed by each output, in order. This allows different arguments to
  have different types

It is up to the user to make sure that the C snippet they provide matches the
types that they declared.

Example. Let's extend the inner product to know about 32-bit floats and also
about producing a rounded integer inner product from 64-bit floats:

    m = npsp.module( name      = "innerlib",
                     docstring = "An inner product module in C",
                     header    = "#include <stdint.h>")
    m.function( "inner",
                "Inner product pywrapped with numpysane_pywrap",

                args_input       = ('a', 'b'),
                prototype_input  = (('n',), ('n',)),
                prototype_output = (),

                Ccode_slice_eval = \
                    {np.float64:
                     r"""
                       double* out = (double*)data_slice__output;
                       const int N = dims_slice__a[0];

                       *out = 0.0;

                       for(int i=0; i<N; i++)
                         *out += *(const double*)(data_slice__a +
                                                  i*strides_slice__a[0]) *
                                 *(const double*)(data_slice__b +
                                                  i*strides_slice__b[0]);
                       return true;""",
                     np.float32:
                     r"""
                       float* out = (float*)data_slice__output;
                       const int N = dims_slice__a[0];

                       *out = 0.0;

                       for(int i=0; i<N; i++)
                         *out += *(const float*)(data_slice__a +
                                                 i*strides_slice__a[0]) *
                                 *(const float*)(data_slice__b +
                                                 i*strides_slice__b[0]);
                       return true;""",
                     (np.float64, np.float64, np.int32):
                     r"""
                       double out = 0.0;
                       const int N = dims_slice__a[0];

                       for(int i=0; i<N; i++)
                         out += *(const double*)(data_slice__a +
                                                 i*strides_slice__a[0]) *
                                *(const double*)(data_slice__b +
                                                 i*strides_slice__b[0]);
                       *(int32_t*)data_slice__output = (int32_t)round(out);
                       return true;""" })

** Verbatim code snippets
As we have seen, some functionality is passed to numpysane_pywrap as C code,
included into the generated source verbatim. There are two areas where such C
code is used: argument validation and slice computation.

*** Argument validation
After the wrapping code confirms that all the shapes and types match the
prototype, it calls a user-provided validation routine once to flag any extra
conditions that are required. A common use case: we're wrapping some C code that
assumes the input data is stored contiguously in memory, so the validation
routine checks that this is true.

This code snippet is provided in the 'Ccode_validate' argument to 'function()'.
The result is returned as a boolean: if the checks pass, we return true. If the
checks fail, we return false, which will result in an exception being thrown. If
you want to throw your own, more informative exception, you can do that as usual
(by calling something like PyErr_Format()) before returning false.

If the 'Ccode_validate' argument is omitted, no additional checks are performed,
and we accept all calls that satisfied the broadcasting and type requirements.

*** Slice computation
This code is executed once for each broadcasted slice to actually do the thing
we're wrapping. This code snippet is required, and is provided in values of the
'Ccode_slice_eval' dict passed to 'function()', as we have seen in the
samples. This also returns a boolean: true on success, false on failure. If
false is ever returned, all subsequent slices are abandoned, and an exception is
thrown. As with the validation code, you can throw a better exception yourself
prior to returning false.

*** Arguments available to the code snippets
Each of the user-supplied code blocks is placed into a separate function in the
generated code, with identical arguments in both cases. These arguments describe
the inputs and outputs, and are meant to be used by the user code. We have
dimensionality information:

    const int       Ndims_full__NAME
    const npy_intp* dims_full__NAME
    const int       Ndims_slice__NAME
    const npy_intp* dims_slice__NAME

where "NAME" is the name of the input or output. The input names are given in
the 'args_input' argument to 'function()'. If we have a single output, the
output name is "output". If we have multiple outputs, their names are "output0",
"output1", ... The ...full... arguments describe the full array, that describes
ALL the broadcasted slices. The ...slice... arguments describe each broadcasted
slice separately. Under most usages, you want the ...slice... information
because the C code we're wrapping only sees one slice at a time. Ndims...
describes how many dimensions we have in the corresponding dims... arrays.
npy_intp is a long integer used internally by numpy for dimension information.

We have memory layout information:

    const npy_intp* strides_full__NAME
    const npy_intp* strides_slice__NAME
    npy_intp        sizeof_element__NAME

NAME and full/slice and npy_intp have the same meanings as before. The
strides... arrays each have length described by the corresponding dims... The
strides contain the step size in bytes, of each dimension. sizeof_element...
describes the size in bytes, of a single data element.

Finally, I have a pointer to the data itself. The validation code gets a pointer
to the start of the data array:

    void*           data__NAME

but the computation code gets a pointer to the start of the slice we're
currently looking at:

    void*           data_slice__NAME

Example: I'm computing a broadcasted slice. An input array 'x' is a
2-dimensional slice of dimension (3,4) of 64-bit floating-point values. I thus
have Ndims_slice__x == 2 and dims_slice__x[] = {3,4} and sizeof_element__x == 8.
An element of this array at i,j is

    *((double*)(data_slice__a + i*strides_slice__a[0] + j*strides_slice__a[1]))

If I defined a validation function that makes sure that 'a' is stored in
contiguous memory, the computation code doesn't need to look at the strides at
all, and element at i,j can be found more simply:

    ((double*)data_slice__a)[ i*dims_slice__a[1] + j ]

*** Contiguity checking
Since checking for memory contiguity is a very common use case for argument
validation, there are convenience macros provided:

    CHECK_CONTIGUOUS__NAME()
    CHECK_CONTIGUOUS_AND_SETERROR__NAME()

    CHECK_CONTIGUOUS_ALL()
    CHECK_CONTIGUOUS_AND_SETERROR_ALL()

The strictest, and most common usage will accept only those calls where ALL
inputs and ALL outputs are stored in contiguous memory. This can be accomplished
by defining the function like

    m.function( ...,
               Ccode_validate = 'return CHECK_CONTIGUOUS_AND_SETERROR_ALL();' )

As before, "NAME" refers to each individual input or output, and "ALL" checks
all of them. These all evaluate to true if the argument in questions IS
contiguous. The ..._AND_SETERROR_... flavor does that, but ALSO raises an
informative exception.

Generally you want to do this in the validation routine only, since it runs only
once. But there's nothing stopping you from checking this in the computation
function too.

Note that each broadcasted slice is processed separately, so the C code being
wrapped usually only cares about each SLICE being contiguous. If the dimensions
above each slice (those being broadcasted) are not contiguous, this doesn't
break the underlying assumptions. Thus the CHECK_CONTIGUOUS_... functions only
check and report the in-slice contiguity. If for some reason you need more than
this, you should write the check yourself, using the strides_full__... and
dims_full__... arrays.

*** Extra arguments
Sometimes it is desired to pass extra arguments to the python wrapper that
aren't broadcasted in any way, but are just passed verbatim to the inner
functions. We can do that with the 'extra_args' argument to 'function()'. This
argument is an tuple of tuples of strings:

    (c_type, name, default_value, parse_arg)

The "c_type" is the C type of the argument ("int", "double", etc). The "name" is
the C name of the argument. This type and name will be available to the argument
validation and slice computation routines as a pointer to the given c_type.

When calling this function, passing these extra arguments is optional. If
omitted, we use the "default_value". Finally, when interpreting the passed
arguments we use PyArg_ParseTupleAndKeywords, and "parse_arg" is the code used
by that function to interpret the argument.

Example. Let's update our inner product example to accept a "scale" argument.

    m.function( "inner",
                "Inner product pywrapped with numpysane_pywrap",

                args_input       = ('a', 'b'),
                prototype_input  = (('n',), ('n',)),
                prototype_output = (),
                extra_args = (("double", "scale", "1", "d"),),

                Ccode_slice_eval = \
                    {np.float64:
                     r"""
                       double* out = (double*)data_slice__output;
                       const int N = dims_slice__a[0];

                       *out = 0.0;

                       for(int i=0; i<N; i++)
                         *out += *(const double*)(data_slice__a +
                                                  i*strides_slice__a[0]) *
                                 *(const double*)(data_slice__b +
                                                  i*strides_slice__b[0]);
                       *out *= *scale;
                       return true;""" })

Now I can optionally scale the result:

    >>> print(innerlib.inner( np.arange(4, dtype=float),
                              np.arange(8, dtype=float).reshape( 2,4)))
    [14. 38.]

    >>> print(innerlib.inner( np.arange(4, dtype=float),
                              np.arange(8, dtype=float).reshape( 2,4),
                              scale = 2.0))
    [28. 76.]

** Examples
For some sample usage, see the wrapper-generator used in the test suite:
https://github.com/dkogan/numpysane/blob/master/test/genpywrap.py

** Planned functionality
Currently, each broadcasted slice is computed sequentially. But since the slices
are inherently independent, this is a natural place to add parallelism. And
implemention this with something like OpenMP should be straightforward. I'll get
around to doing this eventually, but in the meantime, patches are welcome.

'''

import sys
import time
import numpy as np
from numpysane import NumpysaneError
import os

# Technically I'm supposed to use some "resource extractor" something to unbreak
# setuptools. But I'm instead assuming that this was installed via Debian or by
# using the eager_resources tag in setup(). This allows files to remain files,
# and to appear in a "normal" directory, where this script can grab them and use
# them
#
# Aand I try two different directories, in case I'm running in-place

_pywrap_path = ( os.path.dirname( __file__ ) + '/pywrap-templates',
                 sys.prefix + '/share/python-numpysane/pywrap-templates' )

for p in _pywrap_path:
    _module_header_filename = p + '/pywrap_module_header.c'
    _module_footer_filename = p + '/pywrap_module_footer_generic.c'
    _function_filename      = p + '/pywrap_function_generic.c'

    if os.path.exists(_module_header_filename):
        break
else:
    raise NumpysaneError("Couldn't find pywrap templates! Looked in {}".format(_pywrap_path))


def _quote(s, convert_newlines=False):
    r'''Quote string for inclusion in C code

    There should be a library for this. Hopefuly this is correct.

    '''
    s = s.replace('\\', '\\\\')     # Pass all \ through verbatim
    if convert_newlines:
       s = s.replace('\n', '\\n')   # All newlines -> \n
    s = s.replace('"',  '\\"')      # Quote all "
    return s


def _substitute(s, convert_newlines=False, **kwargs):
    r'''format() with specific semantics

    - {xxx} substitutions founc in kwargs are made
    - {xxx} expressions not found in kwargs are left as is
    - {{ }} escaping is not respected: any '{xxx}' is replaced
    - \ and " and \n are handled for C strings
    - if convert_newlines: newlines are converted to \n (useful for C strings).
      Otherwise they're left alone (useful for C code)

    '''

    for k in kwargs.keys():
        v = kwargs[k]
        if isinstance(v, str):
            v = _quote(v, convert_newlines)
        else:
            v = str(v)
        s = s.replace('{' + k + '}', kwargs[k])
    return s



class module:
    def __init__(self, name, docstring, header=''):
        r'''Initialize the python-wrapper-generator

        SYNOPSIS

            import numpysane_pywrap as npsp
            m = npsp.module( name      = "wrapped_library",
                             docstring = r"""wrapped by numpysane_pywrap

                                         Does this thing and does that thing""",
                             header    = '#include "library.h"')

        ARGUMENTS

        - name
          The name of the python module we're creating

        - docstring
          The docstring for this module

        - header
          Optional, defaults to ''. C code to include verbatim. Any #includes or
          utility functions can go here

        '''

        with open( _module_header_filename, 'r') as f:
            self.module_header = f.read() + "\n" + header + "\n"

        with open( _module_footer_filename, 'r') as f:
            self.module_footer = _substitute(f.read(),
                                             MODULE_NAME      = name,
                                             MODULE_DOCSTRING = docstring,
                                             convert_newlines = True)

        self.functions = []


    def function(self,
                 name,
                 docstring,
                 args_input,
                 prototype_input,
                 prototype_output,
                 Ccode_slice_eval,
                 Ccode_validate = None,
                 extra_args    = ()):
        r'''Add a wrapper function to the module we're creating

        SYNOPSIS

        We can wrap a C function inner() like this:

        m.function( "inner",
                    "Inner product pywrapped with npsp",

                    args_input       = ('a', 'b'),
                    prototype_input  = (('n',), ('n',)),
                    prototype_output = (),

                    Ccode_slice_eval = \
                        {np.float64:
                         r"""
                           double* out = (double*)data_slice__output;
                           const int N = dims_slice__a[0];

                           *out = 0.0;

                           for(int i=0; i<N; i++)
                             *out += *(const double*)(data_slice__a +
                                                      i*strides_slice__a[0]) *
                                     *(const double*)(data_slice__b +
                                                      i*strides_slice__b[0]);
                           return true;""" })

        DESCRIPTION

        'function()' is the main workhorse of numpysane_pywrap python wrapping.
        For each C function we want to wrap, 'function()' should be called to
        generate the wrapper code. In the call to 'function()' the user
        specifies how the wrapper should compute a single broadcasted slice, and
        numpysane_pywrap generates the code to do everything else. See the
        numpysane_pywrap module docstring for lots of detail.


        ARGUMENTS

        A summary description of the arguments follows. See the numpysane_pywrap
        module docstring for detail.

        - name
          The name of the function we're wrapping. A python function of this
          name will be generated in this module

        - docstring
          The docstring for this function

        - args_input
          The names of the arguments. This is an tuple of strings. Must have
          the same number of elements as prototype_input

        - prototype_input
          An tuple of tuples that defines the shapes of the inputs. Must have
          the same number of elements as args_input. Each element of the outer
          tuple describes the shape of the corresponding input. Each shape is
          given as an tuple describing the length of each dimension. Each length
          is either

          - a positive integer if we know the expected dimension size
            beforehand, and only those sizes are accepted

          - a string that names the dimension. Any size could be accepted for a
            named dimension, but for any given named dimension, the sizes must
            match across all inputs and outputs

        - prototype_output
          Similar to prototype_input: describes the dimensions of each output.
          In the special case that we have only one output, this can be given as
          a shape tuple instead of a tuple of shape tuples.

        - Ccode_slice_eval
          This argument contains the snippet of C code used to execute the
          operation being wrapped. This argument is a dict mapping a type
          specification to code snippets: different data types require different
          C code to work with them. The type specification is either

          - a numpy type (np.float64, np.int32, etc). We'll use the given code
            if ALL the inputs and ALL the outputs are of this type

          - a tuple of numpy types. These correspond to the inputs and outputs,
            in order. This allows us to use different data types for the various
            inputs and outputs

          The corresponding code snippet is a string of C code that's inserted
          into the generated sources verbatim. Please see the numpysane_pywrap
          module docstring for more detail.

        - Ccode_validate
          A string of C code inserted into the generated sources verbatim. This
          is used to validate the input/output arguments prior to actually
          performing the computation. This runs after we made the broadcasting
          shape checks and type checks. If those checks are all we need, this
          argument may be omitted, and no more checks are made. The most common
          use case is rejecting inputs that are not stored contiguously in
          memory. Please see the numpysane_pywrap module docstring for more
          detail.

        - extra_args
          Defines extra arguments to accept in the validation and slice
          computation functions. These extra arguments are not broadcast or
          interpreted in any way; they're simply passed down from the python
          caller to these functions. Please see the numpysane_pywrap module
          docstring for more detail.

        '''

        if type(args_input) not in (list, tuple) or not all( type(arg) is str for arg in args_input):
            raise NumpysaneError("args_input MUST be a list or tuple of strings")

        Ninputs = len(args_input)
        if len(prototype_input) != Ninputs:
            raise NumpysaneError("Input prototype says we have {} arguments, but names for {} were given. These must match". \
                            format(len(prototype_input), Ninputs))

        # I enumerate each named dimension, starting from -1, and counting DOWN
        named_dims = {}
        if not isinstance(prototype_input, tuple):
            raise NumpysaneError("Input prototype must be given as a tuple")
        for i_arg in range(len(prototype_input)):
            dims_input = prototype_input[i_arg]
            if not isinstance(dims_input, tuple):
                raise NumpysaneError("Input prototype dims must be given as a tuple")
            for i_dim in range(len(dims_input)):
                dim = dims_input[i_dim]
                if isinstance(dim,int):
                    if dim < 0:
                        raise NumpysaneError("Dimension {} in argument '{}' must be a string (named dimension) or an integer>=0. Got '{}'". \
                                        format(i_dim, args_input[i_arg], dim))
                elif isinstance(dim, str):
                    if dim not in named_dims:
                        named_dims[dim] = -1-len(named_dims)
                else:
                    raise NumpysaneError("Dimension {} in argument '{}' must be a string (named dimension) or an integer>=0. Got '{}' (type '{}')". \
                                    format(i_dim, args_input[i_arg], dim, type(dim)))

        # The output is allowed to have named dimensions, but ONLY those that
        # appear in the input. The output may be a single tuple (describing the
        # one output) or it can be a tuple of tuples (describing multiple
        # outputs)
        if not isinstance(prototype_output, tuple):
            raise NumpysaneError("Output prototype dims must be given as a tuple")

        # If a single prototype_output is given, wrap it in a tuple to indicate
        # that we only have one output

        # If None, the single output is returned. If an integer, then a tuple is
        # returned. If Noutputs==1 then we return a TUPLE of length 1
        Noutputs = None
        if all( type(o) is int or type(o) is str for o in prototype_output ):
            prototype_outputs = (prototype_output, )
        else:
            prototype_outputs = prototype_output
            if not all( isinstance(p,tuple) for p in prototype_outputs ):
                raise NumpysaneError("Output dimensions must be integers > 0 or strings. Each output must be a tuple. Some given output aren't tuples: {}". \
                                     format(prototype_outputs))
            Noutputs = len(prototype_outputs)

        for i_output in range(len(prototype_outputs)):
            dims_output = prototype_outputs[i_output]
            for i_dim in range(len(dims_output)):
                dim = dims_output[i_dim]
                if isinstance(dim,int):
                    if dim < 0:
                        raise NumpysaneError("Output {} dimension {} must be a string (named dimension) or an integer>=0. Got '{}'". \
                                        format(i_output, i_dim, dim))
                elif isinstance(dim, str):
                    if dim not in named_dims:
                        # This output is a new named dimension. Output matrices must be passed in to define it
                        named_dims[dim] = -1-len(named_dims)
                else:
                    raise NumpysaneError("Dimension {} in output {} must be a string (named dimension) or an integer>=0. Got '{}' (type '{}')". \
                                    format(i_dim, i_output, dim, type(dim)))



        def expand_prototype(shape):
            r'''Produces a shape string for each argument

            This is the dimensions passed-into this function except, named
            dimensions are consolidated, and set to -1,-2,..., and the whole
            thing is stringified and joined

            '''

            shape = [ dim if isinstance(dim,int) else named_dims[dim] for dim in shape ]
            return ','.join(str(dim) for dim in shape)

        PROTOTYPE_DIM_DEFS = ''
        for i_arg_input in range(Ninputs):
            PROTOTYPE_DIM_DEFS += "    const npy_intp PROTOTYPE_{}[{}] = {{{}}};\n". \
                format(args_input[i_arg_input],
                       len(prototype_input[i_arg_input]),
                       expand_prototype(prototype_input[i_arg_input]));
        if Noutputs is None:
            PROTOTYPE_DIM_DEFS += "    const npy_intp PROTOTYPE_{}[{}] = {{{}}};\n". \
                format("output",
                       len(prototype_output),
                       expand_prototype(prototype_output));
        else:
            for i_output in range(Noutputs):
                PROTOTYPE_DIM_DEFS += "    const npy_intp PROTOTYPE_{}{}[{}] = {{{}}};\n". \
                    format("output",
                           i_output,
                           len(prototype_outputs[i_output]),
                           expand_prototype(prototype_outputs[i_output]));

        PROTOTYPE_DIM_DEFS += "    int Ndims_named = {};\n". \
            format(len(named_dims))


        # Output handling. We unpack each output array into a separate variable.
        # And if we have multiple outputs, we make sure that each one is
        # passed as a pre-allocated array
        #
        # At the start __py__output__arg has no reference
        if Noutputs is None:
            # Just one output. The argument IS the output array
            UNPACK_OUTPUTS = r'''
    int populate_output_tuple__i = -1;
    if(__py__output__arg == Py_None) __py__output__arg = NULL;
    if(__py__output__arg == NULL)
    {
        // One output, not given. Leave everything at NULL (it already is).
        // Will be allocated later
    }
    else
    {
        // Argument given. Treat it as an array
        Py_INCREF(__py__output__arg);
        if(!PyArray_Check(__py__output__arg))
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "Could not interpret given argument as a numpy array");
            goto done;
        }
        __py__output = (PyArrayObject*)__py__output__arg;
        Py_INCREF(__py__output);
    }
'''
        else:
            # Multiple outputs. Unpack, make sure we have pre-made arrays
            UNPACK_OUTPUTS = r'''
    int populate_output_tuple__i = -1;
    if(__py__output__arg == Py_None) __py__output__arg = NULL;
    if(__py__output__arg == NULL)
    {
        __py__output__arg = PyTuple_New({Noutputs});
        if(__py__output__arg == NULL)
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Could not allocate output tuple of length %d", {Noutputs});
            goto done;
        }

        // I made a tuple, but I don't yet have arrays to populate it with. I'll make
        // those later, and I'll fill the tuple later
        populate_output_tuple__i = 0;
    }
    else
    {
        Py_INCREF(__py__output__arg);

        if( !PySequence_Check(__py__output__arg) )
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Have multiple outputs. The given 'out' argument is expected to be a sequence of length %d, but a non-sequence was given",
                         {Noutputs});
            goto done;
        }
        if( PySequence_Size(__py__output__arg) != {Noutputs} )
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Have multiple outputs. The given 'out' argument is expected to be a sequence of length %d, but a sequence of length %d was given",
                         {Noutputs}, PySequence_Size(__py__output__arg));
            goto done;
        }

#define PULL_OUT_OUTPUT_ARRAYS(name)                                                                         \
        __py__ ## name = (PyArrayObject*)PySequence_GetItem(__py__output__arg, i++);                         \
        if(__py__ ## name == NULL || !PyArray_Check(__py__ ## name))                                         \
        {                                                                                                    \
            PyErr_SetString(PyExc_RuntimeError,                                                              \
                            "Have multiple outputs. The given 'out' array MUST contain pre-allocated arrays, but " #name " is not an array"); \
            goto done;                                                                                       \
        }
        int i=0;
        OUTPUTS(PULL_OUT_OUTPUT_ARRAYS)
#undef PULL_OUT_OUTPUT_ARRAYS
    }
'''.replace('{Noutputs}', str(Noutputs))


        Ntypesets = len(Ccode_slice_eval)
        slice_functions = [ "__{}__{}__slice".format(name,i) for i in range(Ntypesets)]
        # The keys of Ccode_slice_eval are either:

        # - a type: all inputs, outputs MUST have this type
        # - a list of types: the types in this list correspond to the inputs and
        #   outputs of the call, in that order.
        #
        # I convert the known-type list to the one-type-per-element form for
        # consistent processing
        known_types = list(Ccode_slice_eval.keys())
        Ninputs_and_outputs = Ninputs + (1 if Noutputs is None else Noutputs)
        for i in range(len(known_types)):
            if isinstance(known_types[i], type):
                known_types[i] = (known_types[i],) * Ninputs_and_outputs
            elif hasattr(known_types[i], '__iter__')        and \
                 len(known_types[i]) == Ninputs_and_outputs and \
                 all(isinstance(t,type) for t in known_types[i]):
                # already a list of types. we're good
                pass
            else:
                raise NumpysaneError("Each of Ccode_slice_eval.keys() MUST be either a type, or a list of types (one per input, output in order)")

        # {TYPESETS} is _(11, 15, 17, 0) _(13, 15, 17, 1)
        # {TYPESET_MATCHES_ARGLIST} is t0,t1,t2
        # {TYPESETS_NAMES} is
        #                         "   (float32,int32)\n"
        #                         "   (float64,int32)\n"
        TYPESETS = ' '.join( ("_(" + ','.join(tuple(str(np.dtype(t).num) for t in known_types[i]) + (str(i),)) + ')') \
                              for i in range(Ntypesets))
        TYPESET_MATCHES_ARGLIST = ','.join(('t' + str(i)) for i in range(Ninputs_and_outputs))
        def parened_type_list(l, Ninputs):
            r'''Converts list of types to string

            Like "(inputs: float32,int32  outputs: float32, float32)"
'''
            si = 'inputs: '  + ','.join( np.dtype(t).name for t in l[:Ninputs])
            so = 'outputs: ' + ','.join( np.dtype(t).name for t in l[Ninputs:])
            return '(' + si + '   ' + so + ')'
        TYPESETS_NAMES = ' '.join(('"  ' + parened_type_list(s,Ninputs) +'\\n"') \
                                  for s in known_types)

        ARGUMENTS_LIST = ['#define ARGUMENTS(_)']
        for i_arg_input in range(Ninputs):
            ARGUMENTS_LIST.append( '_({})'.format(args_input[i_arg_input]) )

        OUTPUTS_LIST = ['#define OUTPUTS(_)']
        if Noutputs is None:
            OUTPUTS_LIST.append( '_({})'.format("output") )
        else:
            for i_output in range(Noutputs):
                OUTPUTS_LIST.append( '_({}{})'.format("output", i_output) )

        if not hasattr(self, 'function_body'):
            with open(_function_filename, 'r') as f:
                self.function_body = f.read()




        function_template = r'''
static
bool {FUNCTION_NAME}({ARGUMENTS})
{
{FUNCTION_BODY}
}
'''
        if Noutputs is None:
            slice_args = ("output",)+args_input
        else:
            slice_args = tuple("output{}".format(i) for i in range(Noutputs))+args_input

        EXTRA_ARGUMENTS_ARG_DEFINE     = ''
        EXTRA_ARGUMENTS_NAMELIST       = ''
        EXTRA_ARGUMENTS_PARSECODES     = ''
        EXTRA_ARGUMENTS_ARGLIST        = []
        EXTRA_ARGUMENTS_ARGLIST_DEFINE = []

        for _type, n, default_value, parsearg in extra_args:
            EXTRA_ARGUMENTS_ARGLIST_DEFINE.append('const {}* {} __attribute__((unused))'.format(_type, n))
            EXTRA_ARGUMENTS_ARG_DEFINE     += "{} {} = {};\n".format(_type, n, default_value)
            EXTRA_ARGUMENTS_NAMELIST       += '"{}",'.format(n)
            EXTRA_ARGUMENTS_PARSECODES     += '"{}"'.format(parsearg)
            EXTRA_ARGUMENTS_ARGLIST.append('&{}'.format(n))
        if len(extra_args) == 0:
            # no extra args. I need a dummy argument to make the C parser happy,
            # so I add a 0. This is because the template being filled-in is
            # f(....., EXTRA_ARGUMENTS_ARGLIST). A blank EXTRA_ARGUMENTS_ARGLIST
            # would leave a trailing ,
            EXTRA_ARGUMENTS_ARGLIST        = ['0']
            EXTRA_ARGUMENTS_ARGLIST_DEFINE = ['int __dummy __attribute__((unused))']
        EXTRA_ARGUMENTS_SLICE_ARG = ', '.join(EXTRA_ARGUMENTS_ARGLIST_DEFINE)
        EXTRA_ARGUMENTS_ARGLIST   = ', '.join(EXTRA_ARGUMENTS_ARGLIST)

        text = ''
        contiguous_macro_template = r'''
#define _CHECK_CONTIGUOUS__{name}(seterror)                                       \
({                                                                                \
  bool result = true;                                                             \
  int Nelems_slice = 1;                                                           \
  for(int i=-1; i>=-Ndims_slice__{name}; i--)                                     \
  {                                                                               \
      if(strides_slice__{name}[i+Ndims_slice__{name}] != sizeof_element__{name}*Nelems_slice) \
      {                                                                           \
          result = false;                                                         \
          if(seterror)                                                            \
            PyErr_Format(PyExc_RuntimeError,                                      \
                         "Variable '{name}' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
          break;                                                                  \
      }                                                                           \
      Nelems_slice *= dims_slice__{name}[i+Ndims_slice__{name}];                  \
  }                                                                               \
  result;                                                                         \
})
#define CHECK_CONTIGUOUS__{name}()              _CHECK_CONTIGUOUS__{name}(false)
#define CHECK_CONTIGUOUS_AND_SETERROR__{name}() _CHECK_CONTIGUOUS__{name}(true)

'''
        for n in slice_args:
            text += contiguous_macro_template.replace("{name}", n)
        text +=                                                                                \
            '\n' +                                                                             \
            '#define CHECK_CONTIGUOUS_ALL() ' +                                                \
            ' && '.join( "CHECK_CONTIGUOUS__"+n+"()" for n in slice_args) +                    \
            '\n' +                                                                             \
            '#define CHECK_CONTIGUOUS_AND_SETERROR_ALL() ' +                                   \
            ' && '.join( "CHECK_CONTIGUOUS_AND_SETERROR__"+n+"()" for n in slice_args) +       \
            '\n'

        # The user provides two sets of C code that we include verbatim in
        # static functions:
        #
        # - The validation function. Evaluated once to check the input for
        #   validity. This is in addition to the broadcasting shape and type
        #   compatibility checks. Probably the user won't be looking at the data
        #   pointer
        # - The slice function. Evaluated once per broadcasted slice to actually
        #   perform the computation. Probably the user will be looking at just
        #   the _slice data, not the _full data
        #
        # These functions have identical prototypes
        arglist = [ arg for n in slice_args for arg in \
                    ("const int Ndims_full__"          + n + " __attribute__((unused))",
                     "const npy_intp* dims_full__"     + n + " __attribute__((unused))",
                     "const npy_intp* strides_full__"  + n + " __attribute__((unused))",
                     "const int Ndims_slice__"         + n + " __attribute__((unused))",
                     "const npy_intp* dims_slice__"    + n + " __attribute__((unused))",
                     "const npy_intp* strides_slice__" + n + " __attribute__((unused))",
                     "npy_intp sizeof_element__"       + n + " __attribute__((unused))",
                     "void* {DATA_ARGNAME}__"          + n + " __attribute__((unused))")] + \
                  EXTRA_ARGUMENTS_ARGLIST_DEFINE
        arglist_string = '\n  ' + ',\n  '.join(arglist)
        text += \
            _substitute(function_template,
                        FUNCTION_NAME = "__{}__validate".format(name),
                        ARGUMENTS     = _substitute(arglist_string, DATA_ARGNAME="data"),
                        FUNCTION_BODY = "return true;" if Ccode_validate is None else Ccode_validate)

        for i in range(Ntypesets):
            # The evaluation function for one slice
            typeset_indices = tuple(Ccode_slice_eval.keys())
            text += \
                _substitute(function_template,
                            FUNCTION_NAME = slice_functions[i],
                            ARGUMENTS     = _substitute(arglist_string, DATA_ARGNAME="data_slice"),
                            FUNCTION_BODY = Ccode_slice_eval[typeset_indices[i]])



        text += \
            ' \\\n  '.join(ARGUMENTS_LIST) + \
            '\n\n' + \
            ' \\\n  '.join(OUTPUTS_LIST) + \
            '\n\n' + \
            _substitute(self.function_body,
                        FUNCTION_NAME              = name,
                        PROTOTYPE_DIM_DEFS         = PROTOTYPE_DIM_DEFS,
                        UNPACK_OUTPUTS             = UNPACK_OUTPUTS,
                        EXTRA_ARGUMENTS_SLICE_ARG  = EXTRA_ARGUMENTS_SLICE_ARG,
                        EXTRA_ARGUMENTS_ARG_DEFINE = EXTRA_ARGUMENTS_ARG_DEFINE,
                        EXTRA_ARGUMENTS_NAMELIST   = EXTRA_ARGUMENTS_NAMELIST,
                        EXTRA_ARGUMENTS_PARSECODES = EXTRA_ARGUMENTS_PARSECODES,
                        EXTRA_ARGUMENTS_ARGLIST    = EXTRA_ARGUMENTS_ARGLIST,
                        TYPESETS                   = TYPESETS,
                        TYPESET_MATCHES_ARGLIST    = TYPESET_MATCHES_ARGLIST,
                        TYPESETS_NAMES             = TYPESETS_NAMES)

        for n in slice_args:
            text += '#undef _CHECK_CONTIGUOUS__{name}\n'.replace('{name}', n)
            text += '#undef CHECK_CONTIGUOUS__{name}\n'.replace('{name}', n)
            text += '#undef CHECK_CONTIGUOUS_AND_SETERROR__{name}\n'.replace('{name}', n)
        text += '#undef CHECK_CONTIGUOUS_ALL\n'
        text += '#undef CHECK_CONTIGUOUS_AND_SETERROR_ALL\n'

        self.functions.append( (name,
                                _quote(docstring, convert_newlines=True),
                                text) )


    def write(self, file=sys.stdout):
        r'''Write out the generated C code

        DESCRIPTION

        Once we defined all of the wrapper functions in this module by calling
        'function()' for each one, we're ready to write out the generated C
        source that defines this module. write() writes out the C source to
        standard output by default.

        ARGUMENTS

        - file
          The python file object to write the output to. Defaults to standard
          output
        '''

        # Get shellquote from the right place in python2 and python3
        try:
            import pipes
            shellquote = pipes.quote
        except:
            # python3 puts this into a different module
            import shlex
            shellquote = shlex.quote
        print("// THIS IS A GENERATED FILE. DO NOT MODIFY WITH CHANGES YOU WANT TO KEEP",
              file=file)
        print("// Generated on {} with   {}\n\n". \
              format(time.strftime("%Y-%m-%d %H:%M:%S"),
                     ' '.join(shellquote(s) for s in sys.argv)),
              file=file)

        print('#define FUNCTIONS(_) \\', file=file)
        print(' \\\n'.join( '  _({}, "{}")'.format(f[0],f[1]) for f in self.functions),
              file=file)
        print("\n")

        print('///////// {{{{{{{{{ ' + _module_header_filename, file=file)
        print(self.module_header,                               file=file)
        print('///////// }}}}}}}}} ' + _module_header_filename, file=file)

        for f in self.functions:
            print('///////// {{{{{{{{{ ' + _function_filename,  file=file)
            print('///////// for function   ' + f[0],           file=file)
            print(f[2],                                         file=file)
            print('///////// }}}}}}}}} ' + _function_filename,  file=file)
            print('\n',                                         file=file)

        print('///////// {{{{{{{{{ ' + _module_footer_filename, file=file)
        print(self.module_footer,                               file=file)
        print('///////// }}}}}}}}} ' + _module_footer_filename, file=file)
