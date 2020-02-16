import sys
import time
import numpy as np
from numpysane import NumpysaneError
import os

# Technically I'm supported to use some "resource extractor" something to
# unbreak setuptools. But I'm instead assuming that this was installed via
# Debian or by using the eager_resources tag in setup(). This allows files to
# remain files, and to appear in a "normal" directory, where this script can
# grab them and use them
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
    def __init__(self, MODULE_NAME, MODULE_DOCSTRING, HEADER=''):
        r'''Initialize the python-wrapper-generator

        Arguments:

        - MODULE_NAME
          The name of the python module we're creating

        - MODULE_DOCSTRING
          The docstring for this module

        - HEADER

          C code to include verbatim. Any #includes or utility functions can go
          here

        '''

        with open( _module_header_filename, 'r') as f:
            self.module_header = f.read() + "\n" + HEADER + "\n"

        with open( _module_footer_filename, 'r') as f:
            self.module_footer = _substitute(f.read(),
                                             MODULE_NAME      = MODULE_NAME,
                                             MODULE_DOCSTRING = MODULE_DOCSTRING,
                                             convert_newlines = True)

        self.functions = []


    def function(self,
                 FUNCTION_NAME,
                 FUNCTION_DOCSTRING,
                 args_input,
                 prototype_input,
                 prototype_output,
                 FUNCTION__slice_code,
                 VALIDATE_code = None,
                 extra_args    = ()):
        r'''Add a function to the python module we're creating

        SYNOPSIS

        If we're wrapping a simple inner product you can do this:

        function( "inner",
                  "Inner-product pywrapped with npsp",

                  prototype_input  = (('n',), ('n',)),
                  prototype_output = (),

                  FUNCTION__slice_code = r"""
                  output.data[0] = inner(a.data,
                                         b.data,
                                         a.strides[0],
                                         b.strides[0],
                                         a.shape[0]);
                  return true;
                  """,

                  "a", "b'")

        Here we generate code to wrap a chunk of C code. The main chunk of
        user-supplied glue code is passed-in with FUNCTION__slice_code. This
        function is given all the input and output buffers, and it's the job of
        the glue code to read and write them.


        ARGUMENTS

        - FUNCTION_NAME
          The name of this function

        - FUNCTION_DOCSTRING
          The docstring for this function

        - args_input
          The names of the arguments. Must have the same number of elements as
          prototype_input

        - prototype_input
          An iterable defining the shapes of the inputs. Each element describes
          the trailing shape of each argument. Each element of this shape
          definition is either an integer (if this dimension of the broadcasted
          slice MUST have this size) or a string (naming this dimension; any
          size is allowed, but the sizes of all dimensions with this name must
          match)

        - prototype_output
          A single shape definition. Similar to prototype_input, but there's
          just one. Named dimensions in prototype_input must match the ones here

        - FUNCTION__slice_code

          This is a dict from numpy type objects to code snippets that form the
          body of a slice_function_t function. This is C code that will be
          included verbatim into the python-wrapping code. For instance, if
          we're wrapping a function called FUNCTION that works on 64-bit
          floating-point values, here we specify the way we call this function.

            typedef struct
            {
                void*           data;
                const npy_intp* strides;
                const npy_intp* shape;
            } nps_slice_t;
            bool __FUNCTION__float64_slice( nps_slice_t output,
                                            nps_slice_t a,
                                            nps_slice_t b )
            {
               // THE PASSED-IN STRING FOR THE 'float' KEY ENDS UP HERE
               ...
               ...
               // This string eventually contains a FUNCTION() call
               FUNCTION(...);
            }

          This function is called for each broadcasting slice. The number of
          arguments and their names are generated from the "prototype_input" and
          "args_input" arguments. The strides and shape define the memory layout
          of the data in memory for this slice. The 'shape' is only knowable at
          runtime because of named dimensions. The inner slice function returns
          true on success.

          Currently any number of data types are supported, but ALL of the
          inputs AND the output MUST share a single, consistent type. No
          implicit type conversions are performed, but the system does check
          for, and report type mismatches

        - VALIDATE_code

          C code that will be included verbatim into the python-wrapping code.
          Any special variable-validation code can be specified here.
          Dimensionality checks against the prototypes are generated
          automatically, but things like stride or type checking can be done
          here.

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
                         "Given output expected to be a sequence of length %d, but a non-sequence was given",
                         {Noutputs});
            goto done;
        }
        if( PySequence_Size(__py__output__arg) != {Noutputs} )
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Given output expected to be a sequence of length %d, but a sequence of length %d was given",
                         {Noutputs}, PySequence_Size(__py__output__arg));
            goto done;
        }

#define PULL_OUT_OUTPUT_ARRAYS(name)                                                                         \
        __py__ ## name = (PyArrayObject*)PySequence_GetItem(__py__output__arg, i++);                         \
        if(__py__ ## name == NULL || !PyArray_Check(__py__ ## name))                                         \
        {                                                                                                    \
            PyErr_SetString(PyExc_RuntimeError,                                                              \
                            "Given output array MUST contain pre-allocated arrays. " #name " is not an array"); \
            goto done;                                                                                       \
        }
        int i=0;
        OUTPUTS(PULL_OUT_OUTPUT_ARRAYS)
    }
'''.replace('{Noutputs}', str(Noutputs))


        Ntypesets = len(FUNCTION__slice_code)
        slice_functions = [ "__{}__{}__slice".format(FUNCTION_NAME,i) for i in range(Ntypesets)]
        # The keys of FUNCTION__slice_code are either:

        # - a type: all inputs, outputs MUST have this type
        # - a list of types: the types in this list correspond to the inputs and
        #   outputs of the call, in that order.
        #
        # I convert the known-type list to the one-type-per-element form for
        # consistent processing
        known_types = list(FUNCTION__slice_code.keys())
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
                raise NumpysaneError("Each of FUNCTION__slice_code.keys() MUST be either a type, or a list of types (one per input, output in order)")

        # {TYPESETS} is _(11, 15, 17, 0) _(13, 15, 17, 1)
        # {TYPESET_MATCHES_ARGLIST} is t0,t1,t2
        # {TYPESETS_NAMES} is
        #                         "   (float32,int32)\n"
        #                         "   (float64,int32)\n"
        TYPESETS = ' '.join( ("_(" + ','.join(tuple(str(np.dtype(t).num) for t in known_types[i]) + (str(i),)) + ')') \
                              for i in range(Ntypesets))
        TYPESET_MATCHES_ARGLIST = ','.join(("t" + str(i)) for i in range(Ninputs_and_outputs))
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

        for _type, name, default_value, parsearg in extra_args:
            EXTRA_ARGUMENTS_ARGLIST_DEFINE.append('const {}* {} __attribute__((unused))'.format(_type, name))
            EXTRA_ARGUMENTS_ARG_DEFINE     += "{} {} = {};\n".format(_type, name, default_value)
            EXTRA_ARGUMENTS_NAMELIST       += '"{}",'.format(name)
            EXTRA_ARGUMENTS_PARSECODES     += '"{}"'.format(parsearg)
            EXTRA_ARGUMENTS_ARGLIST.append('&{}'.format(name))
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
#define CHECK_CONTIGUOUS__{name}()                                                \
({                                                                                \
  bool result = true;                                                             \
  int Nelems_slice = 1;                                                           \
  const int Ndims_slice = Ndims__{name} - Ndims_extra__{name};                    \
  for(int i=-1; i>=-Ndims_slice; i--)                                             \
  {                                                                               \
      if(strides__{name}[i+Ndims__{name}] != sizeof_element__{name}*Nelems_slice) \
      {                                                                           \
          result = false;                                                         \
          break;                                                                  \
      }                                                                           \
      Nelems_slice *= dims__{name}[i+Ndims__{name}];                              \
  }                                                                               \
  result;                                                                         \
})

#define CHECK_CONTIGUOUS_AND_SETERROR__{name}()                                   \
({                                                                                \
  bool result = true;                                                             \
  int Nelems_slice = 1;                                                           \
  const int Ndims_slice = Ndims__{name} - Ndims_extra__{name};                    \
  for(int i=-1; i>=-Ndims_slice; i--)                                             \
  {                                                                               \
      if(strides__{name}[i+Ndims__{name}] != sizeof_element__{name}*Nelems_slice) \
      {                                                                           \
          result = false;                                                         \
          PyErr_Format(PyExc_RuntimeError,                                        \
                       "Variable '{name}' must be contiguous in memory, and it isn't in (at least) dimension %d", i); \
          break;                                                                  \
      }                                                                           \
      Nelems_slice *= dims__{name}[i+Ndims__{name}];                              \
  }                                                                               \
  result;                                                                         \
})

'''
        for n in slice_args:
            text += contiguous_macro_template.replace("{name}", n)
        text +=                                                                                \
            '\n' +                                                                             \
            '#define CHECK_CONTIGUOUS_ALL() ' +                                                \
            ' && '.join( "CHECK_CONTIGUOUS__"+name+"()" for name in slice_args) +              \
            '\n' +                                                                             \
            '#define CHECK_CONTIGUOUS_AND_SETERROR_ALL() ' +                                   \
            ' && '.join( "CHECK_CONTIGUOUS_AND_SETERROR__"+name+"()" for name in slice_args) + \
            '\n'

        # The validation function. Evaluated once. For each argument and
        # output, we pass in the dimensions and the strides (we do NOT pass
        # in data pointers)
        validation_arglist = [ arg for n in slice_args for arg in \
                               ("const int Ndims__"         + n + " __attribute__((unused))",
                                "const int Ndims_extra__"   + n + " __attribute__((unused))",
                                "const npy_intp* dims__"    + n + " __attribute__((unused))",
                                "const npy_intp* strides__" + n + " __attribute__((unused))",
                                "npy_intp sizeof_element__" + n + " __attribute__((unused))") ] + \
                             EXTRA_ARGUMENTS_ARGLIST_DEFINE
        VALIDATION_ARGUMENTS = '\n  ' + ',\n  '.join(validation_arglist)
        text += \
            _substitute(function_template,
                        FUNCTION_NAME = "__{}__validate".format(FUNCTION_NAME),
                        ARGUMENTS     = VALIDATION_ARGUMENTS,
                        FUNCTION_BODY = "return true;" if VALIDATE_code is None else VALIDATE_code)
        for n in slice_args:
            text += '#undef CHECK_CONTIGUOUS__{name}\n'.replace('{name}', n)
            text += '#undef CHECK_CONTIGUOUS_AND_SETERROR__{name}\n'.replace('{name}', n)
        text += '#undef CHECK_CONTIGUOUS_ALL\n'
        text += '#undef CHECK_CONTIGUOUS_AND_SETERROR_ALL\n'

        slice_arglist = [arg for n in slice_args
                         for arg in
                         ("void* data__"              + n + " __attribute__((unused))",
                          "const npy_intp* dims__"    + n + " __attribute__((unused))",
                          "const npy_intp* strides__" + n + " __attribute__((unused))")] + \
                          EXTRA_ARGUMENTS_ARGLIST_DEFINE

        SLICE_ARGUMENTS = '\n  ' + ',\n  '.join(slice_arglist)

        for i in range(Ntypesets):
            # The evaluation function for one slice
            typeset_indices = tuple(FUNCTION__slice_code.keys())
            text += \
                _substitute(function_template,
                            FUNCTION_NAME = slice_functions[i],
                            ARGUMENTS     = SLICE_ARGUMENTS,
                            FUNCTION_BODY = FUNCTION__slice_code[typeset_indices[i]])



        text += \
            ' \\\n  '.join(ARGUMENTS_LIST) + \
            '\n\n' + \
            ' \\\n  '.join(OUTPUTS_LIST) + \
            '\n\n' + \
            _substitute(self.function_body,
                        FUNCTION_NAME              = FUNCTION_NAME,
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


        self.functions.append( (FUNCTION_NAME,
                                _quote(FUNCTION_DOCSTRING, convert_newlines=True),
                                text) )


    def write(self, file=sys.stdout):
        r'''Write out the module definition to stdout

        Call this after the constructor and all the function() calls

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
