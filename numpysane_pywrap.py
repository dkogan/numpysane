import sys

def _substitute(s, convert_newlines=False, **kwargs):
    r'''format() with specific semantics

    - {xxx} substitutions founc in kwargs are made
    - {xxx} expressions not found in kwargs are left as is
    - {{ }} escaping is not respected: any '{xxx}' is replaced
    - \ and " and \n are handled for C strings
    - if convert_newlines: newlines are converted to \n (useful for C strings).
      Otherwise they're left alone (useful for C code)

    '''
    def quote(s):
        r'''Quote string for inclusion in C code

        There should be a library for this. Hopefuly this is correct.

        '''
        s = s.replace('\\', '\\\\')     # Pass all \ through verbatim
        if convert_newlines:
           s = s.replace('\n', '\\n')  # All newlines -> \n
        s = s.replace('"',  '\\"')      # Quote all "
        return s

    for k in kwargs.keys():
        v = kwargs[k]
        if isinstance(v, str):
            v = quote(v)
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

        with open( 'pywrap_module_header.c', 'r') as f:
            self.module_header = f.read() + "\n" + HEADER + "\n"

        with open( 'pywrap_module_footer_generic.c', 'r') as f:
            self.module_footer = _substitute(f.read(),
                                             MODULE_NAME      = MODULE_NAME,
                                             MODULE_DOCSTRING = MODULE_DOCSTRING,
                                             convert_newlines = True)

        self.functions = []


    def function(self,
                 FUNCTION_NAME,
                 FUNCTION_DOCSTRING,
                 prototype_input,
                 prototype_output,
                 __FUNCTION__slice_code,
                 *argnames):
        r'''Add a function to the python module we're creating

        SYNOPSIS

        If we're wrapping a simple inner product you can do this:

        function( "inner",
                  "npsp-wrapped inner product",
                  (('n',), ('n',)),
                  (),

                  r"""
                  output.data[0] = 0.0;
                  for(int i=0; i<a.shape[0]; i++)
                      output.data[0] += \
                           *( (double*)(i*a.strides[0]+(char*)(a.data))) *
                           *( (double*)(i*b.strides[0]+(char*)(b.data)));
                  return true;
                  """,

                  "a", "b'")

        Here we generate code to wrap a chunk of C code. The main chunk of
        user-supplied glue code is passed-in with __FUNCTION__slice_code. This
        function is given all the input and output buffers, and it's the job of
        the glue code to read and write them.


        ARGUMENTS

        - FUNCTION_NAME
          The name of this function

        - FUNCTION_DOCSTRING
          The docstring for this function

        - prototype_input
          An iterable defining the shapes of the inputs. Each element describes
          the trailing shape of each argument. Each element of this shape
          definition is either an integer (if this dimension of the broadcasted
          slice MUST have this size) or a string (naming this dimension; any
          size is allowed, but the sizes of all dimensions with this name must
          match)

        - prootype_output
          A single shape definition. Similar to prototype_input, but there's
          just one. Named dimensions in prototype_input must match the ones here

        - __FUNCTION__slice_code

          C code that will be included verbatim into the python-wrapping code.
          If we're wrapping a function called FUNCTION, this string is the
          contents of the wrapper function:

            typedef struct
            {
                double* data;
                const npy_intp* strides;
                const npy_intp* shape;
            } nps_slice_t;
            bool __FUNCTION__slice( nps_slice_t output,
                                    nps_slice_t a,
                                    nps_slice_t b )
            {
               // THE PASSED-IN STRING GOES HERE
            }

          This function is called for each broadcasting slice. The number of
          arguments and their names are generated from the "prototype_input" and
          "argnames" arguments here. The strides and shape define the memory
          layout of the data in memory for this slice. The 'shape' is only
          knowable at runtime because of named dimensions. Return true on
          soccess.

        - *argnames
          The names of the arguments. Must have the same number of elements as
          prototype_input

        '''

        if len(prototype_input) != len(argnames):
            raise Exception("Input prototype says we have {} arguments, but names for {} were given. These must match". \
                            format(len(prototype_input), len(argnames)))


        function_slice_template = r'''
static
bool __{FUNCTION_NAME}__slice(nps_slice_t output{SLICE_DEFINITIONS})
{
{__FUNCTION__slice_code}
}
'''

        text_function_slice = \
          _substitute(function_slice_template,
                      FUNCTION_NAME          = FUNCTION_NAME,
                      SLICE_DEFINITIONS      = ''.join([", nps_slice_t " + n for n in argnames]),
                      __FUNCTION__slice_code = __FUNCTION__slice_code)
        # I enumerate each named dimension, starting from -1, and counting DOWN
        named_dims = {}
        for shape in prototype_input + (prototype_output,):
            for dim in shape:
                if not isinstance(dim,int) and \
                   dim not in named_dims:
                    named_dims[dim] = -1-len(named_dims)

        def expand_prototype(shape):
            r'''Produces a shape string for each argument

            This is the dimensions passed-into this function except, named
            dimensions are consolidated, and set to -1,-2,..., and the whole
            thing is stringified and joined

            '''

            shape = [ dim if isinstance(dim,int) else named_dims[dim] for dim in shape ]
            return ','.join(str(dim) for dim in shape)

        PROTOTYPE_DIM_DEFS = ''
        for i_arg_input in range(len(argnames)):
            PROTOTYPE_DIM_DEFS += "    const npy_intp PROTOTYPE_{}[{}] = {{{}}};\n". \
                format(argnames[i_arg_input],
                       len(prototype_input[i_arg_input]),
                       expand_prototype(prototype_input[i_arg_input]));
        PROTOTYPE_DIM_DEFS += "    /* Not const. updating the named dimensions in-place */\n"
        PROTOTYPE_DIM_DEFS += "    npy_intp PROTOTYPE__output__[{}] = {{{}}};\n". \
            format(len(prototype_output),
                   expand_prototype(prototype_output));
        PROTOTYPE_DIM_DEFS += "    int Ndims_named = {};\n". \
            format(len(named_dims))


        ARGUMENTS_LIST = ['#define ARGUMENTS(_)']
        for i_arg_input in range(len(argnames)):
            ARGUMENTS_LIST.append( '_({}, NPY_DOUBLE, xxx )'.format(argnames[i_arg_input]) )

        if not hasattr(self, 'function_template'):
            with open('pywrap_function_generic.c', 'r') as f:
                self.function_template = f.read()

        text = \
            text_function_slice + \
            ' \\\n  '.join(ARGUMENTS_LIST) + \
            '\n\n' + \
            _substitute(self.function_template,
                        FUNCTION_NAME      = FUNCTION_NAME,
                        PROTOTYPE_DIM_DEFS = PROTOTYPE_DIM_DEFS)
        self.functions.append( (FUNCTION_NAME,
                                _substitute(FUNCTION_DOCSTRING, convert_newlines=True),
                                text) )


    def write(self, file=sys.stdout):
        r'''Write out the module definition to stdout

        Call this after the constructor and all the function() calls

        '''
        print(self.module_header, file=file)

        for f in self.functions:
            print("// ==================== {} ===================".format(f[0]), file=file)
            print(f[2], file=file)
            print('\n', file=file)

        print('#define FUNCTIONS(_) \\', file=file)
        print(' \\\n'.join( '  _({}, "{}")'.format(f[0],f[1]) for f in self.functions),
              file=file)
        print("\n")

        print(self.module_footer, file=file)


r'''

different types (function docstring hardcodes double too)

contiguities

relationships between dimension sizes

VALIDATE

error handling: no more asserts

multiple arguments? should everything be broadcasting? kwargs?

'''
