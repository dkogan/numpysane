static
PyObject* __pywrap__{FUNCTION_NAME}(PyObject* NPY_UNUSED(self),
                                    PyObject* args,
                                    PyObject* kwargs)
{
#define SLICE_ARG(name) \
    void*           data__    ## name, \
    const npy_intp* dims__    ## name, \
    const npy_intp* strides__ ## name,


    typedef bool (slice_function_t)(OUTPUTS(SLICE_ARG) ARGUMENTS(SLICE_ARG) {EXTRA_ARGUMENTS_SLICE_ARG});


    PyObject* __py__result__    = NULL;
    PyObject* __py__output__arg = NULL;

#define ARG_DEFINE(name) PyArrayObject* __py__ ## name = NULL;
    ARGUMENTS(ARG_DEFINE);
    OUTPUTS(  ARG_DEFINE);
    {EXTRA_ARGUMENTS_ARG_DEFINE};

    SET_SIGINT();

#define NAMELIST(name) #name ,
    char* keywords[] = { ARGUMENTS(NAMELIST) "out",
                         {EXTRA_ARGUMENTS_NAMELIST}
                         NULL };
#define PARSECODE(name) "O&"
#define PARSEARG(name) PyArray_Converter, &__py__ ## name,
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     ARGUMENTS(PARSECODE) "|O" {EXTRA_ARGUMENTS_PARSECODES},
                                     keywords,
                                     ARGUMENTS(PARSEARG)
                                     &__py__output__arg,
                                     {EXTRA_ARGUMENTS_ARGLIST},
                                     NULL))
        goto done;

    // parse_dims() is a helper function to evaluate a given list of arguments
    // in respect to a given broadcasting prototype. This function will flag any
    // errors in the dimensionality of the inputs. If no errors are detected, it
    // returns

    //   dims_extra,dims_named

    // where

    //   dims_extra is the outer dimensions of the broadcast
    //   dims_named is the values of the named dimensions


    // First I initialize dims_extra: the array containing the broadcasted
    // slices. Each argument calls for some number of extra dimensions, and the
    // overall array is as large as the biggest one of those

{PROTOTYPE_DIM_DEFS};
{TYPE_DEFS};
{UNPACK_OUTPUTS};

    // At this point each output array is either NULL or a PyObject with a
    // reference. In all cases, Py_XDECREF() should be done at the end. If we
    // have multiple outputs, either the output sequence is already filled-in
    // with valid arrays (if they were passed-in; I just checked in
    // UNPACK_OUTPUTS) or the output tuple is full of blank spaces, and each
    // output is NULL (if I just made a new tuple). In the latter case I'll fill
    // it in later
    //
    // The output argument in __py__output__arg is NULL if we have a single
    // output that's not yet allocated. Otherwise it has a reference also, so it
    // should be PY_XDECREF() at the end. This __py__output__arg is what we
    // should return, unless it's NULL or Py_None. In that case we need to
    // allocate a new array, and return THAT

    {
        // the maximum of Ndims_extra_this for all the arguments
        int Ndims_extra = 0;

        // It's possible for my arguments (and the output) to have fewer dimensions
        // than required by the prototype, and still pass all the dimensionality
        // checks, assuming implied leading dimensions of length 1. For instance I
        // could receive a scalar where a ('n',) dimension is expected, or a ('n',)
        // vector where an ('m','n') array is expected. I thus make a local (and
        // padded) copy of the strides and dims arrays, and use those where needed.
        // Most of the time these will just be copies of the input. The dimension
        // counts and argument counts will be relatively small, so this is only a
        // tiny bit wasteful
#define DECLARE_PROTOTYPE_LEN(name)                      \
        const int PROTOTYPE_LEN_ ## name = (int)sizeof(PROTOTYPE_ ## name)/sizeof(PROTOTYPE_ ## name[0]);

#define DECLARE_DIM_VARS(name)                      \
        int          __ndim__    ## name = PyArray_NDIM(__py__ ## name); \
        if( __ndim__ ## name < PROTOTYPE_LEN_ ## name )                 \
            /* Too few input dimensions. Add dummy dimension of length-1 */ \
            __ndim__ ## name = PROTOTYPE_LEN_ ## name;                  \
        npy_intp __dims__   ## name[__ndim__ ## name];                  \
        npy_intp __strides__## name[__ndim__ ## name];                  \
        {                                                               \
            const npy_intp* dims_orig    = PyArray_DIMS   (__py__ ## name); \
            const npy_intp* strides_orig = PyArray_STRIDES(__py__ ## name); \
            npy_intp        ndim_orig    = PyArray_NDIM   (__py__ ## name); \
            int i_dim = -1;                                             \
            for(; i_dim >= -ndim_orig; i_dim--)                         \
            {                                                           \
                __dims__   ## name[i_dim + __ndim__ ## name] = dims_orig   [i_dim + ndim_orig]; \
                __strides__## name[i_dim + __ndim__ ## name] = strides_orig[i_dim + ndim_orig]; \
            }                                                           \
            for(; i_dim >= -__ndim__ ## name; i_dim--)                  \
            {                                                           \
                /* extra dummy dimensions, as needed */                 \
                __dims__    ## name[i_dim + __ndim__ ## name] = 1;      \
                __strides__ ## name[i_dim + __ndim__ ## name] = 0;      \
            }                                                           \
        }                                                               \
                                                                        \
                                                                        \
        /* guaranteed >= 0 because of the padding */                    \
        int Ndims_extra__ ## name = __ndim__ ## name - PROTOTYPE_LEN_ ## name;

#define UPDATE_NDIMS_EXTRA(name) \
        if(Ndims_extra < Ndims_extra__ ## name) Ndims_extra = Ndims_extra__ ## name;


        ARGUMENTS(DECLARE_PROTOTYPE_LEN);
        ARGUMENTS(DECLARE_DIM_VARS);
        ARGUMENTS(UPDATE_NDIMS_EXTRA);
        OUTPUTS(  DECLARE_PROTOTYPE_LEN);
        // OUTPUTS(DECLARE_DIM_VARS) done later, AFTER we create the output
        // arrays, which may not exist yet


        npy_intp dims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++)
            dims_extra[i] = 1;

        npy_intp dims_named[Ndims_named];
        for(int i=0; i<Ndims_named; i++)
            dims_named[i] = -1;

#define PARSE_DIMS(name)    \
        if(!parse_dim(dims_named, dims_extra,   \
                      Ndims_extra,              \
                                                \
                      #name,                    \
                      Ndims_extra__ ## name,     \
                      PROTOTYPE_ ## name, PROTOTYPE_LEN_ ## name,   \
                      __dims__   ## name, __ndim__       ## name))  \
            goto done;

        ARGUMENTS(PARSE_DIMS);

        // now have dims_extra,dims_named;

        // I process the types. The output arrays may not have been created yet,
        // in which case I just let NULL pass, and ignore the type. I'll make
        // new arrays later, and those will have the right type
        int selected_typenum = NPY_NOTYPE;
        slice_function_t* slice_function;
        for( int i=0; i<Nknown_typenums; i++ )
        {
#define TYPE_MATCHES(name)                                              \
            &&                                                          \
                (__py__ ## name == NULL ||                              \
                 (PyObject*)__py__ ## name == Py_None ||                \
                 known_typenums[i] == PyArray_DESCR(__py__ ## name)->type_num)

            if(true ARGUMENTS(TYPE_MATCHES) OUTPUTS(TYPE_MATCHES))
            {
                // all arguments match this type!
                selected_typenum = known_typenums[i];
                slice_function   = slice_functions[i];
                break;
            }
        }
        if(selected_typenum == NPY_NOTYPE)
        {
#if PY_MAJOR_VERSION == 3

#define INPUT_PERCENT_S(name) "%S,"
#define INPUT_TYPEOBJ(name) ,(((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) ? \
                              (PyObject*)PyArray_DESCR(__py__ ## name)->typeobj : (PyObject*)Py_None)

            PyErr_Format(PyExc_RuntimeError,
                         "ALL inputs and outputs must have consistent type: one of ({KNOWN_TYPES_LIST_STRING}), instead I got (inputs,output) of type ("
                         ARGUMENTS(INPUT_PERCENT_S)
                         OUTPUTS(INPUT_PERCENT_S)
                         ARGUMENTS(INPUT_TYPEOBJ)
                         OUTPUTS(INPUT_TYPEOBJ) );

#else
            ////////// python2 doesn't support %S
            PyErr_Format(PyExc_RuntimeError,
                         "ALL inputs and outputs must have consistent type: one of ({KNOWN_TYPES_LIST_STRING})");
#endif

            goto done;
        }

#define CHECK_DIMS_NAMED_KNOWN(name)                                    \
        for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                     \
            if(PROTOTYPE_ ## name[i] < 0 &&                             \
               dims_named[-PROTOTYPE_ ## name[i]-1] < 0)                \
            {                                                           \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Output prototype has some named dimension not encountered in the input. The pywrap generator shouldn't have let this happen"); \
                goto done;                                              \
            }
        OUTPUTS(CHECK_DIMS_NAMED_KNOWN);


        // The dimensions of each output must be (dims_extra + PROTOTYPE__output__)
#define CHECK_OR_CREATE_OUTPUT(name)                                    \
        {                                                               \
            int Ndims_output = Ndims_extra + PROTOTYPE_LEN_ ## name;    \
            npy_intp dims_output_want[Ndims_output];                    \
            for(int i=0; i<Ndims_extra; i++)                            \
                dims_output_want[i] = dims_extra[i];                    \
            for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                 \
                if(PROTOTYPE_ ## name[i] < 0 )                          \
                    dims_output_want[i+Ndims_extra] = dims_named[-PROTOTYPE_ ## name[i]-1]; \
                    /* I know the dims_named is defined. Checked it above */ \
                else                                                    \
                    dims_output_want[i+Ndims_extra] = PROTOTYPE_ ## name[i]; \
                                                                        \
            if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
            {                                                           \
                /* An output array was given. I make sure it has the correct dimensions. I */ \
                /* allow leading dimensions of length 1. I compare from the end, */ \
                /* comparing with 1 if either list runs out             */ \
                int Ndims_to_compare1 = PyArray_NDIM(__py__ ## name);   \
                int Ndims_to_compare2 = Ndims_output;                   \
                int Ndims_to_compare = Ndims_to_compare1;               \
                if(Ndims_to_compare < Ndims_to_compare2)                \
                    Ndims_to_compare = Ndims_to_compare2;               \
                                                                        \
                for( int i_dim=-1;                                      \
                     i_dim >= -Ndims_to_compare;                        \
                     i_dim--)                                           \
                {                                                       \
                    int i_dim_var       = i_dim + PyArray_NDIM(__py__ ## name); \
                    int i_dim_output    = i_dim + Ndims_output;         \
                    int dim_var         = i_dim_var    >= 0 ? PyArray_DIMS(__py__ ## name)[i_dim_var   ] : 1; \
                    int dim_output_want = i_dim_output >= 0 ? dims_output_want            [i_dim_output] : 1; \
                    if(dim_var != dim_output_want)                      \
                    {                                                   \
                        PyErr_Format(PyExc_RuntimeError,                \
                                     "Given output array dimension %d mismatch. Expected %d but got %d", \
                                     i_dim,                             \
                                     dim_output_want, dim_var);         \
                        goto done;                                      \
                    }                                                   \
                }                                                       \
            }                                                           \
            else                                                        \
            {                                                           \
                /* No output array available. Make one                  */ \
                __py__ ## name = (PyArrayObject*)PyArray_SimpleNew(Ndims_output, dims_output_want, selected_typenum); \
                if(__py__ ## name == NULL)                              \
                {                                                       \
                    /* Error already set. I simply exit                 */ \
                    goto done;                                          \
                }                                                       \
                                                                        \
                if(populate_output_tuple__i >= 0)                       \
                {                                                       \
                    PyTuple_SET_ITEM(__py__output__arg,                 \
                                     populate_output_tuple__i,          \
                                     (PyObject*)__py__ ## name);        \
                    populate_output_tuple__i++;                         \
                    Py_INCREF(__py__ ## name);                          \
                }                                                       \
                else if(__py__output__arg == NULL)                      \
                {                                                       \
                    /* one output, no output given */                   \
                    __py__output__arg = (PyObject*)__py__ ## name;      \
                    Py_INCREF(__py__output__arg);                       \
                }                                                       \
            }                                                           \
        }
        OUTPUTS(CHECK_OR_CREATE_OUTPUT);

        // Each output variable is now an allocated array, and each one has a
        // reference. The argument __py__output__arg ALSO has a reference

        // Now that the output exists, I can grab the dimensions
        OUTPUTS(DECLARE_DIM_VARS);

#define ARGLIST_VALIDATION(name)                \
        __ndim__      ## name ,                 \
        Ndims_extra__ ## name ,                 \
        __dims__      ## name ,                 \
        __strides__   ## name ,                 \
        PyArray_ITEMSIZE(__py__ ## name),


        if( ! __{FUNCTION_NAME}__validate(OUTPUTS(  ARGLIST_VALIDATION)
                                          ARGUMENTS(ARGLIST_VALIDATION)
                                          {EXTRA_ARGUMENTS_ARGLIST}) )
        {
            PyErr_SetString(PyExc_RuntimeError, "User-provided validation failed!");
            goto done;
        }

        // if no broadcasting involved, just call the function
        if(Ndims_extra == 0)
        {
#define DEFINE_SLICE(name) char* slice_ ## name = PyArray_DATA(__py__ ## name);
            ARGUMENTS(DEFINE_SLICE);
            OUTPUTS(  DEFINE_SLICE);

#define ARGLIST_SLICE(name) \
  (void*)slice_ ## name,                         \
  &__dims__     ## name[ Ndims_extra__ ## name ], \
  &__strides__  ## name[ Ndims_extra__ ## name ],

            if( ! slice_function( OUTPUTS(  ARGLIST_SLICE)
                                  ARGUMENTS(ARGLIST_SLICE)
                                  {EXTRA_ARGUMENTS_ARGLIST})
                )
            {
                PyErr_Format(PyExc_RuntimeError, "__{FUNCTION_NAME}__slice failed!");
            }
            else
                __py__result__ = (PyObject*)__py__output__arg;
            goto done;
        }

#if 0
        // How many elements (not bytes) to advance for each broadcasted dimension.
        // Takes into account the length-1 slieces (implicit and explicit)
        int stride_extra_elements_a[Ndims_extra];
        int stride_extra_elements_b[Ndims_extra];
        for(int idim_extra=0; idim_extra<Ndims_extra; idim_extra++)
        {
            int idim;

            idim = idim_extra + Ndims_extra_a - Ndims_extra;
            if(idim>=0 && __dims__a[idim] != 1)
                stride_extra_elements_a[idim_extra] = __strides__a[idim] / sizeof(double);
            else
                stride_extra_elements_a[idim_extra] = 0;

            idim = idim_extra + Ndims_extra_b - Ndims_extra;
            if(idim>=0 && __dims__b[idim] != 1)
                stride_extra_elements_b[idim_extra] = __strides__b[idim] / sizeof(double);
            else
                stride_extra_elements_b[idim_extra] = 0;
        }
#endif

        // I checked all the dimensions and aligned everything. I have my
        // to-broadcast dimension counts.


        // Iterate through all the broadcasting output, and gather the results
        int idims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++) idims_extra[i] = 0;
        bool next(int* idims, const npy_intp* Ndims, int N)
        {
            for(int i = N-1; i>=0; i--)
            {
                if(++idims[i] < Ndims[i])
                    return true;
                idims[i] = 0;
            }
            return false;
        }
        do
        {
            ARGUMENTS(DEFINE_SLICE);
            OUTPUTS(  DEFINE_SLICE);

            for( int i_dim=-1;
                 i_dim >= -Ndims_extra;
                 i_dim--)
            {

#define ADVANCE_SLICE(name)                         \
                if(i_dim + Ndims_extra__ ## name >= 0 &&                 \
                   __dims__ ## name[i_dim + Ndims_extra__ ## name] != 1) \
                    slice_ ## name += idims_extra[i_dim + Ndims_extra]*__strides__ ## name[i_dim + Ndims_extra__ ## name];

                ARGUMENTS(ADVANCE_SLICE);
                OUTPUTS(  ADVANCE_SLICE);
            }

            if( ! slice_function( OUTPUTS(  ARGLIST_SLICE)
                                  ARGUMENTS(ARGLIST_SLICE)
                                  {EXTRA_ARGUMENTS_ARGLIST})
                )
            {
                PyErr_Format(PyExc_RuntimeError,
                             "__{FUNCTION_NAME}__slice failed!");
                goto done;
            }

        } while(next(idims_extra, dims_extra, Ndims_extra));

        __py__result__ = (PyObject*)__py__output__arg;
    }
 done:

    // I free the arguments (I'm done with them) and the outputs (I'm done with
    // each individual one; the thing I'm returning has its own reference)
#define FREE_PYARRAY(name) Py_XDECREF(__py__ ## name);
    ARGUMENTS(FREE_PYARRAY);
    OUTPUTS(  FREE_PYARRAY);

    if(__py__result__ == NULL)
    {
        // An error occurred. I'm not returning an output, so release that too
        Py_XDECREF(__py__output__arg);
    }

    RESET_SIGINT();
    return __py__result__;
}

#undef ARG_DEFINE
#undef PULL_OUT_OUTPUT_ARRAYS
#undef NAMELIST
#undef PARSECODE
#undef PARSEARG
#undef DECLARE_DIM_VARS
#undef UPDATE_NDIMS_EXTRA
#undef PARSE_DIMS
#undef SLICE_ARG
#undef TYPE_MATCHES
#undef INPUT_PERCENT_S
#undef INPUT_TYPEOBJ
#undef DEFINE_SLICE
#undef ARGLIST_SLICE
#undef ADVANCE_SLICE
#undef FREE_PYARRAY
#undef DECLARE_PROTOTYPE_LEN
#undef CHECK_DIMS_NAMED_KNOWN
#undef CHECK_OR_CREATE_OUTPUT
#undef ARGUMENTS
#undef OUTPUTS
