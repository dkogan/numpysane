static
PyObject* __pywrap__{FUNCTION_NAME}(PyObject* NPY_UNUSED(self),
                                    PyObject* args,
                                    PyObject* kwargs)
{
#define SLICE_ARG(name)                         \
                                                \
    const int       Ndims_full__     ## name,   \
    const npy_intp* dims_full__      ## name,   \
    const npy_intp* strides_full__   ## name,   \
                                                \
    const int       Ndims_slice__    ## name,   \
    const npy_intp* dims_slice__     ## name,   \
    const npy_intp* strides_slice__  ## name,   \
                                                \
    npy_intp        sizeof_element__ ## name,   \
    void*           data_slice__     ## name,


    // The cookie we compute BEFORE computing any slices. This is available to
    // the slice-computation function to do whatever they please. I initialize
    // the cookie to all-zeros. If any cleanup is needed, the COOKIE_CLEANUP
    // code at the end of this function should include an "inited" flag in the
    // cookie in order to know whether the cookie was inited in the first place,
    // and whether any cleanup is actually required
    __{FUNCTION_NAME}__cookie_t  _cookie = {};
    // I'd like to access the "cookie" here in a way identical to how I access
    // it inside the functions, so it must be a cookie_t* cookie
    __{FUNCTION_NAME}__cookie_t* cookie = &_cookie;

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
                                     {EXTRA_ARGUMENTS_ARGLIST_PARSE_PYARG}
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
        // I process the types. The output arrays may not have been created yet,
        // in which case I just let NULL pass, and ignore the type. I'll make
        // new arrays later, and those will have the right type
#define DEFINE_OUTPUT_TYPENUM(name) int selected_typenum__ ## name;
        OUTPUTS(DEFINE_OUTPUT_TYPENUM);
#undef DEFINE_OUTPUT_TYPENUM
        slice_function_t* slice_function;

#define TYPE_MATCHES_ARGLIST(name) int typenum__ ## name,
        bool type_matches(ARGUMENTS(TYPE_MATCHES_ARGLIST)
                          OUTPUTS(  TYPE_MATCHES_ARGLIST)
                          slice_function_t* f)
        {

#define SET_SELECTED_TYPENUM_OUTPUT(name) selected_typenum__ ## name = typenum__ ## name;
#define TYPE_MATCHES(name)                                              \
            && ( __py__ ## name == NULL ||                              \
              (PyObject*)__py__ ## name == Py_None ||                   \
              PyArray_DESCR(__py__ ## name)->type_num == typenum__ ## name )

            if(true ARGUMENTS(TYPE_MATCHES) OUTPUTS(TYPE_MATCHES))
            {
                /* all arguments match this typeset! */
                slice_function = f;
                OUTPUTS(SET_SELECTED_TYPENUM_OUTPUT);
                return true;
            }
            return false;
        }
#undef SET_SELECTED_TYPENUM_OUTPUT
#undef TYPE_MATCHES
#undef TYPE_MATCHES_ARGLIST


#define TYPESETS(_) \
        {TYPESETS}
#define TYPESET_MATCHES({TYPESET_MATCHES_ARGLIST}, i)                   \
        else if( type_matches({TYPESET_MATCHES_ARGLIST},                \
                              __{FUNCTION_NAME}__ ## i ## __slice) )    \
        {                                                               \
            /* matched. type_matches() did all the work. */             \
        }

        if(0) ;
        TYPESETS(TYPESET_MATCHES)
        else
        {

#if PY_MAJOR_VERSION == 3

#define INPUT_PERCENT_S(name) "%S,"
#define INPUT_TYPEOBJ(name) ,(((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) ? \
                              (PyObject*)PyArray_DESCR(__py__ ## name)->typeobj : (PyObject*)Py_None)

            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         {TYPESETS_NAMES}
                         "instead I got types (inputs: " ARGUMENTS(INPUT_PERCENT_S) ")"
                         "   outputs: (" OUTPUTS(INPUT_PERCENT_S) ")\n"
                         "None in an output is not an error: a new array of the right type will be created"
                         ARGUMENTS(INPUT_TYPEOBJ)
                         OUTPUTS(INPUT_TYPEOBJ) );

#else
            ////////// python2 doesn't support %S
            PyErr_Format(PyExc_RuntimeError,
                         "The set of input and output types must correspond to one of these sets:\n"
                         {TYPESETS_NAMES});
#endif

            goto done;
        }
#undef TYPESETS
#undef TYPESET_MATCHES


        // Now deal with dimensionality

        // It's possible for my arguments (and the output) to have fewer
        // dimensions than required by the prototype, and still pass all the
        // dimensionality checks, assuming implied leading dimensions of length
        // 1. For instance I could receive a scalar where a ('n',) dimension is
        // expected, or a ('n',) vector where an ('m','n') array is expected. I
        // initially handle this with Ndims_extra<0 for those arguments and then
        // later, I make copies with actual "1" values in place. I do that because:
        //
        // 1. I want to support the above-described case where implicit leading
        //    length-1 dimensions are used
        //
        // 2. I want to support new named-dimensions in the outputs, pulled from
        //    the in-place arrays
        //
        // #2 requires partial processing of the outputs before they're all
        // guaranteed to exist. So I can't allocate temporary __dims__##name and
        // __strides__##name arrays on the stack: I don't know how big they are
        // yet. But I need explicit dimensions in memory to pass to the
        // validation and slice callbacks. So I do it implicitly first, and then
        // explicitly

        // the maximum of Ndims_extra_this for all the arguments. Each one COULD
        // be <0 but Ndims_extra is capped at the bottom at 0
        int Ndims_extra = 0;

#define DECLARE_DIM_VARS(name)                                          \
        const int       PROTOTYPE_LEN_ ## name = (int)sizeof(PROTOTYPE_ ## name)/sizeof(PROTOTYPE_ ## name[0]); \
        int             __ndim__       ## name = -1;                    \
        const npy_intp* __dims__       ## name = NULL;                  \
        const npy_intp* __strides__    ## name = NULL;                  \
        /* May be <0 */                                                 \
        int             Ndims_extra__  ## name = -1;

#define DEFINE_DIM_VARS(name)                                           \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            __ndim__    ## name = PyArray_NDIM   (__py__ ## name);      \
            __dims__    ## name = PyArray_DIMS   (__py__ ## name);      \
            __strides__ ## name = PyArray_STRIDES(__py__ ## name);      \
            /* May be <0 */                                             \
            Ndims_extra__ ## name = __ndim__ ## name - PROTOTYPE_LEN_ ## name; \
            if(Ndims_extra < Ndims_extra__ ## name)                     \
                Ndims_extra = Ndims_extra__ ## name;                    \
        }



        ARGUMENTS(DECLARE_DIM_VARS);
        ARGUMENTS(DEFINE_DIM_VARS);
        OUTPUTS(  DECLARE_DIM_VARS);
        OUTPUTS(  DEFINE_DIM_VARS);
        // Any outputs that are given are processed here. Outputs that are NOT
        // given are skipped for now. I'll create them later, and do the
        // necessary updates and checks later by expanding DEFINE_DIM_VARS later

        npy_intp dims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++)
            dims_extra[i] = 1;

        npy_intp dims_named[Ndims_named];
        for(int i=0; i<Ndims_named; i++)
            dims_named[i] = -1;

#define PARSE_DIMS(name)                                                \
        if((PyObject*)__py__ ## name != Py_None && __py__ ## name != NULL) \
        {                                                               \
            if(!parse_dim_for_one_arg(dims_named, dims_extra,           \
                                      Ndims_extra,                      \
                                                                        \
                                      #name,                            \
                                      Ndims_extra__ ## name,            \
                                      PROTOTYPE_ ## name, PROTOTYPE_LEN_ ## name, \
                                      __dims__   ## name, __ndim__       ## name, \
                                      is_output))                       \
                goto done;                                              \
        }

        bool is_output = false;
        ARGUMENTS(PARSE_DIMS);
        is_output = true;
        OUTPUTS(  PARSE_DIMS);


        // now have dims_extra,dims_named;


#define CHECK_DIMS_NAMED_KNOWN(name)                                    \
        for(int i=0; i<PROTOTYPE_LEN_ ## name; i++)                     \
            if(PROTOTYPE_ ## name[i] < 0 &&                             \
               dims_named[-PROTOTYPE_ ## name[i]-1] < 0)                \
            {                                                           \
                PyErr_Format(PyExc_RuntimeError,                        \
                             "Output prototype " #name " dimension %d is named, but not defined by the input. You MUST pass in-place output array(s) to define these dimensions", \
                             i);                                         \
                goto done;                                              \
            }
        OUTPUTS(CHECK_DIMS_NAMED_KNOWN);
        // I don't check the inputs; parse_dim() would have barfed if any named
        // input dimension wasn't determined. The outputs don't all exist yet,
        // so I need to check


        // The dimensions of each output must be (dims_extra + PROTOTYPE__output__)
#define CREATE_MISSING_OUTPUT(name)                                    \
        if((PyObject*)__py__ ## name == Py_None || __py__ ## name == NULL) \
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
            /* No output array available. Make one                  */  \
            __py__ ## name = (PyArrayObject*)PyArray_SimpleNew(Ndims_output, dims_output_want, selected_typenum__ ## name); \
            if(__py__ ## name == NULL)                                  \
            {                                                           \
                /* Error already set. I simply exit                 */  \
                goto done;                                              \
            }                                                           \
                                                                        \
            if(populate_output_tuple__i >= 0)                           \
            {                                                           \
                PyTuple_SET_ITEM(__py__output__arg,                     \
                                 populate_output_tuple__i,              \
                                 (PyObject*)__py__ ## name);            \
                populate_output_tuple__i++;                             \
                Py_INCREF(__py__ ## name);                              \
            }                                                           \
            else if(__py__output__arg == NULL)                          \
            {                                                           \
                /* one output, no output given */                       \
                __py__output__arg = (PyObject*)__py__ ## name;          \
                Py_INCREF(__py__output__arg);                           \
            }                                                           \
            DEFINE_DIM_VARS(name);                                      \
        }
        OUTPUTS(CREATE_MISSING_OUTPUT);


        // I'm done messing around with the dimensions. Everything passed, and
        // all the arrays have been created. Some arrays MAY have some implicit
        // length-1 dimensions. I can't communicate this to the validation and
        // slice functions. So I explicitly make copies of the dimension and
        // stride arrays, making any implicit length-1 dimensions explicit. The
        // callbacks then see all the dimension data in memory.
        //
        // Most of the time we won't have any implicit dimensions, so these
        // mounted shapes would then be copies of the normal ones
#define MAKE_MOUNTED_COPIES(name)                                       \
        int __ndim__mounted__ ## name = __ndim__ ## name;               \
        if( __ndim__ ## name < PROTOTYPE_LEN_ ## name )                 \
            /* Too few input dimensions. Add dummy dimension of length 1 */ \
            __ndim__mounted__ ## name = PROTOTYPE_LEN_ ## name;         \
        npy_intp __dims__mounted__    ## name[__ndim__mounted__ ## name]; \
        npy_intp __strides__mounted__ ## name[__ndim__mounted__ ## name]; \
        {                                                               \
            int i_dim = -1;                                             \
            for(; i_dim >= -__ndim__ ## name; i_dim--)                  \
                {                                                       \
                    /* copies of the original shapes */                 \
                    __dims__mounted__   ## name[i_dim + __ndim__mounted__ ## name] = __dims__    ## name[i_dim + __ndim__ ## name]; \
                    __strides__mounted__## name[i_dim + __ndim__mounted__ ## name] = __strides__ ## name[i_dim + __ndim__ ## name]; \
                }                                                       \
            for(; i_dim >= -__ndim__mounted__ ## name; i_dim--)         \
                {                                                       \
                    /* extra dummy dimensions, as needed */             \
                    __dims__mounted__    ## name[i_dim + __ndim__mounted__ ## name] = 1; \
                    __strides__mounted__ ## name[i_dim + __ndim__mounted__ ## name] = 0; \
                }                                                       \
        }                                                               \
        /* Now guaranteed >= 0 because of the padding */                \
        int Ndims_extra__mounted__ ## name = __ndim__mounted__ ## name - PROTOTYPE_LEN_ ## name; \
                                                                        \
        /* Ndims_extra and dims_extra[] are already right */

        ARGUMENTS(MAKE_MOUNTED_COPIES);
        OUTPUTS(  MAKE_MOUNTED_COPIES);







        // Each output variable is now an allocated array, and each one has a
        // reference. The argument __py__output__arg ALSO has a reference

#define ARGLIST_CALL_USER_CALLBACK(name)                                \
        __ndim__mounted__       ## name ,                               \
        __dims__mounted__       ## name,                                \
        __strides__mounted__    ## name,                                \
        __ndim__mounted__       ## name - Ndims_extra__mounted__ ## name, \
        &__dims__mounted__      ## name[  Ndims_extra__mounted__ ## name ], \
        &__strides__mounted__   ## name[  Ndims_extra__mounted__ ## name ], \
        PyArray_ITEMSIZE(__py__ ## name),                               \
        (void*)data_argument__  ## name,

#define DEFINE_DATA_ARGUMENT(name) char* data_argument__ ## name;
#define INIT_DATA_ARGUMENT(name) data_argument__ ## name = PyArray_DATA(__py__ ## name);

        ARGUMENTS(DEFINE_DATA_ARGUMENT);
        OUTPUTS(  DEFINE_DATA_ARGUMENT);
        ARGUMENTS(INIT_DATA_ARGUMENT);
        OUTPUTS(  INIT_DATA_ARGUMENT);

        if( ! __{FUNCTION_NAME}__validate(OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                          ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                          {EXTRA_ARGUMENTS_ARGLIST_CALL_C}) )
        {
            if(PyErr_Occurred() == NULL)
                PyErr_SetString(PyExc_RuntimeError, "User-provided validation failed!");
            goto done;
        }

        // if the extra dimensions are degenerate, just return the empty array
        // we have
        for(int i=0; i<Ndims_extra; i++)
            if(dims_extra[i] == 0)
            {
                __py__result__ = (PyObject*)__py__output__arg;
                goto done;
            }

        // if no broadcasting involved, just call the function
        if(Ndims_extra == 0)
        {
            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  {EXTRA_ARGUMENTS_ARGLIST_CALL_C}) )
            {
                if(PyErr_Occurred() == NULL)
                    PyErr_Format(PyExc_RuntimeError, "__{FUNCTION_NAME}__slice failed!");
            }
            else
                __py__result__ = (PyObject*)__py__output__arg;
            goto done;
        }

#if 0
        // most of these should be __mounted ?

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
            // This loop is awkward. I don't update the slice data pointer
            // incrementally with each slice, but advance each dimension for
            // each slice. There should be a better way
            ARGUMENTS(INIT_DATA_ARGUMENT);
            OUTPUTS(  INIT_DATA_ARGUMENT);
#undef DEFINE_DATA_ARGUMENT
#undef INIT_DATA_ARGUMENT

            for( int i_dim=-1;
                 i_dim >= -Ndims_extra;
                 i_dim--)
            {
#define ADVANCE_SLICE(name)                         \
                if(i_dim + Ndims_extra__mounted__ ## name >= 0 &&                 \
                   __dims__mounted__ ## name[i_dim + Ndims_extra__mounted__ ## name] != 1) \
                    data_argument__ ## name += idims_extra[i_dim + Ndims_extra]*__strides__ ## name[i_dim + Ndims_extra__mounted__ ## name];

                ARGUMENTS(ADVANCE_SLICE);
                OUTPUTS(  ADVANCE_SLICE);
            }

            if( ! slice_function( OUTPUTS(  ARGLIST_CALL_USER_CALLBACK)
                                  ARGUMENTS(ARGLIST_CALL_USER_CALLBACK)
                                  {EXTRA_ARGUMENTS_ARGLIST_CALL_C}) )
            {
                if(PyErr_Occurred() == NULL)
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

    // If we allocated any resource into the cookie earlier, we can clean it up
    // now
    {COOKIE_CLEANUP}

    RESET_SIGINT();
    return __py__result__;
}

#undef ARG_DEFINE
#undef NAMELIST
#undef PARSECODE
#undef PARSEARG
#undef DECLARE_DIM_VARS
#undef DEFINE_DIM_VARS
#undef PARSE_DIMS
#undef SLICE_ARG
#undef INPUT_PERCENT_S
#undef INPUT_TYPEOBJ
#undef ARGLIST_CALL_USER_CALLBACK
#undef ADVANCE_SLICE
#undef FREE_PYARRAY
#undef CHECK_DIMS_NAMED_KNOWN
#undef CREATE_MISSING_OUTPUT
#undef MAKE_MOUNTED_COPIES
#undef ARGUMENTS
#undef OUTPUTS
