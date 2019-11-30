#undef ARG_DEFINE
#undef NAMELIST
#undef PARSECODE
#undef PARSEARG
#undef DECLARE_DIM_VARS
#undef PARSE_DIMS
#undef DEFINE_SLICE
#undef ADVANCE_SLICE
#undef ARGLIST_SLICE
#undef FREE_PYARRAY

static
PyObject* __pywrap__{FUNCTION_NAME}(PyObject* NPY_UNUSED(self),
                                    PyObject* args,
                                    PyObject* kwargs)
{
    PyObject*      __py__result__ = NULL;
    PyArrayObject* __py__output__ = NULL;

#define ARG_DEFINE(name) PyArrayObject* __py__ ## name = NULL;
    ARGUMENTS(ARG_DEFINE);

    SET_SIGINT();

#define NAMELIST(name) #name ,
    char* keywords[] = { ARGUMENTS(NAMELIST) "out",
                         NULL };
#define PARSECODE(name) "O&"
#define PARSEARG(name) PyArray_Converter_leaveNone, &__py__ ## name,
    if(!PyArg_ParseTupleAndKeywords( args, kwargs,
                                     ARGUMENTS(PARSECODE) "|O&",
                                     keywords,
                                     ARGUMENTS(PARSEARG)
                                     PyArray_Converter_leaveNone, &__py__output__, NULL))
        goto done;

    // Helper function to evaluate a given list of arguments in respect to a
    // given broadcasting prototype. This function will flag any errors in the
    // dimensionality of the inputs. If no errors are detected, it returns

    //   dims_extra,dims_named

    // where

    //   dims_extra is the outer dimensions of the broadcast
    //   dims_named is the values of the named dimensions


    // First I initialize dims_extra: the array containing the broadcasted
    // slices. Each argument calls for some number of extra dimensions, and the
    // overall array is as large as the biggest one of those

{PROTOTYPE_DIM_DEFS};

    const int PROTOTYPE_LEN__output__ = (int)sizeof(PROTOTYPE__output__)/sizeof(PROTOTYPE__output__[0]);
    // the maximum of Ndims_extra_this for all the arguments
    int Ndims_extra = 0;


    {
        // It's possible for my arguments (and the output) to have fewer dimensions
        // than required by the prototype, and still pass all the dimensionality
        // checks, assuming implied leading dimensions of length 1. For instance I
        // could receive a scalar where a ('n',) dimension is expected, or a ('n',)
        // vector where an ('m','n') array is expected. I thus make a local (and
        // padded) copy of the strides and dims arrays, and use those where needed.
        // Most of the time these will just be copies of the input. The dimension
        // counts and argument counts will be relatively small, so this is only a
        // tiny bit wasteful
#define DECLARE_DIM_VARS(name)                      \
        const int PROTOTYPE_LEN_ ## name = (int)sizeof(PROTOTYPE_ ## name)/sizeof(PROTOTYPE_ ## name[0]); \
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
                __dims__    ## name[i_dim + __ndim__ ## name] = 1;      \
                __strides__ ## name[i_dim + __ndim__ ## name] = 0;      \
            }                                                           \
        }                                                               \
        void* __data__    ## name = PyArray_DATA   (__py__ ## name);    \
                                                                        \
                                                                        \
        /* guaranteed >= 0 because of the padding */                    \
        int Ndims_extra_ ## name = __ndim__ ## name - PROTOTYPE_LEN_ ## name; \
        if(Ndims_extra < Ndims_extra_ ## name) Ndims_extra = Ndims_extra_ ## name;

        ARGUMENTS(DECLARE_DIM_VARS);


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
                      Ndims_extra_ ## name,     \
                      PROTOTYPE_ ## name, PROTOTYPE_LEN_ ## name,   \
                      __dims__   ## name, __ndim__       ## name))  \
            goto done;

        ARGUMENTS(PARSE_DIMS);

        // now have dims_extra,dims_named;

        // I set up the output array.
        //
        // substitute the named dimensions directly into PROTOTYPE__output__
        for(int i=0; i<PROTOTYPE_LEN__output__; i++)
            if(PROTOTYPE__output__[i] < 0 )
            {
                if(dims_named[-PROTOTYPE__output__[i]-1] >= 0)
                    PROTOTYPE__output__[i] = dims_named[-PROTOTYPE__output__[i]-1];
                else
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "Output prototype has some named dimension not encountered in the input. The pywrap generator shouldn't have let this happen");
                    goto done;
                }
            }

{VALIDATE};

        // The dimensions of the output must be (dims_extra + PROTOTYPE__output__)
        int Ndims_output = Ndims_extra + PROTOTYPE_LEN__output__;
        npy_intp dims_output_want[Ndims_output];
        for(int i=0; i<Ndims_extra; i++)
            dims_output_want[i] = dims_extra[i];
        for(int i=0; i<PROTOTYPE_LEN__output__; i++)
            dims_output_want[i+Ndims_extra] = PROTOTYPE__output__[i];
        if((PyObject*)__py__output__ != Py_None && __py__output__ != NULL)
        {
            // An output array was given. I make sure it has the correct dimensions. I
            // allow leading dimensions of length 1. I compare from the end,
            // comparing with 1 if either list runs out
            int Ndims_to_compare1 = PyArray_NDIM(__py__output__);
            int Ndims_to_compare2 = Ndims_output;
            int Ndims_to_compare = Ndims_to_compare1;
            if(Ndims_to_compare < Ndims_to_compare2)
                Ndims_to_compare = Ndims_to_compare2;

            for( int i_dim=-1;
                 i_dim >= -Ndims_to_compare;
                 i_dim--)
            {
                int i_dim_var       = i_dim + PyArray_NDIM(__py__output__);
                int i_dim_output    = i_dim + Ndims_output;
                int dim_var         = i_dim_var    >= 0 ? PyArray_DIMS(__py__output__)[i_dim_var   ] : 1;
                int dim_output_want = i_dim_output >= 0 ? dims_output_want            [i_dim_output] : 1;
                if(dim_var != dim_output_want)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "Given output array dimension %d mismatch. Expected %d but got %d",
                                 i_dim,
                                 dim_output_want, dim_var);
                    goto done;
                }
            }
        }
        else
        {
            // No output array available. Make one
            __py__output__ = (PyArrayObject*)PyArray_SimpleNew(Ndims_output, dims_output_want, NPY_DOUBLE);
            if(__py__output__ == NULL)
            {
                // Error already set. I simply exit
                goto done;
            }
        }

        // similarly to how I treated the inputs above, I handle the
        // dimensionality of the output. I make sure that output arrays with too
        // few dimensions (but enough elements) work properly. This effectively
        // does nothing useful if we created a new output array (we used exactly
        // the "right" shape), but it's required for passed-in output arrays
        // with funny dimensions
        int __ndim__output = PyArray_NDIM(__py__output__);
        if( __ndim__output < PROTOTYPE_LEN__output__ )
            /* Too few output dimensions. Add dummy dimension of length-1 */
            __ndim__output = PROTOTYPE_LEN__output__;
        npy_intp __dims__output[__ndim__output];
        npy_intp __strides__output[__ndim__output];
        {
            const npy_intp* dims_orig    = PyArray_DIMS   (__py__output__);
            const npy_intp* strides_orig = PyArray_STRIDES(__py__output__);
            npy_intp        ndim_orig    = PyArray_NDIM   (__py__output__);
            int i_dim = -1;
            for(; i_dim >= -ndim_orig; i_dim--)
            {
                __dims__output[i_dim + __ndim__output] = dims_orig   [i_dim + ndim_orig];
                __strides__output[i_dim + __ndim__output] = strides_orig[i_dim + ndim_orig];
            }
            for(; i_dim >= -__ndim__output; i_dim--)
            {
                __dims__output[i_dim + __ndim__output] = 1;
                __strides__output[i_dim + __ndim__output] = 0;
            }
        }

        // if no broadcasting involved, just call the function
        if(Ndims_extra == 0)
        {
#define DEFINE_SLICE(name) char* slice_ ## name = __data__ ## name;
            ARGUMENTS(DEFINE_SLICE);

            char* slice_output = PyArray_DATA(__py__output__);

#define ARGLIST_SLICE(name)                               \
            ,                                                       \
            (nps_slice_t){ .data    = (double*)slice_ ## name,      \
                           .strides = &__strides__ ## name[ Ndims_extra_ ## name ], \
                           .dims    = &__dims__    ## name[ Ndims_extra_ ## name ] }

            if( ! __{FUNCTION_NAME}__slice
                  (
                      (nps_slice_t){ .data    = (double*)slice_output,
                                     .strides = __strides__output,
                                     .dims    = __dims__output }
                                     ARGUMENTS(ARGLIST_SLICE)
                  )
                )
            {
                PyErr_Format(PyExc_RuntimeError,
                             "__{FUNCTION_NAME}__slice failed!");
            }
            else
                __py__result__ = (PyObject*)__py__output__;
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

            char* slice_output = PyArray_DATA(__py__output__);
            for( int i_dim=-1;
                 i_dim >= -Ndims_extra;
                 i_dim--)
            {

#define ADVANCE_SLICE(name)                         \
                if(i_dim + Ndims_extra_ ## name >= 0 &&                 \
                   __dims__ ## name[i_dim + Ndims_extra_ ## name] != 1) \
                    slice_ ## name += idims_extra[i_dim + Ndims_extra]*__strides__ ## name[i_dim + Ndims_extra_ ## name];

                ARGUMENTS(ADVANCE_SLICE);

                if(i_dim + Ndims_extra >= 0 &&
                   __dims__output[i_dim + Ndims_extra] != 1)
                    slice_output += idims_extra[i_dim + Ndims_extra]*__strides__output[i_dim + Ndims_extra];
            }

            if( ! __{FUNCTION_NAME}__slice
                  (
                      (nps_slice_t){ .data    = (double*)slice_output,
                                     .strides = __strides__output,
                                     .dims    = __dims__output }
                                     ARGUMENTS(ARGLIST_SLICE)
                  )
                )
            {
                PyErr_Format(PyExc_RuntimeError,
                             "__{FUNCTION_NAME}__slice failed!");
                goto done;
            }

        } while(next(idims_extra, dims_extra, Ndims_extra));

        __py__result__ = (PyObject*)__py__output__;
    }
 done:

#define FREE_PYARRAY(name) Py_XDECREF(__py__ ## name);
    ARGUMENTS(FREE_PYARRAY);

    if(__py__result__ == NULL)
        // An error occurred. If an output array was passed-in or created, I
        // release it
        Py_XDECREF(__py__output__);

    RESET_SIGINT();
    return __py__result__;
}

#undef ARGUMENTS
