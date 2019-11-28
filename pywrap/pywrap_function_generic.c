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

#define ARG_DEFINE(name, npy_type, dims_ref) PyArrayObject* __py__ ## name = NULL;
    ARGUMENTS(ARG_DEFINE);

    SET_SIGINT();

#define NAMELIST(name, npy_type, dims_ref) #name ,
    char* keywords[] = { ARGUMENTS(NAMELIST) "out",
                         NULL };
#define PARSECODE(name, npy_type, dims_ref) "O&"
#define PARSEARG( name, npy_type, dims_ref) PyArray_Converter_leaveNone, &__py__ ## name,
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

#define DECLARE_DIM_VARS(name, npy_type, dims_ref)                     \
    const int       __ndim__    ## name = PyArray_NDIM   (__py__ ## name); \
    const npy_intp* __dims__    ## name = PyArray_DIMS   (__py__ ## name); \
    void*           __data__    ## name = PyArray_DATA   (__py__ ## name); \
    const npy_intp* __strides__ ## name = PyArray_STRIDES(__py__ ## name); \
                                                                        \
    const int PROTOTYPE_LEN_ ## name = (int)sizeof(PROTOTYPE_ ## name)/sizeof(PROTOTYPE_ ## name[0]); \
                                                                        \
    int Ndims_extra_ ## name = __ndim__ ## name - PROTOTYPE_LEN_ ## name;             \
    if(Ndims_extra < Ndims_extra_ ## name) Ndims_extra = Ndims_extra_ ## name;

    ARGUMENTS(DECLARE_DIM_VARS);

    {
        npy_intp dims_extra[Ndims_extra];
        for(int i=0; i<Ndims_extra; i++)
            dims_extra[i] = 1;

        npy_intp dims_named[Ndims_named];
        for(int i=0; i<Ndims_named; i++)
            dims_named[i] = -1;

#define PARSE_DIMS(name, npy_type, dims_ref)    \
        if(!parse_dim(dims_named, dims_extra,   \
                      #name,                    \
                      Ndims_extra,              \
                                                \
                      Ndims_extra_ ## name,     \
                      PROTOTYPE_ ## name,       \
                      PROTOTYPE_LEN_ ## name,   \
                      __dims__ ## name,         \
                      __ndim__ ## name))        \
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
                    // output prototype has some unknown named dimension. Handle this later
                    assert(0);
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

        // if no broadcasting involved, just call the function
        if(Ndims_extra == 0)
        {
            //_inner(); // ...
            assert(0);
        }

        const npy_intp* strides_slice_output = &PyArray_STRIDES(__py__output__)[ Ndims_extra ];
        const npy_intp* dims_slice_output    = &PyArray_DIMS   (__py__output__)[ Ndims_extra ];

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
                stride_extra_elements_a[idim_extra] = PyArray_STRIDES(_py_a)[idim] / sizeof(double);
            else
                stride_extra_elements_a[idim_extra] = 0;

            idim = idim_extra + Ndims_extra_b - Ndims_extra;
            if(idim>=0 && __dims__b[idim] != 1)
                stride_extra_elements_b[idim_extra] = PyArray_STRIDES(_py_b)[idim] / sizeof(double);
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

#define DEFINE_SLICE(name, npy_type, dims_ref) char* slice_ ## name = __data__ ## name;
            ARGUMENTS(DEFINE_SLICE);

            char* slice_output = PyArray_DATA(__py__output__);
            for( int i_dim=-1;
                 i_dim >= -Ndims_extra;
                 i_dim--)
            {

#define ADVANCE_SLICE(name, npy_type, dims_ref)                         \
                if(i_dim + Ndims_extra_ ## name >= 0 &&                 \
                   __dims__ ## name[i_dim + Ndims_extra_ ## name] != 1) \
                    slice_ ## name += idims_extra[i_dim + Ndims_extra]*PyArray_STRIDES(__py__ ## name)[i_dim + Ndims_extra_ ## name];

                ARGUMENTS(ADVANCE_SLICE);

                if(i_dim + Ndims_extra >= 0 &&
                   PyArray_DIMS(__py__output__)[i_dim + Ndims_extra] != 1)
                    slice_output += idims_extra[i_dim + Ndims_extra]*PyArray_STRIDES(__py__output__)[i_dim + Ndims_extra];
            }


#define ARGLIST_SLICE(name, npy_type, dims_ref)                     \
            ,                                                       \
            (nps_slice_t){ .data    = (double*)slice_ ## name,      \
                           .strides = &__strides__ ## name[ Ndims_extra_ ## name ], \
                           .dims    = &__dims__    ## name[ Ndims_extra_ ## name ] }

            __{FUNCTION_NAME}__slice( (nps_slice_t){ .data    = (double*)slice_output,
                                                     .strides = strides_slice_output,
                                                     .dims    = dims_slice_output }
                                ARGUMENTS(ARGLIST_SLICE) );
                // BARF IF FALSE

        } while(next(idims_extra, dims_extra, Ndims_extra));
#if 0
        double* slice_a0      = __data__a;
        double* slice_b0      = __data__b;
        double* slice_output0 = PyArray_DATA(__py__output__);
        int dim_extra0=0;
        while(dim_extra0<dims_extra[0])
        {

            double* slice_a1      = slice_a0;
            double* slice_b1      = slice_b0;
            double* slice_output1 = slice_output0;
            int dim_extra1=0;
            while(dim_extra1<dims_extra[1])
            {
                //bool success =
                _inner_one_slice(
                    (nps_slice_t){ .data    = slice_output1,
                                   .strides = strides_slice_output,
                                   .dims    = dims_slice_output },
                    (nps_slice_t){ .data    = slice_a1,
                                   .strides = strides_slice_a,
                                   .dims    = dims_slice_a },
                    (nps_slice_t){ .data    = slice_b1,
                                   .strides = strides_slice_b,
                                   .dims    = dims_slice_b } );
                //assert(success);

                dim_extra1++;
                slice_output1 = &slice_output0[PyArray_STRIDES(__py__output__)[1]/sizeof(double)];
                slice_a1      = &slice_a1     [stride_extra_elements_a[1]];
                slice_b1      = &slice_b1     [stride_extra_elements_b[1]];
            }

            dim_extra0++;
            slice_output0 = &slice_output0[PyArray_STRIDES(__py__output__)[0]/sizeof(double)];
            slice_a0      = &slice_a0     [stride_extra_elements_a[0]];
            slice_b0      = &slice_b0     [stride_extra_elements_b[0]];
        }
#endif

        __py__result__ = (PyObject*)__py__output__;
    }
 done:

#define FREE_PYARRAY(name, npy_type, dims_ref) Py_XDECREF(__py__ ## name);
    ARGUMENTS(FREE_PYARRAY);

    if(__py__result__ == NULL)
        // An error occurred. If an output array was passed-in or created, I
        // release it
        Py_XDECREF(__py__output__);

    RESET_SIGINT();
    return __py__result__;
}

#undef ARGUMENTS
