#include "pywrap_header.c"

// user-provided
#include "inner.h"
static
bool VALIDATE(void)
{
    return true;
}
static
bool _inner_one_slice( nps_slice_t output,
                       nps_slice_t a,
                       nps_slice_t b)
{
    output.data[0] = inner(a.data,
                           b.data,
                           a.strides[0],
                           b.strides[0],
                           a.shape[0]);
    return true;
}


#define PROTOTYPE_DIM_DEFS_USER \
    const npy_intp PROTOTYPE_a[1] = {-1};                                    \
    const npy_intp PROTOTYPE_b[1] = {-1};                                    \
    /* Not const. updating the named dimensions in-place */             \
    int PROTOTYPE___output__[0];                                        \
    /* compute this at generation time */                               \
    int Ndims_named = 1;









#define COMMA ,
#define CHECK_LAYOUT(   name, npy_type, dims_ref) \
    if(!IS_NULL(name_pyarrayobj)) {                                     \
        int dims[] = dims_ref;                                          \
        int ndims = (int)sizeof(dims)/(int)sizeof(dims[0]);             \
                                                                        \
        if( ndims > 0 )                                                 \
        {                                                               \
            if( PyArray_NDIM((PyArrayObject*)name_pyarrayobj) != ndims )          \
            {                                                           \
                PyErr_Format(PyExc_RuntimeError, "'" #name "' must have exactly %d dims; got %d", ndims, PyArray_NDIM((PyArrayObject*)name_pyarrayobj)); \
                return false;                                           \
            }                                                           \
            for(int i=0; i<ndims; i++)                                  \
                if(dims[i] >= 0 && dims[i] != PyArray_DIMS((PyArrayObject*)name_pyarrayobj)[i]) \
                {                                                       \
                    PyErr_Format(PyExc_RuntimeError, "'" #name "'must have dimensions '" #dims_ref "' where <0 means 'any'. Dims %d got %ld instead", i, PyArray_DIMS((PyArrayObject*)name_pyarrayobj)[i]); \
                    return false;                                       \
                }                                                       \
        }                                                               \
        if( (int)npy_type >= 0 )                                        \
        {                                                               \
            if( PyArray_TYPE((PyArrayObject*)name_pyarrayobj) != npy_type )       \
            {                                                           \
                PyErr_SetString(PyExc_RuntimeError, "'" #name "' must have type: " #npy_type); \
                return false;                                           \
            }                                                           \
            if( !PyArray_IS_C_CONTIGUOUS((PyArrayObject*)name_pyarrayobj) )       \
            {                                                           \
                PyErr_SetString(PyExc_RuntimeError, "'" #name "'must be c-style contiguous"); \
                return false;                                           \
            }                                                           \
        }                                                               \
    }

#define ARGUMENTS(_)                            \
    _(a, NPY_DOUBLE, {-1 COMMA -1       } )     \
    _(b, NPY_DOUBLE, {-1 COMMA -1       } )



static
PyObject* _pywrap_inner(PyObject* NPY_UNUSED(self),
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

    VALIDATE();


    // Helper function to evaluate a given list of arguments in respect to a
    // given broadcasting prototype. This function will flag any errors in the
    // dimensionality of the inputs. If no errors are detected, it returns

    //   dims_extra,dims_named

    // where

    //   dims_extra is the outer shape of the broadcast
    //   dims_named is the values of the named dimensions



    // First I initialize dims_extra: the array containing the broadcasted
    // slices. Each argument calls for some number of extra dimensions, and the
    // overall array is as large as the biggest one of those


    PROTOTYPE_DIM_DEFS_USER;

    const int PROTOTYPE_LEN___output__ = (int)sizeof(PROTOTYPE___output__)/sizeof(PROTOTYPE___output__[0]);
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

        // unrolled:
        //   for( i_arg in range(len(args)))
        //       parse_dim( i_arg, prototype[i_arg], args[i_arg].shape,
        //                  dims_extra );
        bool parse_dim(// input and output
                       npy_intp* dims_named,
                       npy_intp* dims_extra,

                       // input
                       int Ndims_extra,
                       int Ndims_extra_this,
                       const npy_intp* shape_want,
                       int Nshape_want,
                       const npy_intp* shape_got,
                       int Nshape_got )
        {
            // first, I make sure the input is at least as dimension-ful as the
            // prototype. I do this by prepending dummy dimensions of length-1 as
            // necessary
            if(Ndims_extra_this < 0)
            {
                assert(0);
                // shape_arg = (1,) * (-Ndims_extra_this) + shape_arg;
                return false;
            }

            // MAKE SURE THE PROTOTYPE DIMENSIONS MATCH (the trailing dimensions)
            //
            // Loop through the dimensions. Set the dimensionality of any new named
            // argument to whatever the current argument has. Any already-known
            // argument must match
            for( int i_dim=-1;
                 i_dim >= -Nshape_want;
                 i_dim--)
            {
                int i_dim_shape_want = i_dim + Nshape_want;
                int i_dim_var        = i_dim + Nshape_got;
                int dim_shape_want   = shape_want[i_dim_shape_want];
                if(dim_shape_want < 0)
                {
                    // This is a named dimension. These can have any value, but
                    // ALL dimensions of the same name must thave the SAME value
                    // EVERYWHERE
                    if(dims_named[-dim_shape_want-1] < 0)
                        dims_named[-dim_shape_want-1] = shape_got[i_dim_var];

                    dim_shape_want = dims_named[-dim_shape_want-1];
                }

                // The prototype dimension (named or otherwise) now has a numeric
                // value. Make sure it matches what I have
                if(dim_shape_want != shape_got[i_dim_var])
                {
                    // raise NumpysaneError("Argument {} dimension '{}': expected {} but got {}".
                    //     format(name_arg,
                    //            shape_want[i_dim],
                    //            dim_shape_want,
                    //            shape_got[i_dim_var]))
                    assert(0);
                    return false;
                }
            }

            // I now know that this argument matches the prototype. I look at the
            // extra dimensions to broadcast, and make sure they match with the
            // dimensions I saw previously

            // MAKE SURE THE BROADCASTED DIMENSIONS MATCH (the leading dimensions)
            //
            // This argument has Ndims_extra_this dimensions to broadcast. The
            // current shape to broadcast must be at least as large, and must
            // match
            for( int i_dim=-1;
                 i_dim >= -Ndims_extra_this;
                 i_dim--)
            {
                int i_dim_var   = i_dim - Nshape_want + Nshape_got;
                int i_dim_extra = i_dim + Ndims_extra;
                int dim_arg     = shape_got[i_dim_var];

                if (dim_arg != 1)
                {
                    if( dims_extra[i_dim_extra] == 1)
                        dims_extra[i_dim_extra] = dim_arg;
                    else if(dims_extra[i_dim_extra] != dim_arg)
                    {
                        // raise NumpysaneError("Argument {} prototype {} extra broadcast dim {} mismatch: previous arg set this to {}, but this arg wants {}".
                        //     format(name_arg,
                        //            shape_shape_want,
                        //            i_dim,
                        //            dims_extra[i_dim_extra],
                        //            dim_arg))
                        assert(0);
                        return false;
                    }
                }
            }
            return true;
        }

#define PARSE_DIMS(name, npy_type, dims_ref)            \
        parse_dim(dims_named, dims_extra, Ndims_extra,  \
                                                        \
                  Ndims_extra_ ## name,                 \
                  PROTOTYPE_ ## name,                   \
                  PROTOTYPE_LEN_ ## name,               \
                  __dims__ ## name,                     \
                  __ndim__ ## name) ;

        ARGUMENTS(PARSE_DIMS);

        // now have dims_extra,dims_named;

        // I set up the output array.
        //
        // substitute the named dimensions directly into PROTOTYPE___output__
        for(int i=0; i<PROTOTYPE_LEN___output__; i++)
            if(PROTOTYPE___output__[i] < 0 )
            {
                if(dims_named[-PROTOTYPE___output__[i]-1] >= 0)
                    PROTOTYPE___output__[i] = dims_named[-PROTOTYPE___output__[i]-1];

                // output prototype has some unknown named dimension. Handle this later
                assert(0);
            }

        // The shape of the output must be (dims_extra + PROTOTYPE___output__)
        int Ndims_output = Ndims_extra + PROTOTYPE_LEN___output__;
        npy_intp dims_output_want[Ndims_output];
        for(int i=0; i<Ndims_extra; i++)
            dims_output_want[i] = dims_extra[i];
        for(int i=0; i<PROTOTYPE_LEN___output__; i++)
            dims_output_want[i+Ndims_extra] = PROTOTYPE___output__[i];
        if((PyObject*)__py__output__ != Py_None && __py__output__ != NULL)
        {
            // An output array was given. I make sure it has the correct shape. I
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
                int i_dim_var    = i_dim + PyArray_NDIM(__py__output__);
                int i_dim_output = i_dim + Ndims_output;
                int dim_var      = i_dim_var    >= 0 ? PyArray_DIMS(__py__output__)[i_dim_var   ] : 1;
                int dim_output   = i_dim_output >= 0 ? dims_output_want            [i_dim_output] : 1;
                if(dim_var != dim_output)
                {
                    assert(0);
                    fprintf(stderr, "given output array has mismatched dimensions\n");
                }
            }
        }
        else
        {
            // No output array available. Make one
            __py__output__ = (PyArrayObject*)PyArray_SimpleNew(Ndims_output, dims_output_want, NPY_DOUBLE);
            if(__py__output__ == NULL)
            {
                assert(0);
            }
        }

        // if no broadcasting involved, just call the function
        if(Ndims_extra == 0)
        {
            //_inner(); // ...
            assert(0);
        }

        const npy_intp* strides_slice_output = &PyArray_STRIDES(__py__output__)[ Ndims_extra   ];
        const npy_intp* dims_slice_output    = &PyArray_DIMS   (__py__output__)[ Ndims_extra   ];

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

                // BARF IF FALSE


#define ARGLIST_SLICE(name, npy_type, dims_ref)                         \
                ,                                                       \
                (nps_slice_t){ .data    = (double*)slice_ ## name,      \
                               .strides = &__strides__ ## name[ Ndims_extra_ ## name ], \
                               .shape   = &__dims__    ## name[ Ndims_extra_ ## name ] }

                _inner_one_slice( (nps_slice_t){ .data    = (double*)slice_output,
                                                 .strides = strides_slice_output,
                                                 .shape   = dims_slice_output }
                                    ARGUMENTS(ARGLIST_SLICE) );
            }

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
                _inner_one_slice( (nps_slice_t){ .data    = slice_output1,
                            .strides = strides_slice_output,
                            .shape   = dims_slice_output },
                    (nps_slice_t){ .data    = slice_a1,
                            .strides = strides_slice_a,
                            .shape   = dims_slice_a },
                    (nps_slice_t){ .data    = slice_b1,
                            .strides = strides_slice_b,
                            .shape   = dims_slice_b } );
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



        // Have slice. Each slice has a set of input/output pointers where:
        //
        // - type we may or may not want (could be double* or anything else)
        //
        // - dimensionality we want (from the signature), but not necessarily
        //   the strides we want. We have the strides

        // user needs to provide the inner function to do all the work in this
        // state
        //
        // Need to handle unknown dimensionality for out at the start


        __py__result__ = (PyObject*)__py__output__;
    }
 done:

#define FREE_PYARRAY(name, npy_type, dims_ref) Py_XDECREF(__py__ ## name);
    ARGUMENTS(FREE_PYARRAY);

    RESET_SIGINT();
    return __py__result__;
}

static const char _pywrap_inner_docstring[] =
"xxx"
    ;
static PyMethodDef methods[] =
    { PYMETHODDEF_ENTRY(_pywrap_, inner, METH_VARARGS | METH_KEYWORDS),
      {}
    };


#define MODULENAME "innermodule"
#define MODULEDOC  "inner module blah blah blah"

#if PY_MAJOR_VERSION == 2

PyMODINIT_FUNC initinnermodule(void)
{
    PyObject* module =
        Py_InitModule3(MODULENAME, methods, MODULEDOC);
    import_array();
}

#else

static struct PyModuleDef module_def =
    {
     PyModuleDef_HEAD_INIT,
     MODULENAME, MODULEDOC,
     -1,
     methods
    };

PyMODINIT_FUNC PyInit_innermodule(void)
{
    PyObject* module = PyModule_Create(&module_def);
    import_array();
    return module;
}

#endif

