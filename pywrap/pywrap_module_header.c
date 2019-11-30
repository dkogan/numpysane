#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <stdbool.h>
#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
#include <signal.h>
#include <assert.h>

// Python is silly. There's some nuance about signal handling where it sets a
// SIGINT (ctrl-c) handler to just set a flag, and the python layer then reads
// this flag and does the thing. Here I'm running C code, so SIGINT would set a
// flag, but not quit, so I can't interrupt the solver. Thus I reset the SIGINT
// handler to the default, and put it back to the python-specific version when
// I'm done
#define SET_SIGINT() struct sigaction sigaction_old;                    \
do {                                                                    \
    if( 0 != sigaction(SIGINT,                                          \
                       &(struct sigaction){ .sa_handler = SIG_DFL },    \
                       &sigaction_old) )                                \
    {                                                                   \
        PyErr_SetString(PyExc_RuntimeError, "sigaction() failed");      \
        goto done;                                                      \
    }                                                                   \
} while(0)
#define RESET_SIGINT() do {                                             \
    if( 0 != sigaction(SIGINT,                                          \
                       &sigaction_old, NULL ))                          \
        PyErr_SetString(PyExc_RuntimeError, "sigaction-restore failed"); \
} while(0)


// just like PyArray_Converter(), but leave None as None
static
int PyArray_Converter_leaveNone(PyObject* obj, PyObject** address)
{
    if(obj == Py_None)
    {
        *address = Py_None;
        Py_INCREF(Py_None);
        return 1;
    }
    return PyArray_Converter(obj,address);
}

typedef struct
{
    double* data;
    const npy_intp* strides;
    const npy_intp* dims;
} nps_slice_t;

static
bool parse_dim(// input and output
               npy_intp* dims_named,
               npy_intp* dims_extra,

               // input
               int Ndims_extra,
               const char* arg_name,
               int Ndims_extra_this,
               const npy_intp* dims_want, int Ndims_want,
               const npy_intp* dims_got,  int Ndims_got )
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
         i_dim >= -Ndims_want;
         i_dim--)
    {
        int i_dim_dims_want = i_dim + Ndims_want;
        int i_dim_var        = i_dim + Ndims_got;
        int dim_dims_want   = dims_want[i_dim_dims_want];
        if(dim_dims_want < 0)
        {
            // This is a named dimension. These can have any value, but
            // ALL dimensions of the same name must thave the SAME value
            // EVERYWHERE
            if(dims_named[-dim_dims_want-1] < 0)
                dims_named[-dim_dims_want-1] = dims_got[i_dim_var];

            dim_dims_want = dims_named[-dim_dims_want-1];
        }

        // The prototype dimension (named or otherwise) now has a numeric
        // value. Make sure it matches what I have
        if(dim_dims_want != dims_got[i_dim_var])
        {
            if(dims_want[i_dim_dims_want] < 0)
                PyErr_Format(PyExc_RuntimeError,
                             "Argument '%s': prototype says dimension %d (named dimension %d) has length %d, but got %d",
                             arg_name,
                             i_dim, dims_want[i_dim_dims_want],
                             dim_dims_want,
                             dims_got[i_dim_var]);
            else
                PyErr_Format(PyExc_RuntimeError,
                             "Argument '%s': prototype says dimension %d has length %d, but got %d",
                             arg_name,
                             i_dim,
                             dim_dims_want,
                             dims_got[i_dim_var]);
            return false;
        }
    }

    // I now know that this argument matches the prototype. I look at the
    // extra dimensions to broadcast, and make sure they match with the
    // dimensions I saw previously

    // MAKE SURE THE BROADCASTED DIMENSIONS MATCH (the leading dimensions)
    //
    // This argument has Ndims_extra_this dimensions to broadcast. The
    // current dimensions to broadcast must be at least as large, and must
    // match
    for( int i_dim=-1;
         i_dim >= -Ndims_extra_this;
         i_dim--)
    {
        int i_dim_var   = i_dim - Ndims_want + Ndims_got;
        int i_dim_extra = i_dim + Ndims_extra;
        int dim_arg     = dims_got[i_dim_var];

        if (dim_arg != 1)
        {
            if( dims_extra[i_dim_extra] == 1)
                dims_extra[i_dim_extra] = dim_arg;
            else if(dims_extra[i_dim_extra] != dim_arg)
            {
                PyErr_Format(PyExc_RuntimeError,
                             "Argument '%s' dimension %d (broadcasted dimension %d) mismatch. Previously saw length %d, but here have length %d",
                             arg_name,
                             i_dim-Ndims_want, i_dim,
                             dims_extra[i_dim_extra],
                             dim_arg);
                return false;
            }
        }
    }
    return true;
}
