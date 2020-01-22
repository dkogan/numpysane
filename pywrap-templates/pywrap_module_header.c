#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <stdbool.h>
#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
#include <signal.h>

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

static
bool parse_dim_for_one_arg(// input and output
                           npy_intp* dims_named,
                           npy_intp* dims_extra,

                           // input
                           int Ndims_extra,
                           const char* arg_name,
                           int Ndims_extra_var,
                           const npy_intp* dims_want, int Ndims_want,
                           const npy_intp* dims_var,  int Ndims_var,
                           bool is_output)
{
    // MAKE SURE THE PROTOTYPE DIMENSIONS MATCH (the trailing dimensions)
    //
    // Loop through the dimensions. Set the dimensionality of any new named
    // argument to whatever the current argument has. Any already-known
    // argument must match
    for( int i_dim=-1;
         i_dim >= -Ndims_want;
         i_dim--)
    {
        int i_dim_want = i_dim + Ndims_want;
        int dim_want   = dims_want[i_dim_want];

        int i_dim_var = i_dim + Ndims_var;
        // if we didn't get enough dimensions, use dim=1
        int dim_var = i_dim_var >= 0 ? dims_var[i_dim_var] : 1;

        if(dim_want < 0)
        {
            // This is a named dimension. These can have any value, but
            // ALL dimensions of the same name must thave the SAME value
            // EVERYWHERE
            if(dims_named[-dim_want-1] < 0)
                dims_named[-dim_want-1] = dim_var;

            dim_want = dims_named[-dim_want-1];
        }

        // The prototype dimension (named or otherwise) now has a numeric
        // value. Make sure it matches what I have
        if(dim_want != dim_var)
        {
            if(dims_want[i_dim_want] < 0)
                PyErr_Format(PyExc_RuntimeError,
                             "Argument '%s': prototype says dimension %d (named dimension %d) has length %d, but got %d",
                             arg_name,
                             i_dim, (int)dims_want[i_dim_want],
                             dim_want,
                             dim_var);
            else
                PyErr_Format(PyExc_RuntimeError,
                             "Argument '%s': prototype says dimension %d has length %d, but got %d",
                             arg_name,
                             i_dim,
                             dim_want,
                             dim_var);
            return false;
        }
    }

    // I now know that this argument matches the prototype. I look at the
    // extra dimensions to broadcast, and make sure they match with the
    // dimensions I saw previously

    // MAKE SURE THE BROADCASTED DIMENSIONS MATCH (the leading dimensions)
    //
    // This argument has Ndims_extra_var dimensions above the prototype (may be
    // <0 if there're implicit leading length-1 dimensions at the start). The
    // current dimensions to broadcast must match
    //
    // Also handle special broadcasting logic for outputs. The extra dimensions
    // on the output must match the extra dimensions from the inputs EXACTLY.
    // Some things could be reasonably supported, but it's all not very useful,
    // and error-prone
    if(is_output && Ndims_extra_var != Ndims_extra)
    {
        PyErr_Format(PyExc_RuntimeError,
                     "Outputs must match the broadcasted dimensions EXACTLY. '%s' has %d extra, broadcasted dimensions while the inputs have %d",
                     arg_name,
                     Ndims_extra_var, Ndims_extra);
        return false;
    }

    for( int i_dim=-1;
         i_dim >= -Ndims_extra_var;
         i_dim--)
    {
        int i_dim_var = i_dim - Ndims_want + Ndims_var;
        // if we didn't get enough dimensions, use dim=1
        int dim_var = i_dim_var >= 0 ? dims_var[i_dim_var] : 1;

        if (dim_var != 1)
        {
            int i_dim_extra = i_dim + Ndims_extra;
            if(i_dim_extra < 0)
            {
                PyErr_Format(PyExc_RuntimeError,
                             "Argument '%s' dimension %d (broadcasted dimension %d) i_dim_extra<0: %d. This shouldn't happen. There's a bug in the implicit-leading-dimension logic. Please report",
                             arg_name,
                             i_dim-Ndims_want, i_dim,
                             i_dim_extra);
                return false;
            }

            if(is_output && dims_extra[i_dim_extra] != dim_var)
            {
                PyErr_Format(PyExc_RuntimeError,
                             "Outputs must match the broadcasted dimensions EXACTLY. '%s' dimension %d (broadcasted dimension %d) has length %d, while the inputs have %d",
                             arg_name,
                             i_dim-Ndims_want, i_dim,
                             dim_var, dims_extra[i_dim_extra]);
                return false;
            }

            if( dims_extra[i_dim_extra] == 1)
                dims_extra[i_dim_extra] = dim_var;
            else if(dims_extra[i_dim_extra] != dim_var)
            {
                PyErr_Format(PyExc_RuntimeError,
                             "Argument '%s' dimension %d (broadcasted dimension %d) mismatch. Previously saw length %d, but here have length %d",
                             arg_name,
                             i_dim-Ndims_want, i_dim,
                             (int)dims_extra[i_dim_extra],
                             dim_var);
                return false;
            }
        }
    }
    return true;
}
