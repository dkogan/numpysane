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

                           // so-far-seen named dimensions. Initially these are
                           // <0 to indicate that they're unknown. As the
                           // broadcasting rules determine the values of these,
                           // the values are stored here (>= 0), and checked for
                           // consistency
                           npy_intp* dims_named,

                           // so-far-seen broadcasted dimensions. Initially
                           // these are 1 to indicate that these are compatible
                           // with anything. As non-1 values are seen, those are
                           // stored here (> 1), and checked for consistency
                           npy_intp* dims_extra,

                           // input
                           const int Ndims_extra,
                           const int Ndims_extra_inputs_only,
                           const char* arg_name,
                           const int Ndims_extra_var,
                           const npy_intp* dims_want, const int Ndims_want,
                           const npy_intp* dims_var,  const int Ndims_var,
                           const bool is_output)
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

    // outputs may be bigger than the inputs (this will result in multiple
    // identical copies in each slice), but may not be smaller. I check that
    // existing extra dimensions are sufficiently large. And then I check to
    // make sure we have enough extra dimensions
    if(is_output)
    {
        for( int i_dim=-1;
             i_dim >= -Ndims_extra_var;
             i_dim--)
        {
            const int i_dim_var = i_dim - Ndims_want + Ndims_var;
            // if we didn't get enough dimensions, use dim=1
            const int dim_var = i_dim_var >= 0 ? dims_var[i_dim_var] : 1;

            const int i_dim_extra = i_dim + Ndims_extra;

            if(dim_var < dims_extra[i_dim_extra])
            {
                PyErr_Format(PyExc_RuntimeError,
                             "Output '%s' dimension %d (broadcasted dimension %d) too small. Inputs have length %d but this output has length %d",
                             arg_name,
                             i_dim-Ndims_want, i_dim,
                             (int)dims_extra[i_dim_extra],
                             dim_var);
                return false;
            }
        }

        // I look through extra dimensions above what this output has to make
        // sure that the output array is big-enough to hold all the output. I
        // only care about the broadcasted slices defined by the input. Because
        // I don't NEED to store all the duplicates created by the output-only
        // broadcasting
        for( int i_dim=-Ndims_extra_var-1;
             i_dim >= -Ndims_extra_inputs_only;
             i_dim--)
        {
            const int i_dim_extra = i_dim + Ndims_extra;

            // What if this test passes, but a subsequent output increases
            // dims_extra[i_dim_extra] so that this would have failed? That is
            // OK. Extra dimensions in the outputs do not create new and
            // different results, and I don't need to make sure I have room to
            // store duplicates
            if(dims_extra[i_dim_extra] > 1)
            {
                // This dimension was set, but this array has a DIFFERENT value
                PyErr_Format(PyExc_RuntimeError,
                             "Argument '%s' dimension %d (broadcasted dimension %d) is too small: this dimension of this output is too small to hold the broadcasted results of size %d",
                             arg_name,
                             i_dim-Ndims_want, i_dim,
                             (int)dims_extra[i_dim_extra]);
                return false;
            }
        }
    }


    for( int i_dim=-1;
         i_dim >= -Ndims_extra_var;
         i_dim--)
    {
        const int i_dim_var = i_dim - Ndims_want + Ndims_var;
        // if we didn't get enough dimensions, use dim=1
        const int dim_var = i_dim_var >= 0 ? dims_var[i_dim_var] : 1;

        const int i_dim_extra = i_dim + Ndims_extra;


        if (dim_var != 1)
        {
            if(i_dim_extra < 0)
            {
                PyErr_Format(PyExc_RuntimeError,
                             "Argument '%s' dimension %d (broadcasted dimension %d) i_dim_extra<0: %d. This shouldn't happen. There's a bug in the implicit-leading-dimension logic. Please report",
                             arg_name,
                             i_dim-Ndims_want, i_dim,
                             i_dim_extra);
                return false;
            }

            // I have a new value for this dimension
            if( dims_extra[i_dim_extra] == 1)
                // This dimension wasn't set yet; I set it
                dims_extra[i_dim_extra] = dim_var;
            else if(dims_extra[i_dim_extra] != dim_var)
            {
                // This dimension was set, but this array has a DIFFERENT value
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
