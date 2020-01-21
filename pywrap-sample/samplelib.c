#include <stdio.h>
#include <stdint.h>

// A demo C library being wrapped by numpysane_pywrap. This library can compute
// inner and outer products with arbitrary strides. The inner product is
// implemented with 32-bit integers, 64-bit integers and 64-bit floats


#define DEFINE_INNER_T(T)                                               \
T inner_ ## T(const T* a,                                               \
              const T* b,                                               \
              int stride_a,                                             \
              int stride_b,                                             \
              int n)                                                    \
{                                                                       \
    T s = 0.0;                                                          \
    for(int i=0; i<n; i++)                                              \
        s += *( (T*)(i*stride_a+(char*)(a))) * *( (T*)(i*stride_b+(char*)(b))); \
    return s;                                                           \
}

DEFINE_INNER_T(int32_t)
DEFINE_INNER_T(int64_t)
DEFINE_INNER_T(double)

void outer(// out assumed contiguous
           double* out,
           const double* a,
           const double* b,
           int stride_a,
           int stride_b,
           int n)
{
    int iout = 0;
    for(int j=0; j<n; j++)
        for(int i=0; i<n; i++)
            out[iout++] =
                *( (double*)(j*stride_a+(char*)(a))) *
                *( (double*)(i*stride_b+(char*)(b)));
}
