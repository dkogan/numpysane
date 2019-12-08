#include <stdio.h>
#include <stdint.h>

// Header for a demo C library being wrapped by numpysane_pywrap. This library
// can compute inner and outer products with arbitrary strides. The inner
// product is implemented with 32-bit integers, 64-bit integers and 64-bit
// floats. The outer product is defined with floats only

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

void outer(double* out,
           int stride_out_incol,
           int stride_out_inrow,

           const double* a,
           const double* b,
           int stride_a,
           int stride_b,
           int n)
{
    for(int j=0; j<n; j++)
        for(int i=0; i<n; i++)
            *( (double*)(j*stride_out_incol + i*stride_out_inrow + (char*)(out))) =
                *( (double*)(j*stride_a+(char*)(a))) *
                *( (double*)(i*stride_b+(char*)(b)));
}
