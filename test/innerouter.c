#include <stdio.h>
#include <stdint.h>

// Test C library being wrapped by numpysane_pywrap. This library can compute
// inner and outer products

// Inner product supports arbitrary strides, and 3 data types
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

// Outer product supports arbitrary strides, and only the "double" data type
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

// inner and outer product together. Only contiguous data is supported. "double"
// only
double innerouter(double* out,

                  const double* a,
                  const double* b,
                  int n)
{
    outer(out,
          n*sizeof(double), sizeof(double),
          a, b,
          sizeof(double), sizeof(double),
          n);
    return inner_double(a, b,
                        sizeof(double), sizeof(double),
                        n);
}
