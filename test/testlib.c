#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
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
// only. non-broadcasted "scale" argument scales the output. Similarly, the
// floating-point number in scale_string scales the output, if non-NULL
double innerouter(double* out,

                  const double* a,
                  const double* b,
                  double scale,
                  const char* scale_string,
                  int n)
{
    outer(out,
          n*sizeof(double), sizeof(double),
          a, b,
          sizeof(double), sizeof(double),
          n);
    if(scale_string != NULL)
        scale *= atof(scale_string);
    for(int i=0; i<n*n; i++)
        out[i] *= scale;

    double inner_result =
        inner_double(a, b,
                     sizeof(double), sizeof(double),
                     n);
    return inner_result * scale;
}


#include <stdlib.h>
#define DEFINE_SORTED_INDICES_T(T)                                      \
static int compar_indices_ ## T(const void* _i0, const void* _i1,       \
                                void* _x)                               \
{                                                                       \
    const int i0 = *(const int*)_i0;                                    \
    const int i1 = *(const int*)_i1;                                    \
    const T* x = (const T*)_x;                                          \
    if( x[i0] < x[i1] ) return -1;                                      \
    if( x[i0] > x[i1] ) return  1;                                      \
    return 0;                                                           \
}                                                                       \
/* Assumes that indices_order[] has room for at least N values */       \
void sorted_indices_ ## T(/* output */                                  \
                          int* indices_order,                           \
                                                                        \
                          /* input */                                   \
                          const T* x, int N)                            \
{                                                                       \
    for(int i=0; i<N; i++)                                              \
        indices_order[i] = i;                                           \
    qsort_r(indices_order, N, sizeof(indices_order[0]),                 \
            compar_indices_ ## T, (void*)x);                            \
}
DEFINE_SORTED_INDICES_T(float)
DEFINE_SORTED_INDICES_T(double)
