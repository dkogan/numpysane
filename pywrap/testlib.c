#include <stdio.h>
#include <stdint.h>

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
    for(int i=0; i<n; i++)
        for(int j=0; j<n; j++)
            out[iout++] =
                *( (double*)(i*stride_a+(char*)(a))) *
                *( (double*)(j*stride_b+(char*)(b)));
}
