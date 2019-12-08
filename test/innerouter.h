#pragma once

// Header for a test C library being wrapped by numpysane_pywrap. This library
// can compute inner and outer products

#include <stdint.h>

// Inner product supports arbitrary strides, and 3 data types
#define DECLARE_INNER_T(T)                      \
T inner_ ## T(const T* a,                       \
              const T* b,                       \
              int stride_a,                     \
              int stride_b,                     \
              int n);
DECLARE_INNER_T(int32_t)
DECLARE_INNER_T(int64_t)
DECLARE_INNER_T(double)

// Outer product supports arbitrary strides, and only the "double" data type
void outer(double* out,
           int stride_out_incol,
           int stride_out_inrow,

           const double* a,
           const double* b,
           int stride_a,
           int stride_b,
           int n);

// inner and outer product together. Only contiguous data is supported. "double"
// only
double innerouter(double* out,

                  const double* a,
                  const double* b,
                  int n);

