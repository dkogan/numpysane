#pragma once

// Header for a demo C library being wrapped by numpysane_pywrap. This library
// can compute inner and outer products with arbitrary strides. The inner
// product is implemented with 32-bit integers, 64-bit integers and 64-bit
// floats

#include <stdint.h>

#define DECLARE_INNER_T(T)                      \
T inner_ ## T(const T* a,                       \
              const T* b,                       \
              int stride_a,                     \
              int stride_b,                     \
              int n);
DECLARE_INNER_T(int32_t)
DECLARE_INNER_T(int64_t)
DECLARE_INNER_T(double)
#define inner inner_double

void outer(// out assumed contiguous
           double* out,
           const double* a,
           const double* b,
           int stride_a,
           int stride_b,
           int n);
