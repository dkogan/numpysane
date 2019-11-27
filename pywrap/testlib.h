#pragma once

double inner(const double* a,
             const double* b,
             int stride_a,
             int stride_b,
             int n);

void outer(// out assumed contiguous
           double* out,
           const double* a,
           const double* b,
           int stride_a,
           int stride_b,
           int n);
