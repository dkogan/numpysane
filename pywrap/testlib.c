#include <stdio.h>

double inner(const double* a,
             const double* b,
             int stride_a,
             int stride_b,
             int n)
{
    double s = 0.0;
    for(int i=0; i<n; i++)
        s += *( (double*)(i*stride_a+(char*)(a))) * *( (double*)(i*stride_b+(char*)(b)));
    return s;
}
