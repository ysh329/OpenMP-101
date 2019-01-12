#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N (30)

int main(int argc, char *argv[])
{
    int nthreads, tid, idx;
    float a[N], b[N], c[N];
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);

    // init
    #pragma omp parallel for
    for(idx=0; idx<N; ++idx)
    {
        a[idx] = b[idx] = 1.0;
    }

    // vec add
    #pragma omp parallel for
    for(idx=0; idx<N; ++idx)
    {
        c[idx] = a[idx] + b[idx];
        tid = omp_get_thread_num();
        printf("Thread %2d: c[%2d]=%.2f\n", tid, idx, c[idx]);
    }

    return 0;
}
