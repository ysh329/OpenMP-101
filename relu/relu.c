#include <omp.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>

void print(const float *x, const int len)
{
    assert(x && len>0);
    for(int idx=0; idx<len; ++idx)
    {
        printf("%.2f ", (float)x[idx]);
    }
    printf("\n");
    return;
}

float *relu(float *x, const int len, float *res)
{
    assert(x && len>0 && res);
    int nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    #pragma omp parallel for
    for(int idx=0; idx<len; ++idx)
    {
        res[idx] = x[idx]>0 ? x[idx] : 0;
    }
    return res;
}

int main(int argc, char *argv[])
{
    // init
    const size_t n = 10;
    float *x = calloc(10, sizeof(float));
    memset(x, -1, n*sizeof(float)); // memset only can init value with 0 or -1
    print(x, n);

    // relu
    x = relu(x, n, x);
    print(x, n);

    if(x) free(x); x = NULL;

    return 0;
}
