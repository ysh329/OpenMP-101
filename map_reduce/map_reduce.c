#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

//#define DEBUG
#define DTYPE int

typedef DTYPE (*map_func_t)(DTYPE);
typedef DTYPE (*reduce_func_t)(DTYPE, DTYPE);

void print(const DTYPE *in, const int len)
{
    assert(in && len>0);
    for(int idx=0; idx<len; ++idx)
    {
        printf("%.2f ", (float)in[idx]);
    }
    printf("\n");
    return;
}

DTYPE *init(DTYPE *in, const int len, const DTYPE val)
{
    assert(in && len>0);
    #pragma omp parallel for
    for(int idx=0; idx<len; ++idx)
    {
        in[idx] = val;
    }
    return in;
}

DTYPE *map(register map_func_t f, register const DTYPE *in, register DTYPE *out, register const int len)
{
    assert(f && in && out && len>0);
    #pragma omp parallel for
    for(register int idx=0; idx<len; ++idx)
    {
        out[idx] = f(in[idx]);
    }
    return out;
}

DTYPE reduce(register reduce_func_t f, register const DTYPE *in, register const int len)
{
    assert(f && in && len>0);
    DTYPE res = 0;
    #pragma omp parallel for
    for(register int idx=0; idx<len; ++idx)
    {
        DTYPE e = in[idx];
        #pragma omp critical
        res = f(res, e);
    }
    return res;
}


DTYPE add(register const DTYPE x, register const DTYPE y)
{
    return x+y;
}

DTYPE poww(register const DTYPE x)
{
    return x*x;
}

int main(int argc, char *argv[])
{
    int use_thread_num = omp_get_num_threads();
    int current_thread_idx;
    double start_time = 0.0;
    double end_time = 0.0;
    printf("use_thread_num = %d\n", use_thread_num);

    // init
    const int n = 1e6;
    printf("n = %d\n", n);
    DTYPE *a = calloc(n, sizeof(DTYPE));
    DTYPE *b = calloc(n, sizeof(DTYPE));
    DTYPE *c = calloc(n, sizeof(DTYPE));

    printf("start init\n");
    a = init(a, n, 1);
    b = init(b, n, 2);
    c = init(c, n, 3);

#ifdef DEBUG
    print(a, n);
    print(b, n);
    print(c, n);
    printf("\n");
#endif

    // map, reduce
    printf("start map\n");
    start_time = omp_get_wtime(); 
    DTYPE *mapped_a = map(poww, a, a, n);
    end_time = omp_get_wtime(); 
    printf("mapped_a:%.4f\n", end_time-start_time);

    start_time = omp_get_wtime(); 
    DTYPE *mapped_b = map(poww, b, b, n);
    end_time = omp_get_wtime(); 
    printf("mapped_b:%.4f\n", end_time-start_time);

    start_time = omp_get_wtime(); 
    DTYPE *mapped_c = map(poww, c, c, n);
    end_time = omp_get_wtime(); 
    printf("mapped_c:%.4f\n", end_time-start_time);

#ifdef DEBUG
    print(a, n);
    print(b, n);
    print(c, n);
    printf("\n");
#endif

    printf("start reduce\n");
    start_time = omp_get_wtime(); 
    DTYPE reduced_a = reduce(add, a, n);
    end_time = omp_get_wtime();
    printf("finish reduced_a:%.4f\n", end_time-start_time);

    start_time = omp_get_wtime(); 
    DTYPE reduced_b = reduce(add, b, n);
    end_time = omp_get_wtime();
    printf("finish reduced_b:%.4f\n", end_time-start_time);

    start_time = omp_get_wtime(); 
    DTYPE reduced_c = reduce(add, c, n);
    end_time = omp_get_wtime();
    printf("finish reduced_b:%.4f\n", end_time-start_time);

    printf("reduced_a:%.2f\n", (float)reduced_a);
    printf("reduced_b:%.2f\n", (float)reduced_b);
    printf("reduced_c:%.2f\n", (float)reduced_c);

    // free
    if(a) free(a); a = NULL;
    if(b) free(b); b = NULL;
    if(c) free(c); c = NULL;

    return 0;
}
