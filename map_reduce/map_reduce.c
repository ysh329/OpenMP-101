#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

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
    //#pragma omp parallel for
    #pragma omp parallel for
    for(int idx=0; idx<len; ++idx)
    {
        in[idx] = val;
    }
    return in;
}

DTYPE *map(map_func_t f, const DTYPE *in, DTYPE *out, const int len)
{
    assert(f && in && out && len>0);
    #pragma omp parallel for
    for(int idx=0; idx<len; ++idx)
    {
        out[idx] = f(in[idx]);
    }
    return out;
}

DTYPE reduce(reduce_func_t f, const DTYPE *in, const int len)
{
    assert(f && in && len>0);
    DTYPE res = 0;
    #pragma omp parallel for
    for(int idx=0; idx<len; ++idx)
    {
        res = f(res, in[idx]);
    }
    return res;
}


DTYPE add(const DTYPE x, const DTYPE y)
{
    return x+y;
}

DTYPE poww(const DTYPE x)
{
    return x*x;
}

int main(int argc, char *argv[])
{
    int use_thread_num = omp_get_num_threads();
    int current_thread_idx;
    printf("use_thread_num = %d\n", use_thread_num);

    // init
    const int n = 10;
    printf("n = %d\n", n);
    DTYPE *a = calloc(n, sizeof(DTYPE));
    DTYPE *b = calloc(n, sizeof(DTYPE));
    DTYPE *c = calloc(n, sizeof(DTYPE));

    a = init(a, n, 1);
    b = init(b, n, 2);
    c = init(c, n, 3);

    print(a, n);
    print(b, n);
    print(c, n);
    printf("\n");

    // map, reduce
    DTYPE *mapped_a = map(poww, a, a, n);
    DTYPE *mapped_b = map(poww, b, b, n);
    DTYPE *mapped_c = map(poww, c, c, n);
    print(a, n);
    print(b, n);
    print(c, n);
    printf("\n");

    DTYPE reduced_a = reduce(add, a, n);
    DTYPE reduced_b = reduce(add, b, n);
    DTYPE reduced_c = reduce(add, c, n);

    printf("reduced_a:%.2f\n", (float)reduced_a);
    printf("reduced_b:%.2f\n", (float)reduced_b);
    printf("reduced_c:%.2f\n", (float)reduced_c);

    // free
    if(a) free(a); a = NULL;
    if(b) free(b); b = NULL;
    if(c) free(c); c = NULL;

    return 0;
}
