#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>

//#define DEBUG
#define DTYPE float

void print_mat(const DTYPE *mat, const int r, const int c, const char *name);
DTYPE *init_mat(DTYPE *mat, const int r, const int c, const DTYPE val);
DTYPE *rand_mat(DTYPE *mat, const int r, const int c, const DTYPE min_val, const DTYPE max_val);
DTYPE *mat_mult(const int m, const int n, const int k,
                const DTYPE *a, const int lda,
                const DTYPE *b, const int ldb, 
                      DTYPE *c, const int ldc);


#define MAT(i,j) mat[(i)*c+(j)]
void print_mat(const DTYPE *mat, const int r, const int c, const char *name)
{
#ifdef DEBUG
    assert(mat && r>0 && c>0);
    printf("---- %s ----\n", name);
    for(int i=0; i<r; ++i)
    {
        for(int j=0; j<c; ++j)
        {
            printf("%.2f ", (float)MAT(i, j));
        }
        printf("\n");
    }
    printf("\n\n");
#endif
    return;
}


DTYPE *init_mat(DTYPE *mat, const int r, const int c, const DTYPE val)
{
    assert(mat && r>0 && c>0);
    return rand_mat(mat, r, c, val, val);
}

DTYPE *rand_mat(DTYPE *mat, const int r, const int c, const DTYPE min_val, const DTYPE max_val)
{
#ifdef DEBUG
    assert(mat && r>0 && c>0 && min_val<=max_val);
#endif
    srand(time( NULL ));
    int range_val = ((int)(max_val - min_val)==0) ? 0x7ffffffff : (int)(max_val-min_val);
    for(int i=0; i<r; ++i)
    {
        for(int j=0; j<c; ++j)
        {
            MAT(i,j) = (rand() % range_val) + min_val;
        }
    }
    return mat;
}

#define A(i,j) a[(i)*lda+(j)]
#define B(i,j) b[(i)*ldb+(j)]
#define C(i,j) c[(i)*ldc+(j)]
DTYPE *mat_mult(const int m, const int n, const int k,
                const DTYPE *a, const int lda,
                const DTYPE *b, const int ldb, 
                      DTYPE *c, const int ldc)
{
#ifdef DEBUG
    assert(a && b && c && lda>0 && ldb>0 && ldc>0);
    assert(m>0 && n>0 && k>0);
#endif
    //int nthreads = omp_get_num_threads();
    //printf("Number of threads = %d\n", nthreads);
    #pragma omp parallel for
    for(register int i=0; i<m; i+=4)
    {
        //#pragma omp parallel for
        for(register int j=0; j<n; ++j)
        {
            //#pragma omp parallel for
            register DTYPE *a0p = &A(i,   0);
            register DTYPE *a1p = &A(i+1, 0);
            register DTYPE *a2p = &A(i+2, 0);
            register DTYPE *a3p = &A(i+3, 0);
            register DTYPE  bp0 = B(0,   j);

            for(register int p=0; p<k; ++p)
            {
                C(i, j)   += *a0p * bp0;
                C(i+1, j) += *a1p * bp0;
                C(i+2, j) += *a2p * bp0;
                C(i+3, j) += *a3p * bp0;
                ++a0p; ++a1p; ++a2p; ++a3p;
            }
        }
    }
    return c;
}

int main(int argc, char *argv[])
{
    // init 
    const int loop_times = 10;

    const int m = 1024;
    const int n = 1024;
    const int k = 1024;

    const int lda = k;
    const int ldb = n;
    const int ldc = n;

    double start_time = 0.0;
    double end_time   = 0.0;

    DTYPE *a = calloc(m*lda, sizeof(DTYPE));
    DTYPE *b = calloc(k*ldb, sizeof(DTYPE));
    DTYPE *c = calloc(m*ldc, sizeof(DTYPE));

    print_mat(a, m, lda, "a");
    print_mat(b, k, ldb, "b");
    print_mat(c, m, ldc, "c");

    a = rand_mat(a, m, lda, 1, 10);
    b = rand_mat(b, k, ldb, 1, 10);
    c = init_mat(c, m, ldc, 0);

    print_mat(a, m, lda, "init a");
    print_mat(b, k, ldb, "init b");
    print_mat(c, m, ldc, "init c");

    // mat mult
    for(int idx=0; idx<loop_times; ++idx)
    {
        start_time = omp_get_wtime();
        c = mat_mult(m, n, k, a, lda, b, ldb, c, ldc);
        end_time = omp_get_wtime();
        printf("idx:%d time:%f second(s)\n", idx, end_time-start_time);
    }

    print_mat(c, m, ldc, "mat mult c");

    if(a) free(a); a = NULL;
    if(b) free(b); b = NULL;
    if(c) free(c); c = NULL;
   
    return 0;
}
