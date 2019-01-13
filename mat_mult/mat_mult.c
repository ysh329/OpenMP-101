#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>

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
    return;
}


DTYPE *init_mat(DTYPE *mat, const int r, const int c, const DTYPE val)
{
    assert(mat && r>0 && c>0);
    return rand_mat(mat, r, c, val, val);
}

DTYPE *rand_mat(DTYPE *mat, const int r, const int c, const DTYPE min_val, const DTYPE max_val)
{
    assert(mat && r>0 && c>0 && min_val<=max_val);
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
    assert(a && b && c && lda>0 && ldb>0 && ldc>0);
    assert(m>0 && n>0 && k>0);
    int nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    #pragma omp parallel for
    for(int i=0; i<m; ++i)
    {
        //#pragma omp parallel for
        for(int j=0; j<n; ++j)
        {
            //#pragma omp parallel for
            for(int p=0; p<k; ++p)
            {
                C(i, j) += A(i, p) * B(p, j);
            }
        }
    }
    return c;
}

int main(int argc, char *argv[])
{
    // init 
    const int m = 2;
    const int n = 3;
    const int k = 5;

    const int lda = k;
    const int ldb = n;
    const int ldc = n;

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
    c = mat_mult(m, n, k, a, lda, b, ldb, c, ldc);

    print_mat(c, m, ldc, "mat mult c");

    if(a) free(a); a = NULL;
    if(b) free(b); b = NULL;
    if(c) free(c); c = NULL;
   
    return 0;
}
