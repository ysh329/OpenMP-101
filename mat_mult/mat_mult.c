#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

void print(const float *mat, const int r, const int c)
{
    assert(mat && r>0 && c>0);
    // TODO
    return;
}

#define A(i,j) a[(i)*lda+(j)]
#define B(i,j) b[(i)*ldb+(j)]
#define C(i,j) c[(i)*ldc+(j)]
float *mat_mult(const int m, const int n, const int k,
                const float *a, const int lda,
                const float *b, const int ldb, 
                      float *c, const int ldc)
{
    assert(a && b && c && lda>0 && ldb>0 && ldc>0);
    assert(m>0 && n>0 && k>0);
    int nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    #pragma omp parallel for
    for(int i=0; i<m; ++i)
    {
        #pragma omp parallel for
        for(int j=0; j<n; ++j)
        {
            #pragma omp parallel for
            for(int p=0; p<k; ++p)
            {
                C(i, j) += A(i, p) * B(p, j);
            }
        }
    }
    return c;
}

int main(int argc, char []argv)
{
    // init 
    const int m = 2;
    const int n = 3;
    const int k = 5;

    const int lda = k;
    const int ldb = m;
    const int ldc = n;

    float *a = calloc(m*lda, sizeof(float));
    float *b = calloc(k*ldb, sizeof(float));
    float *c = calloc(m*ldc, sizeof(float));

    // mat mult
    c = mat_mult(m, n, k, a, lda, b, ldb, c, ldc);
   
    return 0;
}
