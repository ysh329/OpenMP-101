/*
This program will numerically compute the integral of
                  4/(1+x*x) 
				  
from 0 to 1.  The value of this integral is pi -- which 
is great since it gives us an easy way to check the answer.
The is the original sequential program.  It uses the timer
from the OpenMP runtime library
History: Written by Tim Mattson, 11/99.
*/
#include <stdio.h>
#include <omp.h>
static long max_iter_num = 1000000000;
double step;
int main ()
{
    int i;
    double x, pi, sum = 0.0;
    double start_time, run_time;

    step = 1.0/(double) max_iter_num;
    start_time = omp_get_wtime();

    for(int i = 1; i <= max_iter_num; i++)
    {
        x = (i-0.5)*step;
        sum = sum + 4.0/(1.0+x*x);
    }

    pi = step * sum;
    run_time = omp_get_wtime() - start_time;
    printf("\n pi with %ld steps is %lf in %lf seconds\n ",max_iter_num,pi,run_time);

    return 0;
}	
