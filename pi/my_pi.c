#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

//pi = (4/1) - (4/3) + (4/5) - (4/7) + (4/9) - (4/11) + (4/13) - (4/15) ...
// pi * 4 = 1/1 - 1/3 + 1/5 - 1/7 + 1/9 - 1/11 + 1/13 - 1/15 ...
double calc_pi(const long max_iter_times)
{
    register double cur_pi = 0;
    for(register int idx=1; idx<=max_iter_times; idx+=2)
    {
       cur_pi = (idx>>1 & 1) ? cur_pi-4./idx : cur_pi+4./idx;//TODO 
    }
    return cur_pi;
}

int main()
{
    const long max_iter_times = 1000000000;
    double start_time = 0.0;
    double run_time = 0.0;
    
    start_time = omp_get_wtime();
    double pi = calc_pi(max_iter_times);
    run_time = omp_get_wtime() - start_time;

    printf("pi: %.128f\n", pi);
    printf("time: %lf\n", run_time);
    printf("steps: %ld\n", max_iter_times);
    return 0;
}

