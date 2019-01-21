#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

//pi = (4/1) - (4/3) + (4/5) - (4/7) + (4/9) - (4/11) + (4/13) - (4/15) ...
// pi * 4 = 1/1 - 1/3 + 1/5 - 1/7 + 1/9 - 1/11 + 1/13 - 1/15 ...
double calc_pi_gregory(const long max_iter_times)
{
    register double cur_pi = 0;
    for(register int idx=1; idx<=max_iter_times; idx+=2)
    {
       cur_pi += (idx>>1 & 1) ? -4./idx : 4./idx;
    }
    return cur_pi;
}

double calc_pi_nilakantha(const long max_iter_times)
{
    register double cur_pi = 3;
    for(register int idx=2; idx<=max_iter_times; idx+=2)
    {
       cur_pi += (idx>>1 & 1) ? 4./(idx*(idx+1)*(idx+2)) : -4./(idx*(idx+1)*(idx+2));//TODO 
    }
    return cur_pi;
}

int main()
{
    const long max_iter_times = 1000000000;
    double start_time = 0.0;
    double run_time = 0.0;
    double pi = 0.0;
    
    start_time = omp_get_wtime();
    //pi = calc_pi_gregory(max_iter_times);
    pi = calc_pi_nilakantha(max_iter_times);
    run_time = omp_get_wtime() - start_time;

    printf("pi: %.80f\n", pi);
    printf("time: %lf\n", run_time);
    printf("steps: %ld\n", max_iter_times);

    return 0;
}

