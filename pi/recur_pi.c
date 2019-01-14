/*
This program will numerically compute the integral of
                  4/(1+x*x) 
			  
from 0 to 1.  The value of this integral is pi -- which 
is great since it gives us an easy way to check the answer.
This version of the program uses a divide and concquer algorithm
and recurrsion.
History: Written by Tim Mattson, 10/2013
*/
#include <omp.h>
#include <stdio.h>
static long num_steps = 1024*1024*1024;
#define MIN_BLK  1024*1024*256

double pi_comp(int Nstart,int Nfinish,double step)
{  int i,iblk;
   double x, sum = 0.0,sum1, sum2;
   if (Nfinish-Nstart < MIN_BLK){
      for (i=Nstart;i< Nfinish; i++){
         x = (i+0.5)*step;
         sum = sum + 4.0/(1.0+x*x); 
      }
   }
   else{
      iblk = Nfinish-Nstart;
      sum1 = pi_comp(Nstart,         Nfinish-iblk/2,step);
      sum2 = pi_comp(Nfinish-iblk/2, Nfinish,       step);
      sum = sum1 + sum2;
   }return sum;
}
 int main ()
 {
   int i;
   double step, pi, sum;
   double init_time, final_time;
   step = 1.0/(double) num_steps;

   init_time = omp_get_wtime();
   sum = pi_comp(0,num_steps,step);
   pi = step * sum;
   final_time = omp_get_wtime() - init_time;
   printf(" for %ld steps pi = %f in %f secs\n",num_steps,pi,final_time);
  
 }  
