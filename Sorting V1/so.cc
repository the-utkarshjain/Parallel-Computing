#include "so.h"
#include <algorithm>
#include <iostream>
#include <omp.h>
using namespace std;

inline void arraymerge(data_t *a, int size, int *index, int N)
{
  while ( N>1 ) 
  {
    #pragma omp parallel for num_threads(N)
    for(int i=0; i<N+1; i++ ) 
      index[i]=i*size/N; 

    #pragma omp parallel for num_threads(N)
    for(int i=0; i<N; i+=2 ) 
    {
      inplace_merge(a+index[i],a+index[i+1],a + index[i+2]);
    }
    N=N>>1;
  }

}
 
void psort(int n, data_t* data) 
{
  int threads = omp_get_max_threads();
  if(threads==5 || threads== 3)
    threads--;
  else if(threads==6 || threads==7)
    threads=4;
  else if(threads>=8)
    threads=8;

  int *index = (int *)malloc((threads+1)*sizeof(int));
  for(int i=0; i<threads+1; i++) 
  {
    index[i]=i*n/threads;
  }

  #pragma omp parallel for num_threads(threads)
  for(int i=0; i<threads; i++)
  {
    std::sort(data+index[i], data +index[i+1]);
  }
  if(threads>1 ) 
    arraymerge(data,n,index,threads);
  
  free(index);
}
