#include "so.h"
#include <algorithm>
#include <iostream>
using namespace std;

void swap(data_t* data, int i, int j)
{
	data_t temp;
	temp = data[i];
	data[i] = data[j];
	data[j] = temp;
}

int partition(data_t* data, int begin, int end)
{
	int middle = (begin+end)/2;
	data_t pivot = data[middle];
	int pointer = begin;
	int i;

	swap(data, middle, end);

	for(i=begin; i<end; ++i)
	{
		if(data[i] < pivot)
		{
			//swapping data wrt the pivot
			swap(data, i, pointer);
			++pointer;
			// cout<<i<<" "<<data[i]<<" "<<pointer;
		}
	}
	// cout<<pointer;
	swap(data, pointer, end);
	return pointer;
}

void q_sort(data_t* data, int begin, int end)
{
	if(begin>=end)
		return;

	if(end-begin < 100)
	{
		sort(data+begin, data+end+1);
		return;
	}
	int pivot;

	pivot = partition(data, begin, end);
	// cout<<pivot;
	#pragma omp task
	{	
		//recursive call to one part
		q_sort(data, begin, pivot-1);
	}
	#pragma omp task
	{
		//recursive call to second part
		q_sort(data, pivot+1, end);
	}
}

void psort(int n, data_t* data) {

    #pragma omp parallel
    {
    	#pragma omp single
    	{
    		q_sort(data, 0 , n-1);
    	}
    }
}
