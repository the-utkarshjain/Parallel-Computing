#include "cp.h"
#include <cmath>
#include <cstring>
#include <cstdlib>

using namespace std;

void correlate(int ny, int nx, const float *data, float *result)
{
	int nb = 5;
	int na = (nx + nb - 1) / nb;
	int nab = na * nb;

	double *utkarsh = (double *)malloc(ny*nx*sizeof(double));

	for(int i=0;i<ny;i++){
		double sum = 0;
		for(int j = 0; j<nx; j++){
			sum += (double)data[j + i*nx];
		}
		sum = sum/nx;
		double mag = 0;

		for(int j = 0; j<nx; j++){
			utkarsh[j + i*nx] =(double)data[j + i*nx] - sum;
			mag += utkarsh[j + i*nx]*utkarsh[j + i*nx];
		}

		for(int j = 0; j<nx; j++){
			utkarsh[j + i*nx] /= sqrt(mag);
		}
		
	}

	double norm = 0;

	for (int y = 0; y < ny; y++)
	{
		for (int x = y; x < ny; x++)
		{

			if (x == y)
			{
				result[x + y * ny] = 1;
				continue;
			}
			
			norm = 0;
			double norma[nb] = {0};
			for (int i = 0; i < na-1; i++)
			{
				for(int k=0;k<nb;k++){
					norma[k] += utkarsh[nb * i + k+ nx*y] * utkarsh[nb * i + k+ nx*x];
				}
			}
			for(int i=nab-nb; i<nx;i++){
				norma[0] += utkarsh[x*nx+i]*utkarsh[y*nx+i];
			}

			for(int i=0; i<nb;i++){
				norm += norma[i];
			}
			result[x + y * ny] = (float)norm;
		}
	}

	free(utkarsh);
}