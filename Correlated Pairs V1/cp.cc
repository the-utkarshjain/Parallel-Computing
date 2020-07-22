#include "cp.h"
#include <cmath>

void correlate(int ny, int nx, const float* data, float* result) {
    
    double mean[ny] = {0};
	double stddev[ny] = {0};
	double utkarsh;

	for(int i=0;i<ny;i++){
		for(int j = 0; j<nx; j++){
			utkarsh = (double)data[j + i*nx];
			mean[i] += utkarsh;
			stddev[i] += utkarsh*utkarsh;
		}

		mean[i] = mean[i]/nx;
		stddev[i] -= nx*mean[i]*mean[i];
	}

	double norm = 0;

	for(int y=0; y<ny; y++){
		for(int x=y ; x<ny; x++){

			if(x == y){
				result[x + y*ny] = 1;
				continue;
			}

			norm = 0;
			for(int i=0; i<nx; i++){
				norm += (double)data[i + x*nx]*(double)data[i+ y*nx];
			}

			norm = (norm - nx*mean[y]*mean[x])/sqrt(stddev[x]*stddev[y]);
			result[x + y*ny] = norm;
		}
	}

}