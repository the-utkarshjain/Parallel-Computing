#include "mf.h"
#include <algorithm>
using namespace std;

void mf(int ny, int nx, int hy, int hx, const float *in, float *out)
{
    
    #pragma omp parallel 
    {   int limitx, limity;
        #pragma omp for collapse(2)
            for (int y = 0; y < ny; y++)
            {   
                for (int x = 0; x < nx; x++)
                {
                    double temp = 0.0;
                    double pixels[(2*hx+1) * (2*hy+1)];
                    int i = 0;

                    for (limity = max(0, y - hy); limity < min(ny, y + hy+1); limity++)    
                    {
                        // #pragma omp for
                        for (limitx = max(0, x - hx); limitx < min(nx, x + hx+1); limitx++)
                        {
                            // #pragma omp critical 
                            {
                                pixels[i] = (double)in[limitx + nx * limity];
                                i = i+1;
                            }
                            
                        }
                    }
                    // #pragma omp barrier
                    nth_element(pixels, pixels+i/2, pixels+i);
                    temp = pixels[i / 2];
                    if (i % 2 == 0)
                    {
                        nth_element(pixels, pixels+i/2-1, pixels+i);
                        temp += pixels[(i)/2-1];
                        temp = temp/2;
                    }
                    
                    out[x + nx*y] = (float)temp;
                }
            }
        

    }
    
}
