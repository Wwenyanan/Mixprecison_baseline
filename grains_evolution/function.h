#include "./paras/para1.h"
#include "stdio.h"
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <fstream>
#include <string>
#include <sstream>
#include <mma.h>
#include "device_launch_parameters.h"
#include <cuda_fp16.h>
#include <math.h>
using namespace std;
using namespace nvcuda;
#define CHECK_ERROR(error) checkCudaError(error, __FILE__, __LINE__)
#define CHECK_STATE(msg) checkCudaState(msg, __FILE__, __LINE__)
inline void checkCudaError(cudaError_t error, const char *file, const int line)
{
   if (error != cudaSuccess) {
      std::cerr << "CUDA CALL FAILED:" << file << "( " << line << ")- " << cudaGetErrorString(error) << std::endl;
      exit(EXIT_FAILURE);
   }
}
inline void checkCudaState(const char *msg, const char *file, const int line)
{
   cudaError_t error = cudaGetLastError();
   if (error != cudaSuccess) {
      std::cerr << "---" << msg << " Error---" << std::endl;
      std::cerr << file << "( " << line << ")- " << cudaGetErrorString(error) << std::endl;
      exit(EXIT_FAILURE);
   }
}

__global__ void kernel1_pure(highprecision *etaa,highprecision *etab,highprecision* etaa_lap,highprecision *dfdetaa,highprecision *etaa_out){
   int unitindex_x=blockIdx.z%unitdimX;
   int unitindex_y=blockIdx.z/unitdimX;
   int x_offset=threadIdx.x ;
   int x_start=unitindex_x*unitx;
   int x=x_start+x_offset;
   int y_offset=threadIdx.y;
   int y=unitindex_y*unity+y_offset;
   int xs1=x>0?x-1:dimX-1;int ys1=y>0?y-1:dimY-1;
   int xa1=x<dimX-1?x+1:0;int ya1=y<dimY-1?y+1:0;
   highprecision(*etaad)[dimX]=(highprecision(*)[dimX])etaa;
   highprecision(*etabd)[dimX]=(highprecision(*)[dimX])etab;
   highprecision(*etaa_lapd)[dimX]=(highprecision(*)[dimX])etaa_lap;
   highprecision(*dfdetaad)[dimX]=(highprecision(*)[dimX])dfdetaa;
   highprecision(*etaa_outd)[dimX]=(highprecision(*)[dimX])etaa_out;
   etaa_lapd[y][x]=(etaad[y][xs1]+etaad[y][xa1]+etaad[ys1][x]+etaad[ya1][x]-4.0*etaad[y][x])/dxdy;
   highprecision sum=pow(etabd[y][x],2);
   dfdetaad[y][x]=1.0*(2.0*1.0*etaad[y][x]*sum+pow(etaad[y][x],3)-etaad[y][x]);
   etaa_outd[y][x]=etaad[y][x]-dtime*mobil*(dfdetaad[y][x]-grcoef*etaa_lapd[y][x]);

}
