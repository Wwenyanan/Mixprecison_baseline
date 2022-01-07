#include "./paras/para5.h"
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

//浓度场part1
__global__ void con1_pure(highprecision *con,highprecision *con_lap,highprecision *eta1,highprecision *eta2,highprecision *dfdcon,highprecision *dummy,int i){
    int unitindex_x=blockIdx.z%unitdimX;
    int unitindex_y=blockIdx.z/unitdimX;
    int x_offset=threadIdx.x ;
    int x_start=unitindex_x*unitx;
    int x=x_start+x_offset;
    int y_offset=threadIdx.y;
    int y=unitindex_y*unity+y_offset;
    int xs1=x>0?x-1:dimX-1;int ys1=y>0?y-1:dimY-1;
    int xa1=x<dimX-1?x+1:0;int ya1=y<dimY-1?y+1:0;
    highprecision(*cond)[dimX]=(highprecision(*)[dimX])con;
    highprecision(*con_lapd)[dimX]=(highprecision(*)[dimX])con_lap;
    highprecision(*eta1d)[dimX]=(highprecision(*)[dimX])eta1;
    highprecision(*eta2d)[dimX]=(highprecision(*)[dimX])eta2;
    highprecision(*dfdcond)[dimX]=(highprecision(*)[dimX])dfdcon;
    highprecision(*dummyd)[dimX]=(highprecision(*)[dimX])dummy;
    con_lapd[y][x]=(cond[y][xs1]+cond[y][xa1]+cond[ys1][x]+cond[ya1][x]-4.0*cond[y][x])/dxdy;
    //对浓度场求导
    highprecision sum2=pow(eta1d[y][x],2)+pow(eta2d[y][x],2);
    highprecision sum3=pow(eta1d[y][x],3)+pow(eta2d[y][x],3);
    dfdcond[y][x]=1.0*(2.0*cond[y][x]+4.0*sum3-6.0*sum2)-2.0*16.0*pow(cond[y][x],2)*(1.0-cond[y][x])+2.0*16.0*cond[y][x]*pow(1.0-cond[y][x],2);
    dummyd[y][x]=dfdcond[y][x]-0.5*coefm*con_lapd[y][x];
}
__global__ void con2_pure(highprecision* dummy,highprecision* dummy_lap,highprecision* con,highprecision* eta1,highprecision* eta2,int i){
    int unitindex_x=blockIdx.z%unitdimX;
    int unitindex_y=blockIdx.z/unitdimX;
    int x_offset=threadIdx.x ;
    int x_start=unitindex_x*unitx;
    int x=x_start+x_offset;
    int y_offset=threadIdx.y;
    int y=unitindex_y*unity+y_offset;
    int xs1=x>0?x-1:dimX-1;int ys1=y>0?y-1:dimY-1;
    int xa1=x<dimX-1?x+1:0;int ya1=y<dimY-1?y+1:0;
    highprecision(*dummyd)[dimX]=(highprecision(*)[dimX])dummy;
    highprecision(*dummy_lapd)[dimX]=(highprecision(*)[dimX])dummy_lap;
    highprecision(*cond)[dimX]=(highprecision(*)[dimX])con;
    highprecision(*eta1d)[dimX]=(highprecision(*)[dimX])eta1;
    highprecision(*eta2d)[dimX]=(highprecision(*)[dimX])eta2;
    dummy_lapd[y][x]=(dummyd[y][xs1]+dummyd[y][xa1]+dummyd[ys1][x]+dummyd[ya1][x]-4.0*dummyd[y][x])/dxdy;
    highprecision phi=pow(cond[y][x],3)*(10.0-15.0*cond[y][x]+6.0*pow(cond[y][x],2)); //插值函数
    highprecision sum=eta1d[y][x]*eta2d[y][x]*2;
    highprecision mobil=dvol*phi+dvap*(1.0-phi)+dsur*cond[y][x]*(1.0-cond[y][x])+dgrb*sum;
    cond[y][x]=cond[y][x]+dtime*mobil*dummy_lapd[y][x];
    if(cond[y][x]>=1) cond[y][x]=1;
    else if(cond[y][x]<0)cond[y][x]=0;
}
//相场变化
__global__ void phi1_pure(highprecision* eta1,highprecision* eta1_out,highprecision* eta2,highprecision* eta1_lap,highprecision* dfdeta1,highprecision* con,int i){
    int unitindex_x=blockIdx.z%unitdimX;
    int unitindex_y=blockIdx.z/unitdimX;
    int x_offset=threadIdx.x ;
    int x_start=unitindex_x*unitx;
    int x=x_start+x_offset;
    int y_offset=threadIdx.y;
    int y=unitindex_y*unity+y_offset;
    highprecision(*eta1d)[dimX]=(highprecision(*)[dimX])eta1;
    highprecision(*eta2d)[dimX]=(highprecision(*)[dimX])eta2;
    highprecision(*eta1_outd)[dimX]=(highprecision(*)[dimX])eta1_out;
    highprecision(*eta1_lapd)[dimX]=(highprecision(*)[dimX])eta1_lap;
    highprecision(*dfdeta1d)[dimX]=(highprecision(*)[dimX])dfdeta1;
    highprecision(*cond)[dimX]=(highprecision(*)[dimX])con;
    int xs1=x>0?x-1:dimX-1;int ys1=y>0?y-1:dimY-1;
    int xa1=x<dimX-1?x+1:0;int ya1=y<dimY-1?y+1:0;
    //lap算子
    eta1_lapd[y][x]=(eta1d[y][xs1]+eta1d[y][xa1]+eta1d[ys1][x]+eta1d[ya1][x]-4.0*eta1d[y][x])/dxdy;
    //自由能对相求导
    highprecision sum2=pow(eta1d[y][x],2)+pow(eta2d[y][x],2);
    dfdeta1d[y][x]=1.0*(-12.0*pow(eta1d[y][x],2)*(2.0-cond[y][x])+12.0*eta1d[y][x]*(1.0-cond[y][x])+12.0*eta1d[y][x]*sum2);
    eta1_outd[y][x]=eta1d[y][x]-dtime*coefl*(dfdeta1d[y][x]-0.5*coefk*eta1_lapd[y][x]);
    if(eta1_outd[y][x]>=1) eta1_outd[y][x]=1;
    else if(eta1_outd[y][x]<0)eta1_outd[y][x]=0;
}
__global__ void phi2_pure(highprecision* eta2,highprecision* eta2_out,highprecision* eta1,highprecision* eta2_lap,highprecision* dfdeta2,highprecision* con,int i){
    int unitindex_x=blockIdx.z%unitdimX;
    int unitindex_y=blockIdx.z/unitdimX;
    highprecision(*eta1d)[dimX]=(highprecision(*)[dimX])eta1;
    highprecision(*eta2d)[dimX]=(highprecision(*)[dimX])eta2;
    highprecision(*eta2_outd)[dimX]=(highprecision(*)[dimX])eta2_out;
    highprecision(*eta2_lapd)[dimX]=(highprecision(*)[dimX])eta2_lap;
    highprecision(*dfdeta2d)[dimX]=(highprecision(*)[dimX])dfdeta2;
    highprecision(*cond)[dimX]=(highprecision(*)[dimX])con;
    int x_offset=threadIdx.x ;
    int x_start=unitindex_x*unitx;
    int x=x_start+x_offset;
    int y_offset=threadIdx.y;
    int y=unitindex_y*unity+y_offset;
    int xs1=x>0?x-1:dimX-1;int ys1=y>0?y-1:dimY-1;
    int xa1=x<dimX-1?x+1:0;int ya1=y<dimY-1?y+1:0;
    //lap算子
    eta2_lapd[y][x]=(eta2d[y][xs1]+eta2d[y][xa1]+eta2d[ys1][x]+eta2d[ya1][x]-4.0*eta2d[y][x])/dxdy;
    //自由能对相求导
    highprecision sum2=pow(eta1d[y][x],2)+pow(eta2d[y][x],2);
    dfdeta2d[y][x]=1.0*(-12.0*pow(eta2d[y][x],2)*(2.0-cond[y][x])+12.0*eta2d[y][x]*(1.0-cond[y][x])+12.0*eta2d[y][x]*sum2);
    eta2_outd[y][x]=eta2d[y][x]-dtime*coefl*(dfdeta2d[y][x]-0.5*coefk*eta2_lapd[y][x]);
    if(eta2_outd[y][x]>=1) eta2_outd[y][x]=1;
    else if(eta2_outd[y][x]<0)eta2_outd[y][x]=0;
}
