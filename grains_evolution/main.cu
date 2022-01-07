#include "function.h"
int main(int argc,char* argv[]){
    //空间申请
    highprecision *eta1,*eta2,*eta1_lap,*eta2_lap,*dfdeta1,*dfdeta2,*eta1_out,*eta2_out;
    CHECK_ERROR(cudaMallocManaged((void**)&eta1,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&eta2,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&eta1_out,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&eta2_out,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&eta1_lap,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&eta2_lap,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&dfdeta1,sizeof(highprecision)*dimX*dimY));
    CHECK_ERROR(cudaMallocManaged((void**)&dfdeta2,sizeof(highprecision)*dimX*dimY));
    // 数据准备
    for(int y=1;y<=dimY;y++){
        for(int x=1;x<=dimX;x++){
            float dis1=sqrt(pow(x-Rx,2)+pow(y-Ry,2));
            if(dis1<=R){
                eta1[(y-1)*dimX+x-1]=0;
                eta2[(y-1)*dimX+x-1]=1;
            }
            else{
                eta1[(y-1)*dimX+x-1]=1;
                eta2[(y-1)*dimX+x-1]=0;
            }

        }
    }


    // 线程数量设置
    dim3 blocks_pure(unitx,unity);
    dim3 grids_pure(1,1,unitdimX*unitdimY);
    for(int i=0;i<timesteps;i++){
        kernel1_pure<<<grids_pure,blocks_pure>>>(eta1,eta2,eta1_lap,dfdeta1,eta1_out);
        cudaDeviceSynchronize();
        kernel1_pure<<<grids_pure,blocks_pure>>>(eta2,eta1,eta2_lap,dfdeta2,eta2_out);
        cudaDeviceSynchronize();
        swap(eta1,eta1_out);
        swap(eta2,eta2_out);
    }

    // 释放空间
    CHECK_ERROR(cudaFree(eta1));
    CHECK_ERROR(cudaFree(eta2));
    CHECK_ERROR(cudaFree(eta1_out));
    CHECK_ERROR(cudaFree(eta2_out));
    CHECK_ERROR(cudaFree(eta1_lap));
    CHECK_ERROR(cudaFree(eta2_lap));
    CHECK_ERROR(cudaFree(dfdeta1));
    CHECK_ERROR(cudaFree(dfdeta2));
    return 0;

}