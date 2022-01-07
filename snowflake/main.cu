#include "function.h"
int main(int argc,char* argv[]){
   highprecision *phi,*phi_lap,*tempr,*tempr_lap,*phidx,*phidy,*epsilon,*epsilon_deri;
   CHECK_ERROR(cudaMallocManaged((void**)&phi,sizeof(highprecision)*dimX*dimY));
   CHECK_ERROR(cudaMallocManaged((void**)&phi_lap,sizeof(highprecision)*dimX*dimY));
   CHECK_ERROR(cudaMallocManaged((void**)&tempr,sizeof(highprecision)*dimX*dimY));
   CHECK_ERROR(cudaMallocManaged((void**)&tempr_lap,sizeof(highprecision)*dimX*dimY));
   CHECK_ERROR(cudaMallocManaged((void**)&phidx,sizeof(highprecision)*dimX*dimY));
   CHECK_ERROR(cudaMallocManaged((void**)&phidy,sizeof(highprecision)*dimX*dimY));
   CHECK_ERROR(cudaMallocManaged((void**)&epsilon,sizeof(highprecision)*dimX*dimY));
   CHECK_ERROR(cudaMallocManaged((void**)&epsilon_deri,sizeof(highprecision)*dimX*dimY));
   for(int y=0;y<dimY;y++){
      for(int x=0;x<dimX;x++){
         if(pow(y-(dimY/2+8),2)+pow(x-(dimX/2+8),2)<seed){
            phi[y*dimX+x]=1.0;
         }
      }
    }
    dim3 blocks_pure(unitx,unity);
    dim3 grids_pure(1,1,unitdimX*unitdimY);
    for(int i=0;i<timesteps;i++){
        kernel1_pure<<<grids_pure,blocks_pure>>>(phi,phi_lap,tempr,tempr_lap,phidx,phidy,epsilon,epsilon_deri);
        cudaDeviceSynchronize();
        kernel2_pure<<<grids_pure,blocks_pure>>>(phi,phi_lap,epsilon,epsilon_deri,phidx,phidy,tempr,tempr_lap);
        cudaDeviceSynchronize();


    }
    CHECK_ERROR(cudaFree(phi));CHECK_ERROR(cudaFree(phi_lap));CHECK_ERROR(cudaFree(tempr));
    CHECK_ERROR(cudaFree(tempr_lap));CHECK_ERROR(cudaFree(phidx));CHECK_ERROR(cudaFree(phidy));
    CHECK_ERROR(cudaFree(epsilon));CHECK_ERROR(cudaFree(epsilon_deri));
    return 0;
}