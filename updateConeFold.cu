#include "structs.cuh"
#include "semaphore.h"
#include "LBMconsts.cuh"
#include "phys.h"
#include "compact-steps.cuh"

void debug_print();

void calcConeFold(int it, std::vector<double>& timings){
  cuTimer t0;
  using namespace CompStep;

  const size_t shmem_size = Nb.x*Nb.y*Nb.z*Cell::Qn*sizeof(ftype);
  printf("Required Shmem size=%ld\n", shmem_size);
  CHECK_ERROR(cudaFuncSetAttribute(compactStepConeFold, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));

  static_assert(Nx%Nb.x==0);
  static_assert(Ny%Nb.y==0);
  static_assert(Nz%Nb.z==0);
  static_assert(Nx>=Ny);
  static_assert(Ny>=Nz);
  //int ix=Nx-Nb.x-(parsHost.iStep-1)*Nb.x;
  for(int ix=Nx-Nb.x; ix>=0; ix-=Nb.x) {
    compactStepConeFold<<<Ny/Nb.y*Nz/Nb.z,dim3(Nb.x,Nb.y,Nb.z),shmem_size>>>(ix);
    cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
  }
  debug_print(); timings.push_back( t0.getlaptime() );
}


