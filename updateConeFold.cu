#include "structs.cuh"
#include "semaphore.h"
#include "LBMconsts.cuh"
#include "phys.h"
#include "compact-steps.cuh"
#include "lbm-steps.cuh"

void debug_print();
__managed__ float* img_buf;
__managed__ cuDevTimers dev_times;

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

  img_buf = parsHost.arr4im.Arr3Dbuf;
  #if DRAW_WAVEFRONT>1
  const int period = Nx/Nb.x;
  int ix=Nx-Nb.x-(parsHost.iStep-1)%period*Nb.x; {
  #else
  for(int ix=Nx-Nb.x; ix>=0; ix-=Nb.x) {
  #endif
    #ifdef ENABLE_DEVICE_TIMERS
    dev_times.reset();
    #endif
    compactStepConeFold <<< Ny/Nb.y*Nz/Nb.z, CompStep::Nblk,shmem_size >>> (ix);
    //compactStepConeFold<<<Ny/Nb.y*Nz/Nb.z,dim3(Nb.x,Nb.y,Nb.z),shmem_size>>>(ix);
    cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
    #ifdef ENABLE_DEVICE_TIMERS
    printf("Average device clocks, steps: ");
    for(int i=0; i<dev_times.N; i++) printf("%g ", double(dev_times.clocks[i])/(Ny/Nb.y*Nz/Nb.z*parsHost.Nt) );
    printf("\n");
    #endif
  }
  debug_print(); timings.push_back( t0.getlaptime() );
}


