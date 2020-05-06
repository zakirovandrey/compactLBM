#include "structs.cuh"
#include "semaphore.h"
#include "LBMconsts.cuh"
#include "phys.h"
#include "compact-steps.cuh"
#include "lbm-steps.cuh"

void calcLBM(int it, std::vector<double>& timings);
void calcConeFold(int it, std::vector<double>& timings);
void simple_drop();
void debug_print();
void calcStep(int REV=1){
  cuTimer calct;
  parsHost.iStep++;
  std::vector<double> timings;
  int Ntiles=0;
  calcConeFold(parsHost.iStep, timings);
  copy2dev( parsHost, pars );
  copy2dev( PPhost, PPdev );
  double phys_time=parsHost.iStep;
  double calc_time = calct.gettime();
  printf("Step %6d (physical time %6.3f ms) | Performance: %.2f ms (%.2f MLU/sec) | timings: ", 
      parsHost.iStep ,phys_time, calc_time,
      (unsigned long)Nx*Ny*Nz*parsHost.Nt/calc_time*1e-3     );
  for(auto tmg: timings) printf("%.2f ",tmg);
  printf("\n");
}

void calcLBM(int it, std::vector<double>& timings){
  cuTimer t0;
  using namespace CompStep;

  const size_t shmem_size = 48*1024;
  CHECK_ERROR(cudaFuncSetAttribute(compactStep<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));
  CHECK_ERROR(cudaFuncSetAttribute(compactStep<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));

  compactStep<0><<<dim3(Nx/Nb.x,Ny/Nb.y,Nz/Nb.z),dim3(Nb.x,Nb.y,Nb.z),shmem_size>>>();
  cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
  debug_print(); timings.push_back( t0.getlaptime() );

  compactStep<1><<<dim3(Nx/Nb.x,Ny/Nb.y,Nz/Nb.z),dim3(Nb.x,Nb.y,Nb.z),shmem_size>>>();
  cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
  debug_print(); timings.push_back( t0.getlaptime() );
}

void debug_print(){
   return;
}
