#include "structs.cuh"
#include "init.h"
#include "LBMconsts.cuh"
#include "phys.h"
#include <nvfunctional>

#include "materials.cuh"

template<class F> __global__ void fill(F);

void init(){
  parsHost.iStep=0;
  //  CHECK_ERROR( cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MaxLevel) );
  //    CHECK_ERROR( cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 100e6) );
  copy2dev( parsHost, pars );
  copy2dev( PPhost, PPdev );

  //CHECK_ERROR( cudaDeviceSetLimit(cudaLimitStackSize, 8*1024) );
  //size_t newStackSize = 32*1024;
  //   CHECK_ERROR( cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024*1024*1024) );

  printf("Malloc data\n");
  parsHost.data.malloc_data(Nx,Ny,Nz);
  copy2dev( parsHost, pars );
  copy2dev( PPhost, PPdev );

  cuTimer init_timer;
  fill<<<dim3(Nx/4,Ny/4,Nz/4),dim3(4,4,4)>>>( [] __device__(int ix, int iy,int iz) {return blank_mat(ix,iy,iz);} );
  cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
  printf("\n");

  printf("Initialization time: %.2f ms\n", init_timer.gettime());
  
  copy2dev( parsHost, pars );
  copy2dev( PPhost, PPdev );

  printf("Peak performance in 10^6 LU/s (normalized to locality coefficient), i.e. update one cell required one cell load + one store:\n");
  const double memclocks[] = {6008, 4513, 1754};
  const size_t memwidth[] = {384, 256, 4096};
  const size_t datatrans = 2*Cell::Qn*sizeof(ftype);
  printf("TitanX: %g | GTX1080: %g | TeslaV100(PCIe): %g \n", 
      memclocks[0]*memwidth[0]/8/datatrans, 
      memclocks[1]*memwidth[1]/8/datatrans, 
      memclocks[2]*memwidth[2]/8/datatrans
  );

}

template<class F> __global__ void fill(F func){
  int ix = threadIdx.x+blockIdx.x*blockDim.x;
  int iy = threadIdx.y+blockIdx.y*blockDim.y;
  int iz = threadIdx.z+blockIdx.z*blockDim.z;
  Cell c;
  //Cell c = pars.data.get_cell_compact<0>(ix,iy,iz);
  ftype4 Vrho = func(ix,iy,iz);

  ftype feq[LBMconsts::Qn];
  c.calcEq(feq, Vrho);
  for(int iq=0; iq<LBMconsts::Qn; iq++) c.f[iq]=feq[iq];

  pars.data.set_cell_compact<0>(c, ix,iy,iz);
}

