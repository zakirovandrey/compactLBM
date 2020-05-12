#include "LBMconsts.cuh"
#include "structs.cuh"
#include "phys.h"
#include "data.cuh"
#include "compact-steps.cuh"
#include "lbm-steps.cuh"

template<int parity>__global__ __launch_bounds__(CompStep::Nblk) void compactStep_base(){
  const int ix = blockIdx.x*CompStep::Nb.x+threadIdx.x;
  const int iy = blockIdx.y*CompStep::Nb.y+threadIdx.y;
  const int iz = blockIdx.z*CompStep::Nb.z+threadIdx.z;
  
  Cell c = pars.data.get_cell_compact<parity>(ix,iy,iz);
  c.collision();
  pars.data.set_cell_compact<parity>(c, ix,iy,iz);
}

template<int parity> __global__ __launch_bounds__(CompStep::Nblk) void compactStep(){
  const int ix = (blockIdx.x*CompStep::Nb.x+threadIdx.x + parity)%Nx;
  const int iy = (blockIdx.y*CompStep::Nb.y+threadIdx.y + parity)%Ny;
  const int iz = (blockIdx.z*CompStep::Nb.z+threadIdx.z + parity)%Nz;

  const int thid = threadIdx.x + threadIdx.y*CompStep::Nb.x + threadIdx.z*CompStep::Nb.x*CompStep::Nb.y;
  const int warp_id = thid/32;
  const int lane_id = thid%32;
  const int ix0 = blockIdx.x*CompStep::Nb.x + parity;
  const int iy0 = blockIdx.y*CompStep::Nb.y + parity;
  const int iz0 = blockIdx.z*CompStep::Nb.z + parity;

  const int Qn = Cell::Qn;

  //__shared__ ftype fi_sh[Nbs*Qn];
  extern __shared__ ftype fi_sh[];
  
  load_store_datablock(thid, make_int3(ix0,iy0,iz0), fi_sh, RW::Load);
  //for(int iq=0; iq<Cell::Qn; iq++) fi_sh[thid*Qn+iq] = pars.data.tiles[ix+iy*Nx+iz*Nx*Ny].f[iq];
  __syncthreads();

  Cell c;
  const char3 pbits3 = make_char3(threadIdx.x&1,threadIdx.y&1,threadIdx.z&1);
  const char pbits = pbits3.x|pbits3.y<<1|pbits3.z<<2;
  
  for(int iq=0; iq<Cell::Qn; iq++) {
    const char4 shifts = LBMconsts::compact_access[iq+(Cell::Qn)*pbits];
    const int3 shCrd = make_int3(threadIdx.x+shifts.x, threadIdx.y+shifts.y, threadIdx.z+shifts.z);
    const int shid = shCrd.x+shCrd.y*CompStep::Nb.x + shCrd.z*CompStep::Nb.x*CompStep::Nb.y;
    const int rev_iq = shifts.w;
    c.f[iq] = fi_sh[ shid*Qn + rev_iq ];
  }

  //for(int iq=0; iq<Cell::Qn; iq++) c.f[iq]++;
  c.collision();

  for(int iq=0; iq<Cell::Qn; iq++) {
    const char4 shifts = LBMconsts::compact_access[iq+(Cell::Qn)*pbits];
    const int3 shCrd = make_int3(threadIdx.x+shifts.x, threadIdx.y+shifts.y, threadIdx.z+shifts.z);
    const int shid = shCrd.x+shCrd.y*CompStep::Nb.x + shCrd.z*CompStep::Nb.x*CompStep::Nb.y;
    const int rev_iq = shifts.w;
    fi_sh[ shid*Qn + rev_iq ] = c.f[iq];
  }

  __syncthreads();
  load_store_datablock(thid, make_int3(ix0,iy0,iz0), fi_sh, RW::Store);
  //for(int iq=0; iq<Cell::Qn; iq++) pars.data.tiles[ix+iy*Nx+iz*Nx*Ny].f[iq] = fi_sh[thid*Qn+iq];

}

template<int parity>__global__ __launch_bounds__(CompStep::Nblk) void compactStep_4tests(){
  const int ix = blockIdx.x*CompStep::Nb.x+threadIdx.x;
  const int iy = blockIdx.y*CompStep::Nb.y+threadIdx.y;
  const int iz = blockIdx.z*CompStep::Nb.z+threadIdx.z;
  
  Cell c;
  
  const char3 pbits3 = make_char3(ix&1,iy&1,iz&1)^parity;
  const char pbits = pbits3.x|pbits3.y<<1|pbits3.z<<2;
  for(int iq=0; iq<Cell::Qn; iq++) {
    const int3 gCrd = make_int3(ix,iy,iz);
    c.f[iq] = pars.data.tiles[ gCrd.x + gCrd.y*Nx+ gCrd.z*Nx*Ny ].f[iq];
  }

  for(int i=0;i<LBMconsts::Qn;i++) c.f[i]++;

  for(int iq=0; iq<Cell::Qn; iq++) {
    const int3 gCrd = make_int3(ix,iy,iz);
    pars.data.tiles[ gCrd.x + gCrd.y*Nx+ gCrd.z*Nx*Ny ].f[iq] = c.f[iq];
  }

}


template __global__ void compactStep<0>();
template __global__ void compactStep<1>();
