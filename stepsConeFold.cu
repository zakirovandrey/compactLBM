#include "LBMconsts.cuh"
#include "structs.cuh"
#include "phys.h"
#include "data.cuh"
#include "compact-steps.cuh"

__device__ inline void load_store_cells_perimeter(const int3 crd0, ftype* fi_sh, const int shift){
  const int Nbsz=CompStep::Nbsz;
  const int thid = threadIdx.x + threadIdx.y*CompStep::Nb.x + threadIdx.z*CompStep::Nb.x*CompStep::Nb.y;
  const int Nbs = CompStep::Nb.x*CompStep::Nb.y*CompStep::Nb.z;
  const int Nwarps = Nbs/32;
  const int warp_id = thid/32;
  const int lane_id = thid%32;
  if(warp_id>=Nwarps) return;
  const int iq_l=lane_id;
  const int Qn = Cell::Qn;
  if(iq_l<Qn) {
    for(int ind=warp_id; ind<Nbs; ind+=Nwarps) {
      const int ix = ind%Nbsz;
      const int iy = ind/Nbsz%Nbsz;
      const int iz = ind/(Nbsz*Nbsz);
      if(ix!=0 && iy!=0 && iz!=0) continue;
      const int ind_sh = (ix+shift)%Nbsz + (iy+shift)%Nbsz*Nbsz + (iz+shift)%Nbsz*Nbsz*Nbsz; 
      const int gixL = ( crd0.x+ix )%Nx;
      const int giyL = ( crd0.y+iy )%Ny;
      const int gizL = ( crd0.z+iz )%Nz;
      const int3 cycsh=make_int3(ix?0:Nbsz,iy?0:Nbsz,iz?0:Nbsz);
      const int gixR = ( crd0.x+cycsh.x+ix )%Nx;
      const int giyR = ( crd0.y+cycsh.y+iy )%Ny;
      const int gizR = ( crd0.z+cycsh.z+iz )%Nz;
      const int iq=iq_l;
      pars.data.tiles[gixL+giyL*Nx+gizL*Nx*Ny].f[iq] = fi_sh[ind_sh*Qn+iq];
      fi_sh[ind_sh*Qn+iq] = pars.data.tiles[gixR+giyR*Nx+gizR*Nx*Ny].f[iq];
      /*if(iq==0 && ix==0 && iy==Nbsz/2 && iz==Nbsz/2 && blockIdx.x==0)  printf(
          "shift=%d save at %d %d %d load %d %d %d from value from %d = %g\n",
          shift, gixL,giyL,gizL, gixR,giyR,gizR, ind_sh*Qn+iq, fi_sh[ind_sh*Qn+iq]);*/
    }
  }
}

__device__ inline void coneFoldLoop(int3 gCrd, ftype* fi_sh){
  const int Nbsz=CompStep::Nbsz;
  const int Qn=Cell::Qn;
  for(int it=0; it<pars.Nt; it++, gCrd+=make_int3(1) ){
    const int shift = it%Nbsz;
    if(gCrd.x>=Nx) gCrd.x-=Nx;
    if(gCrd.y>=Ny) gCrd.y-=Ny;
    if(gCrd.z>=Nz) gCrd.z-=Nz;
    Cell c;
    const char3 pbits3 = make_char3(threadIdx.x&1,threadIdx.y&1,threadIdx.z&1);
    const char pbits = pbits3.x|pbits3.y<<1|pbits3.z<<2;
  
    for(int iq=0; iq<Cell::Qn; iq++) {
      const char4 shifts = LBMconsts::compact_access[iq+(Cell::Qn)*pbits];
      int3 shCrd = make_int3(threadIdx)+make_int3(shift)+make_int3(shifts.x,shifts.y,shifts.z);
      shCrd.x%=Nbsz; shCrd.y%=Nbsz; shCrd.z%=Nbsz;
      const int shid = shCrd.x+shCrd.y*Nbsz + shCrd.z*Nbsz*Nbsz;
      const int rev_iq = shifts.w;
      c.f[iq] = fi_sh[ shid*Qn + rev_iq ];
    }

    //for(int iq=0; iq<Cell::Qn; iq++) c.f[iq]++;
    c.collision();

    for(int iq=0; iq<Cell::Qn; iq++) {
      const char4 shifts = LBMconsts::compact_access[iq+(Cell::Qn)*pbits];
      int3 shCrd = make_int3(threadIdx)+make_int3(shift)+make_int3(shifts.x,shifts.y,shifts.z);
      shCrd.x%=Nbsz; shCrd.y%=Nbsz; shCrd.z%=Nbsz;
      const int shid = shCrd.x+shCrd.y*CompStep::Nbsz + shCrd.z*CompStep::Nbsz*CompStep::Nbsz;
      const int rev_iq = shifts.w;
      fi_sh[ shid*Qn + rev_iq ] = c.f[iq];
    }
   
   __syncthreads();
    load_store_cells_perimeter(gCrd, fi_sh, shift);
   __syncthreads();
  }
}

__global__ __launch_bounds__(CompStep::Nblk) void compactStepConeFold(int ix_base){
//ix_base-=128;
//if(blockIdx.x!=128+16) return;
  const int parity=0;
  //const int ix = (blockIdx.x*CompStep::Nb.x+threadIdx.x + parity)%Nx;
  //const int iy = (blockIdx.y*CompStep::Nb.y+threadIdx.y + parity)%Ny;
  //const int iz = (blockIdx.z*CompStep::Nb.z+threadIdx.z + parity)%Nz;
  const int Nbsz = CompStep::Nbsz;
  const int ind1s = blockIdx.x%(Ny/Nbsz)*Nbsz;
  const int ind2s = blockIdx.x/(Ny/Nbsz)*Nbsz;
  const int ix0 = ix_base          +ind1s+ind2s;
  const int iy0 = Ny-CompStep::Nb.y-ind1s;
  const int iz0 = Nz-CompStep::Nb.z-ind2s;
  //printf("ix_base=%d ix0,iy0,iz0=%d %d %d\n", ix_base,ix0,iy0,iz0);

  const int thid = threadIdx.x + threadIdx.y*CompStep::Nb.x + threadIdx.z*CompStep::Nb.x*CompStep::Nb.y;

  const int Qn = Cell::Qn;

  extern __shared__ ftype fi_sh[];
  //if(thid==0) printf("block=%d ix0,iy0,iz0=%d %d %d\n", blockIdx.x, ix0%Nx,iy0%Ny,iz0%Nz);
  
  load_store_datablock(make_int3(ix0,iy0,iz0), fi_sh, RW::Load);
  __syncthreads();

  coneFoldLoop(make_int3(ix0,iy0,iz0), fi_sh);

  const int shift = pars.Nt%CompStep::Nbsz;

  __syncthreads();
  load_store_datablock(make_int3(ix0,iy0,iz0)+make_int3(pars.Nt), fi_sh, RW::Store, shift);

}
