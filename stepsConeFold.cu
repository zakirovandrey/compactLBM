#include "LBMconsts.cuh"
#include "structs.cuh"
#include "phys.h"
#include "data.cuh"
#include "compact-steps.cuh"
#include "lbm-steps.cuh"

__device__ inline void load_store_cells_perimeter(const int3 crd0, ftype* fi_sh, const int shift){
  const int Nbsz=CompStep::Nbsz;
  const int thid = threadIdx.x + threadIdx.y*CompStep::Nb.x + threadIdx.z*CompStep::Nb.x*CompStep::Nb.y;
  const int Nbs = CompStep::Nb.x*CompStep::Nb.y*CompStep::Nb.z;
  const int warpSize=32;
  const int Nwarps = Nbs/32;
  const int warp_id = thid/32;
  const int lane_id = thid%32;
  if(warp_id>=Nwarps) return;
  const int iq_l=lane_id;
  const int Qn = Cell::Qn;

  //for(int ind=warp_id; ind<Nbs; ind+=Nwarps) {
  //  const int ix = ind%Nbsz;
  //  const int iy = ind/Nbsz%Nbsz;
  //  const int iz = ind/(Nbsz*Nbsz);
  //  if(ix!=0 && iy!=0 && iz!=0) continue;
  const int ind_b1 = Nbsz*Nbsz;
  const int ind_b2 = 2*Nbsz*Nbsz-Nbsz;
  const int NfaceCells = 3*Nbsz*Nbsz-3*Nbsz+1;

  #define USE_PRECALC_SHIFTS
  const enum {UseIndArrSuccessively, UseIndArrSparsely} PreCalcArray
     = UseIndArrSuccessively;

  #ifdef USE_PRECALC_SHIFTS
  const int Nc4warp = (NfaceCells+Nwarps-1)/Nwarps;
  const int index_seq_loop = warp_id*Nc4warp+lane_id%(warpSize/2); static_assert( Nc4warp <= warpSize/2 );
  const int index_seq_warp = warp_id+(lane_id%(warpSize/2))*Nwarps; static_assert( NfaceCells <= Nwarps*warpSize/2 );
  int index=0;
  if(PreCalcArray==UseIndArrSuccessively) index = index_seq_loop; else if(PreCalcArray==UseIndArrSparsely) index = index_seq_warp;
  int _iface,_ix,_iy,_iz;
  if( index<Nbsz*Nbsz ) {
    _iface=0;
    _ix = index%Nbsz;
    _iy = index/Nbsz;
    _iz = 0;
  } else if( index>=Nbsz*Nbsz && index<ind_b2 ) {
    _iface=1;
    _ix = index%Nbsz;
    _iy = 0;
    _iz = index/Nbsz-Nbsz+1;
  } else if( index>=ind_b2 && index<NfaceCells) {
    _iface=2;
    _ix = 0;
    _iy = (index-ind_b2)%(Nbsz-1)+1;
    _iz = (index-ind_b2)/(Nbsz-1)+1;
  }
  int exch_ind_sh = (_ix+shift)%Nbsz + (_iy+shift)%Nbsz*Nbsz + (_iz+shift)%Nbsz*Nbsz*Nbsz; 
  const int3 cycsh=make_int3( _ix?0:Nbsz, _iy?0:Nbsz, _iz?0:Nbsz )*(lane_id>=warpSize/2);
  int exch_gixLR = ( crd0.x+cycsh.x+_ix )%Nx;
  int exch_giyLR = ( crd0.y+cycsh.y+_iy )%Ny;
  int exch_gizLR = ( crd0.z+cycsh.z+_iz )%Nz;
  int exch_glob_indLR = exch_gixLR + exch_giyLR*Nx + exch_gizLR*Nx*Ny;
  if(PreCalcArray==UseIndArrSuccessively) {
      for(int indiq=warp_id*Nc4warp*Qn, iface=0; indiq<min((warp_id+1)*Nc4warp,NfaceCells)*Qn; indiq+=warpSize) {
        const int ind=(indiq+lane_id)/Qn;
        const int iq=(indiq+lane_id)%Qn;
        const int ilane  = ind-warp_id*Nc4warp;

        const int glob_indL = __shfl_sync(0xffffffff, exch_glob_indLR, ilane);
        const int glob_indR = __shfl_sync(0xffffffff, exch_glob_indLR, ilane+warpSize/2);
        const int ind_sh    = __shfl_sync(0xffffffff, exch_ind_sh   , ilane);
        if( indiq+lane_id >= min( (warp_id+1)*Nc4warp,NfaceCells)*Qn ) break;

        pars.data.tiles[glob_indL].f[iq] = fi_sh[ind_sh*Qn+iq];
        fi_sh[ind_sh*Qn+iq] = pars.data.tiles[glob_indR].f[iq];
      }
  } else if(PreCalcArray==UseIndArrSparsely) {
      for(int ind=warp_id, ilane=0, iface=0; ind<NfaceCells; ind+=Nwarps, ilane++) {
        const int glob_indL = __shfl_sync(0xffffffff, exch_glob_indLR, ilane);
        const int glob_indR = __shfl_sync(0xffffffff, exch_glob_indLR, ilane+warpSize/2);
        const int ind_sh    = __shfl_sync(0xffffffff, exch_ind_sh   , ilane);
        if(iq_l<Qn) {
          const int iq=iq_l;
          pars.data.tiles[glob_indL].f[iq] = fi_sh[ind_sh*Qn+iq];
          fi_sh[ind_sh*Qn+iq] = pars.data.tiles[glob_indR].f[iq];
        }
      }
  }
 #else
  if(iq_l>=Qn) return;
  for(int ind=warp_id, iface=0; ind<NfaceCells; ind+=Nwarps) {
    if( ind>=Nbsz*Nbsz && ind<ind_b2 ) iface=1;
    if( ind>=ind_b2 ) iface=2;
    int ix=0, iy=0, iz=0;
    if(iface==0) {
      ix = ind%Nbsz;
      iy = ind/Nbsz;
    } else if(iface==1) {
      ix = ind%Nbsz;
      iz = ind/Nbsz-Nbsz+1;
    } else if(iface==2) {
      iy = (ind-ind_b2)%(Nbsz-1)+1;
      iz = (ind-ind_b2)/(Nbsz-1)+1;
    }
  /*for(int ind=warp_id, iface=0; ind<3*Nbsz*Nbsz; ind+=Nwarps) {
    iface = ind/(Nbsz*Nbsz);
    const int ind1=ind%Nbsz;
    const int ind2=ind/Nbsz-iface*Nbsz;
    if( iface==2 && ind1==0) continue;
    if( iface>0 && ind2==0 ) continue;;
    //if( warp_id != ind%Nwarps ) continue;
    int ix=0, iy=0, iz=0;
    if(iface==0) {
      ix = ind1;
      iy = ind2;
    } else if(iface==1) {
      ix = ind1;;
      iz = ind2;
    } else if(iface==2) {
      iy = ind1;
      iz = ind2;
    }*/
    const int ind_sh = (ix+shift)%Nbsz + (iy+shift)%Nbsz*Nbsz + (iz+shift)%Nbsz*Nbsz*Nbsz; 
    const int gixL = ( crd0.x+ix )%Nx;
    const int giyL = ( crd0.y+iy )%Ny;
    const int gizL = ( crd0.z+iz )%Nz;
    const int3 cycsh=make_int3(ix?0:Nbsz,iy?0:Nbsz,iz?0:Nbsz);
    const int gixR = ( crd0.x+cycsh.x+ix )%Nx;
    const int giyR = ( crd0.y+cycsh.y+iy )%Ny;
    const int gizR = ( crd0.z+cycsh.z+iz )%Nz;
    const int iq=iq_l;
    const int glob_indL = gixL+giyL*Nx+gizL*Nx*Ny;
    const int glob_indR = gixR+giyR*Nx+gizR*Nx*Ny;
    pars.data.tiles[glob_indL].f[iq] = fi_sh[ind_sh*Qn+iq];
    fi_sh[ind_sh*Qn+iq] = pars.data.tiles[glob_indR].f[iq];
    /*if(iq==0 && ix==0 && iy==Nbsz/2 && iz==Nbsz/2 && blockIdx.x==0)  printf(
        "shift=%d save at %d %d %d load %d %d %d from value from %d = %g\n",
        shift, gixL,giyL,gizL, gixR,giyR,gizR, ind_sh*Qn+iq, fi_sh[ind_sh*Qn+iq]);*/
  }
  #endif
}

__device__ inline void coneFoldLoop(int3 gCrd, ftype* fi_sh){
  const int Nbsz=CompStep::Nbsz;
  const int Qn=Cell::Qn;
  const char3 pbits3 = make_char3(threadIdx.x&1,threadIdx.y&1,threadIdx.z&1);
  const char pbits = pbits3.x|pbits3.y<<1|pbits3.z<<2;
  for(int it=0; it<pars.Nt; it++, gCrd+=make_int3(1) ){
    const char shift = it%Nbsz;
    if(gCrd.x>=Nx) gCrd.x-=Nx;
    if(gCrd.y>=Ny) gCrd.y-=Ny;
    if(gCrd.z>=Nz) gCrd.z-=Nz;
    Cell c;

    short sh_index[Qn];
    for(int iq=0; iq<Cell::Qn; iq++) {
      const char4 shifts = LBMconsts::compact_access[iq+(Cell::Qn)*pbits];
      char3 shCrd = make_char3( (threadIdx.x+shift+shifts.x)%Nbsz, (threadIdx.y+shift+shifts.y)%Nbsz, (threadIdx.z+shift+shifts.z)%Nbsz );
      const short shid = shCrd.x+shCrd.y*Nbsz + shCrd.z*Nbsz*Nbsz;
      const short rev_iq = shifts.w;
      sh_index[iq] =  shid*Qn + rev_iq ;
    }
    for(int iq=0; iq<Cell::Qn; iq++) c.f[iq] = fi_sh[ sh_index[iq] ];

    //for(int iq=0; iq<Cell::Qn; iq++) c.f[iq]++;
    c.collision();

    for(int iq=0; iq<Cell::Qn; iq++) fi_sh[ sh_index[iq] ] = c.f[iq];
   
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
