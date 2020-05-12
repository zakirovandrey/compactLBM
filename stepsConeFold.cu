#include "LBMconsts.cuh"
#include "structs.cuh"
#include "phys.h"
#include "data.cuh"
#include "compact-steps.cuh"
#include "lbm-steps.cuh"

extern __managed__ cuDevTimers dev_times;

__device__ inline void load_store_cells_perimeter(const int3 crd0, ftype* fi_sh, const int shift){
  const int Nbsz=CompStep::Nbsz;
  const int thid = threadIdx.x + threadIdx.y*CompStep::Nb.x + threadIdx.z*CompStep::Nb.x*CompStep::Nb.y;
  const int Nbs = CompStep::Nblk;
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
  const enum {UseShmem, UseIndArrSuccessively, UseIndArrSparsely, UseAllThreadsSuccessively} PreCalcArray
     = UseShmem;
     //= UseAllThreadsSuccessively;
     //= UseIndArrSuccessively;
     //= UseIndArrSparsely;

  #ifdef USE_PRECALC_SHIFTS
  auto get_ixyz_from_index = [] __device__ (const int _index){
    int _iface,_ix,_iy,_iz;
    if(_index<ind_b1 ) {
      _iface=0;
      _ix =_index%Nbsz;
      _iy =_index/Nbsz;
      _iz = 0;
    } else if(_index>=ind_b1 && _index<ind_b2 ) {
      _iface=1;
      _ix = _index%Nbsz;
      _iy = 0;
      _iz = _index/Nbsz-Nbsz+1;
    } else if(_index>=ind_b2 && _index<NfaceCells) {
      _iface=2;
      _ix = 0;
      _iy = (_index-ind_b2)%(Nbsz-1)+1;
      _iz = (_index-ind_b2)/(Nbsz-1)+1;
    }
    return make_int3(_ix,_iy,_iz);
  };
  if(PreCalcArray==UseShmem){
    register TimeCounter time_counter(dev_times);
    static_assert(CompStep::Nblk>=NfaceCells);
    const int th_b1 = ((ind_b1-1)/32+1)*32;
    const int th_b2 = ((th_b1+ind_b2-ind_b1-1)/32+1)*32;
    bool skip=0;
    int index = thid;
    /*if(thid>=ind_b1 && thid<th_b1 || thid>=th_b1+ind_b2-ind_b1 && thid<th_b2) skip=1;
    else if(thid>=th_b2) index = (thid-th_b2)+ind_b2;
    else if(thid>=th_b1) index=(thid-th_b1)+ind_b1;
    static_assert(CompStep::Nblk>=NfaceCells+th_b2-ind_b2);*/
    __shared__ int2 indexes_glob[NfaceCells];
    __shared__ short indexes_loc[NfaceCells];
    if(index<NfaceCells && !skip) {
      const int3 icrd = get_ixyz_from_index(index);
      const short ind_sh = (icrd.x+shift)%Nbsz + (icrd.y+shift)%Nbsz*Nbsz + (icrd.z+shift)%Nbsz*Nbsz*Nbsz; 
      const int3 cycsh=make_int3( icrd.x?0:Nbsz, icrd.y?0:Nbsz, icrd.z?0:Nbsz );
      int2 glob_indLR;
      glob_indLR.x = (crd0.x+icrd.x)%Nx + (crd0.y+icrd.y)%Ny*Nx + (crd0.z+icrd.z)%Nz*Nx*Ny;
      glob_indLR.y = (crd0.x+cycsh.x+icrd.x)%Nx + (crd0.y+cycsh.y+icrd.y)%Ny*Nx + (crd0.z+cycsh.z+icrd.z)%Nz*Nx*Ny;
      indexes_glob[index] = glob_indLR;
      indexes_loc[index] = ind_sh;
    }
    __syncthreads(); if(thid==0) time_counter.lap();
    __syncthreads();
    const int QnP = Tile::Qn4; 
    const int Nc4blk=Nwarps*warpSize/QnP;
    #pragma unroll 1
    for(int indiq=thid,iter=0; iter<(NfaceCells-1)/Nc4blk+1; indiq+=Nc4blk*QnP, iter++) {
      if( thid>=Nc4blk*QnP || indiq>=NfaceCells*QnP ) break;
      const int iq = thid%QnP;
      const int ind = indiq/QnP;
      const short ind_sh = indexes_loc[ind];
      const int2 glob_indLR = indexes_glob[ind];
      const int glob_indL = glob_indLR.x;
      const int glob_indR = glob_indLR.y;
      const int indsh0 = SHMEM_INDEX(iq*4, ind_sh);
      ftype4 fromsh;
      const int SlumpLength = QnP*4-Qn;
      if(SlumpLength<4 || iq*4<Qn-0) fromsh.x = fi_sh[indsh0+0];
      if(SlumpLength<3 || iq*4<Qn-1) fromsh.y = fi_sh[indsh0+1];
      if(SlumpLength<2 || iq*4<Qn-2) fromsh.z = fi_sh[indsh0+2];
      if(SlumpLength<1 || iq*4<Qn-3) fromsh.w = fi_sh[indsh0+3];
      pars.data.tiles[glob_indL].f4[iq] = fromsh;
      const ftype4 fromGlobR = pars.data.tiles[glob_indR].f4[iq];
      if(SlumpLength<4 || iq*4<Qn-0) fi_sh[indsh0+0] = fromGlobR.x;
      if(SlumpLength<3 || iq*4<Qn-1) fi_sh[indsh0+1] = fromGlobR.y;
      if(SlumpLength<2 || iq*4<Qn-2) fi_sh[indsh0+2] = fromGlobR.z;
      if(SlumpLength<1 || iq*4<Qn-3) fi_sh[indsh0+3] = fromGlobR.w;
      //const int indsh0 = SHMEM_INDEX(iq*2, ind_sh);
      //ftype2 fromsh;
      //const int SlumpLength = QnP*2-Qn;
      //if(SlumpLength<2 || iq*2<Qn-0) fromsh.x = fi_sh[indsh0+0];
      //if(SlumpLength<1 || iq*2<Qn-1) fromsh.y = fi_sh[indsh0+1];
      //pars.data.tiles[glob_indL].f2[iq] = fromsh;
      //const ftype2 fromGlobR = pars.data.tiles[glob_indR].f2[iq];
      //if(SlumpLength<2 || iq*2<Qn-0) fi_sh[indsh0+0] = fromGlobR.x;
      //if(SlumpLength<1 || iq*2<Qn-1) fi_sh[indsh0+1] = fromGlobR.y;
    }
    /*__syncthreads();
    const int Nc4blk=Nwarps*warpSize/Qn;
    for(int indiq=thid,iter=0; iter<(NfaceCells-1)/Nc4blk+1; indiq+=Nc4blk*Qn, iter++) {
      if( thid>=Nc4blk*Qn || indiq>=NfaceCells*Qn ) break;
      const int iq = thid%Qn;
      const int ind = indiq/Qn;
      const short ind_sh = indexes_loc[ind];
      const int2 glob_indLR = indexes_glob[ind];
      const int glob_indL = glob_indLR.x;
      const int glob_indR = glob_indLR.y;
      pars.data.tiles[glob_indL].f[iq] = fi_sh[SHMEM_INDEX(iq, ind_sh)];
      fi_sh[SHMEM_INDEX(iq, ind_sh)] = pars.data.tiles[glob_indR].f[iq];
    }
    __syncthreads(); if(thid==0) time_counter.lap();*/
    return;
  }

  const int Nc4warp = (NfaceCells+Nwarps-1)/Nwarps;
  static_assert( Nc4warp <= warpSize );
  const bool canCollectTwoValsPerWarp = (Nc4warp<=warpSize/2)?1:0;
  const int hWarpSz = (canCollectTwoValsPerWarp)?(warpSize/2):(warpSize);

  const int index_seq_loop = warp_id*Nc4warp+lane_id%hWarpSz; static_assert( Nc4warp <= hWarpSz );
  const int index_seq_warp = warp_id+(lane_id%hWarpSz)*Nwarps; static_assert( NfaceCells <= Nwarps*hWarpSz );
  int index=0;
  if(PreCalcArray==UseIndArrSuccessively) index = index_seq_loop; else if(PreCalcArray==UseIndArrSparsely) index = index_seq_warp;
  const int3 icrd = get_ixyz_from_index(index);
  int exch_ind_sh = (icrd.x+shift)%Nbsz + (icrd.y+shift)%Nbsz*Nbsz + (icrd.z+shift)%Nbsz*Nbsz*Nbsz; 
  int exch_glob_indLR;
  int exch_glob_indL, exch_glob_indR;
  if(canCollectTwoValsPerWarp) {
    const int3 cycsh=make_int3( icrd.x?0:Nbsz, icrd.y?0:Nbsz, icrd.z?0:Nbsz )*(lane_id>=warpSize/2);
    int exch_gixLR = ( crd0.x+cycsh.x+icrd.x )%Nx;
    int exch_giyLR = ( crd0.y+cycsh.y+icrd.y )%Ny;
    int exch_gizLR = ( crd0.z+cycsh.z+icrd.z )%Nz;
    exch_glob_indLR = exch_gixLR + exch_giyLR*Nx + exch_gizLR*Nx*Ny;
  } else{
    const int3 cycsh=make_int3( icrd.x?0:Nbsz, icrd.y?0:Nbsz, icrd.z?0:Nbsz );
    int exch_gixL = ( crd0.x+icrd.x )%Nx;
    int exch_giyL = ( crd0.y+icrd.y )%Ny;
    int exch_gizL = ( crd0.z+icrd.z )%Nz;
    int exch_gixR = ( crd0.x+cycsh.x+icrd.x )%Nx;
    int exch_giyR = ( crd0.y+cycsh.y+icrd.y )%Ny;
    int exch_gizR = ( crd0.z+cycsh.z+icrd.z )%Nz;
    exch_glob_indL = exch_gixL + exch_giyL*Nx + exch_gizL*Nx*Ny;
    exch_glob_indR = exch_gixR + exch_giyR*Nx + exch_gizR*Nx*Ny;
  }
  if(PreCalcArray==UseIndArrSuccessively) {
      for(int indiq=warp_id*Nc4warp*Qn, iface=0; indiq<min((warp_id+1)*Nc4warp,NfaceCells)*Qn; indiq+=warpSize) {
        const int ind=(indiq+lane_id)/Qn;
        const int iq=(indiq+lane_id)%Qn;
        const int ilane  = ind-warp_id*Nc4warp;

        const int ind_sh    = __shfl_sync(0xffffffff, exch_ind_sh   , ilane);
        int glob_indL,glob_indR;
        if(canCollectTwoValsPerWarp) {
          glob_indL = __shfl_sync(0xffffffff, exch_glob_indLR, ilane);
          glob_indR = __shfl_sync(0xffffffff, exch_glob_indLR, ilane+warpSize/2);
        } else {
          glob_indL = __shfl_sync(0xffffffff, exch_glob_indL, ilane);
          glob_indR = __shfl_sync(0xffffffff, exch_glob_indR, ilane);
        }
        if( indiq+lane_id >= min( (warp_id+1)*Nc4warp,NfaceCells)*Qn ) break;

        pars.data.tiles[glob_indL].f[iq] = fi_sh[SHMEM_INDEX(iq, ind_sh)];
        fi_sh[SHMEM_INDEX(iq, ind_sh)] = pars.data.tiles[glob_indR].f[iq];
      }
  } else if(PreCalcArray==UseIndArrSparsely) {
      for(int ind=warp_id, ilane=0, iface=0; ind<NfaceCells; ind+=Nwarps, ilane++) {
        const int ind_sh    = __shfl_sync(0xffffffff, exch_ind_sh   , ilane);
        int glob_indL,glob_indR;
        if(canCollectTwoValsPerWarp) {
          glob_indL = __shfl_sync(0xffffffff, exch_glob_indLR, ilane);
          glob_indR = __shfl_sync(0xffffffff, exch_glob_indLR, ilane+warpSize/2);
        } else {
          glob_indL = __shfl_sync(0xffffffff, exch_glob_indL, ilane);
          glob_indR = __shfl_sync(0xffffffff, exch_glob_indR, ilane);
        }
        if(iq_l<Qn) {
          const int iq=iq_l;
          pars.data.tiles[glob_indL].f[iq] = fi_sh[SHMEM_INDEX(iq, ind_sh)];
          fi_sh[SHMEM_INDEX(iq, ind_sh)] = pars.data.tiles[glob_indR].f[iq];
        }
      }
  } else if(PreCalcArray==UseAllThreadsSuccessively){
    const int Nc4blk=Nwarps*warpSize/Qn;
    const int index1 = warp_id*warpSize/Qn+Nc4blk*(lane_id%warpSize);
    const int index2 = index1+1;
    const int index3 = index1+2;
    int3 icrd[3]; 
    int indexxx[3] = {index1,index2,index3};
    icrd[0] = get_ixyz_from_index(index1);
    icrd[1] = get_ixyz_from_index(index2);
    icrd[2] = get_ixyz_from_index(index3);
    int exch_ind_sh[3]; for(int i=0;i<3; i++) exch_ind_sh[i] = (icrd[i].x+shift)%Nbsz + (icrd[i].y+shift)%Nbsz*Nbsz + (icrd[i].z+shift)%Nbsz*Nbsz*Nbsz; 
    int exch_glob_indLR;
    int exch_glob_indL[3], exch_glob_indR[3];
    int3 cycsh[3]; for(int i=0;i<3;i++) cycsh[i]=make_int3( icrd[i].x?0:Nbsz, icrd[i].y?0:Nbsz, icrd[i].z?0:Nbsz );
    int exch_gixL[3],exch_gixR[3];
    for(int i=0;i<3; i++) exch_glob_indL[i] = (crd0.x+icrd[i].x)%Nx + (crd0.y+icrd[i].y)%Ny*Nx + (crd0.z+icrd[i].z)%Nz*Nx*Ny;
    for(int i=0;i<3; i++) exch_glob_indR[i] = (crd0.x+cycsh[i].x+icrd[i].x)%Nx + (crd0.y+cycsh[i].y+icrd[i].y)%Ny*Nx + (crd0.z+cycsh[i].z+icrd[i].z)%Nz*Nx*Ny;
    //for(int i=0;i<3; i++) exch_glob_indL[i] = icrd[i].x;//)%Nx + (crd0.y+icrd[i].y)%Ny*Nx + (crd0.z+icrd[i].z)%Nz*Nx*Ny;
    //for(int i=0;i<3; i++) exch_glob_indR[i] = icrd[i].y;
    //for(int i=0;i<3; i++) exch_ind_sh[i] = indexxx[i];//icrd[i].z;
    for(int indiq=thid,iter=0; iter<(NfaceCells-1)/Nc4blk+1; indiq+=Nc4blk*Qn, iter++) {
      int fetch_ic=0;
      const int overdiff = thid%warpSize-thid%Qn;
      if(overdiff<=0) fetch_ic=0; else if(overdiff>0 && overdiff<=Qn) fetch_ic=1; else fetch_ic=2;
      const int ilane  = iter;

      int ind_sh_all[3];
      ind_sh_all[0] = __shfl_sync(0xffffffff, exch_ind_sh[0], ilane);
      ind_sh_all[1] = __shfl_sync(0xffffffff, exch_ind_sh[1], ilane);
      ind_sh_all[2] = __shfl_sync(0xffffffff, exch_ind_sh[2], ilane);
      const int ind_sh = ind_sh_all[fetch_ic];
      int glob_indL,glob_indR;
      int glob_indL_all[3],glob_indR_all[3];
      /*if(canCollectTwoValsPerWarp) {
        glob_indL = __shfl_sync(0xffffffff, exch_glob_indLR, ilane);
        glob_indR = __shfl_sync(0xffffffff, exch_glob_indLR, ilane+warpSize/2);
      } else*/ {
          glob_indL_all[0] = __shfl_sync(0xffffffff, exch_glob_indL[0], ilane);
          glob_indL_all[1] = __shfl_sync(0xffffffff, exch_glob_indL[1], ilane);
          glob_indL_all[2] = __shfl_sync(0xffffffff, exch_glob_indL[2], ilane);
          glob_indR_all[0] = __shfl_sync(0xffffffff, exch_glob_indR[0], ilane);
          glob_indR_all[1] = __shfl_sync(0xffffffff, exch_glob_indR[1], ilane);
          glob_indR_all[2] = __shfl_sync(0xffffffff, exch_glob_indR[2], ilane);
          glob_indL = glob_indL_all[fetch_ic];
          glob_indR = glob_indR_all[fetch_ic];
        }
      if( thid>=Nc4blk*Qn || indiq>=NfaceCells*Qn ) break;
      const int iq = thid%Qn;

      //if(blockIdx.x==0 && crd0.x==0 && crd0.y==0 && crd0.z==0) 
      //  printf("indiq %d iter %d iq %d ix iy iz = %d %d %d fetch_ic=%d thid=%d index1=%d lane_id=%d\n", indiq, iter, iq, glob_indL, glob_indR,ind_sh, fetch_ic, thid, index1, lane_id);

      pars.data.tiles[glob_indL].f[iq] = fi_sh[SHMEM_INDEX(iq, ind_sh)];
      fi_sh[SHMEM_INDEX(iq, ind_sh)] = pars.data.tiles[glob_indR].f[iq];
    }

  }
  #else
  //if(iq_l>=Qn) return;
  const int Nc4blk=Nwarps*warpSize/Qn;
  for(int indiq=thid,iter=0; iter<(NfaceCells-1)/Nc4blk+1; indiq+=Nc4blk*Qn, iter++) {
    if( thid>=Nc4blk*Qn || indiq>=NfaceCells*Qn ) break;
    const int iq = thid%Qn;
    const int ind = indiq/Qn;
    int iface=0;
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
    const int glob_indL = gixL+giyL*Nx+gizL*Nx*Ny;
    const int glob_indR = gixR+giyR*Nx+gizR*Nx*Ny;
    pars.data.tiles[glob_indL].f[iq] = fi_sh[SHMEM_INDEX(iq, ind_sh)];
    fi_sh[SHMEM_INDEX(iq, ind_sh)] = pars.data.tiles[glob_indR].f[iq];
    /*if(iq==0 && ix==0 && iy==Nbsz/2 && iz==Nbsz/2 && blockIdx.x==0)  printf(
        "shift=%d save at %d %d %d load %d %d %d from value from %d = %g\n",
        shift, gixL,giyL,gizL, gixR,giyR,gizR, ind_sh*Qn+iq, fi_sh[SHMEM_INDEX(iq, ind_sh)]);*/
  }
  #endif
}
/*__device__ inline void load_store_cells_perimeter_fine_tuning(const int3 crd0, ftype* fi_sh, const int shift){
  const int Nbsz=CompStep::Nbsz;
  static_assert(Nbsz==8);
  const int thid = threadIdx.x + threadIdx.y*CompStep::Nb.x + threadIdx.z*CompStep::Nb.x*CompStep::Nb.y;
  const int Nbs = CompStep::Nb.x*CompStep::Nb.y*CompStep::Nb.z;
  const int warpSize=32;
  const int Nwarps = Nbs/32;
  const int warp_id = thid/32;
  const int lane_id = thid%32;
  if(warp_id>=Nwarps) return;
  const int iq_l=lane_id;
  const int Qn = Cell::Qn;

  const int ind_b1 = Nbsz*Nbsz;
  const int ind_b2 = 2*Nbsz*Nbsz-Nbsz;
  const int NfaceCells = 3*Nbsz*Nbsz-3*Nbsz+1;

  const int Nc4warp = 10;
  int index = warp_id*Nc4warp+lane_id%(warpSize/2); static_assert( Nc4warp <= warpSize/2 );
  if(lane_id%(warpSize/2)==10) index = 160+warp_id;
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
  for(int indiq=warp_id*Nc4warp*Qn, iface=0; indiq<min((warp_id+1)*Nc4warp,NfaceCells)*Qn; indiq+=warpSize) {
    const int ind=(indiq+lane_id)/Qn;
    const int iq=(indiq+lane_id)%Qn;
    const int ilane  = ind-warp_id*Nc4warp;

    const int glob_indL = __shfl_sync(0xffffffff, exch_glob_indLR, ilane);
    const int glob_indR = __shfl_sync(0xffffffff, exch_glob_indLR, ilane+warpSize/2);
    const int ind_sh    = __shfl_sync(0xffffffff, exch_ind_sh   , ilane);
    if( indiq+lane_id >= min( (warp_id+1)*Nc4warp,NfaceCells)*Qn ) break;

    pars.data.tiles[glob_indL].f[iq] = fi_sh[SHMEM_INDEX(iq, ind_sh)];
    fi_sh[SHMEM_INDEX(iq, ind_sh)] = pars.data.tiles[glob_indR].f[iq];
  }
  if(warp_id<9) {
    const int ind=160+warp_id;
    const int iq=lane_id%Qn;
    const int ilane  = 10;

    const int glob_indL = __shfl_sync(0xffffffff, exch_glob_indLR, ilane);
    const int glob_indR = __shfl_sync(0xffffffff, exch_glob_indLR, ilane+warpSize/2);
    const int ind_sh    = __shfl_sync(0xffffffff, exch_ind_sh   , ilane);
    if(iq<Qn){
      pars.data.tiles[glob_indL].f[iq] = fi_sh[SHMEM_INDEX(iq, ind_sh)];
      fi_sh[SHMEM_INDEX(iq, ind_sh)] = pars.data.tiles[glob_indR].f[iq];
    }
  }
}*/

__device__ inline void coneFoldLoop(int3 gCrd, ftype* fi_sh){
  const int Nbsz=CompStep::Nbsz;
  const int Qn=Cell::Qn;
  const int thid = threadIdx.x + threadIdx.y*CompStep::Nb.x + threadIdx.z*CompStep::Nb.x*CompStep::Nb.y;
  //const short thid = threadIdx.x;
  const short Nbs = CompStep::Nb.x*CompStep::Nb.y*CompStep::Nb.z;
  if(thid>=CompStep::Nblk) return;
  /*short sh_index_p[Qn*2];
  for(int itt=0;itt<2;itt++) for(int iq=0; iq<Cell::Qn; iq++) {
    const char3 pbits3 = make_char3(crd_loc.x&1,crd_loc.y&1,crd_loc.z&1)^itt;
    const char pbits = pbits3.x|pbits3.y<<1|pbits3.z<<2;
    const char4 shifts = LBMconsts::compact_access[iq+(Cell::Qn)*pbits];
    char3 shCrd = make_char3( (crd_loc.x+shifts.x)%Nbsz, (crd_loc.y+shifts.y)%Nbsz, (crd_loc.z+shifts.z)%Nbsz );
    const short shid = shCrd.x+shCrd.y*Nbsz + shCrd.z*Nbsz*Nbsz;
    const short rev_iq = shifts.w;
    sh_index_p[itt*Qn+iq] =  SHMEM_INDEX(rev_iq, shid);
  }*/
  for(int it=0; it<pars.Nt; it++, gCrd+=make_int3(1) ){
    const char shift = it%Nbsz;
    if(gCrd.x>=Nx) gCrd.x-=Nx;
    if(gCrd.y>=Ny) gCrd.y-=Ny;
    if(gCrd.z>=Nz) gCrd.z-=Nz;
    Cell c;

    //for(int ind_cf=thid; ind_cf<Nbs; ind_cf+=blockDim.x*blockDim.y*blockDim.z) {
    const int Ncells4thrd = (Nbs+CompStep::Nblk-1)/CompStep::Nblk;
    for(int icell=0; icell<Ncells4thrd; icell++) {
      const int ind_cf = thid + icell*CompStep::Nblk;
      if( Nbs%CompStep::Nblk!=0 ) if( ind_cf >= CompStep::Nb.x*CompStep::Nb.y*CompStep::Nb.z) break;
      const short3 crd_loc = make_short3( ind_cf%CompStep::Nb.x, ind_cf/CompStep::Nb.x%CompStep::Nb.y, ind_cf/(CompStep::Nb.x*CompStep::Nb.y) );

      //#define UPDATE_USUAL
      #define USE_CHOOSE_FROM_8_SHIFTS 
      //#define SH_DATA_FIXED
      #ifdef SH_DATA_FIXED
      for(int iq=0; iq<Cell::Qn; iq++) c.f[iq] = fi_sh[ sh_index_p[(it&1)*Qn + iq] ];
      c.collision();
      for(int iq=0; iq<Cell::Qn; iq++) fi_sh[ sh_index_p[(it&1)*Qn + iq] ] = c.f[iq];
      #elif defined USE_CHOOSE_FROM_8_SHIFTS 
      const char3 pbits3 = make_char3(crd_loc.x&1,crd_loc.y&1,crd_loc.z&1);
      const char pbits = pbits3.x|pbits3.y<<1|pbits3.z<<2;
      short sh_index[Qn];
      short shids[8];
      //#pragma unroll 8
      for(int ic=0; ic<8; ic++) {
        const char3 shifts = make_char3(ic&1, (ic>>1)&1, (ic>>2)&1) -pbits3;
        const char3 shCrd = make_char3( (crd_loc.x+shift+shifts.x)%Nbsz, (crd_loc.y+shift+shifts.y)%Nbsz, (crd_loc.z+shift+shifts.z)%Nbsz );
        shids[ic] = shCrd.x+shCrd.y*Nbsz + shCrd.z*Nbsz*Nbsz;
      }
      const char3 shCrd_cntr = make_char3( (crd_loc.x+shift)%Nbsz, (crd_loc.y+shift)%Nbsz, (crd_loc.z+shift)%Nbsz );
      const short shid_cntr = shCrd_cntr.x+shCrd_cntr.y*Nbsz + shCrd_cntr.z*Nbsz*Nbsz;
      const int3 globCrd_cntr = make_int3( (crd_loc.x+gCrd.x)%Nx, (crd_loc.y+gCrd.y)%Ny, (crd_loc.z+gCrd.z)%Nz );
	    bool maybeBC=0;
      if( globCrd_cntr.x==0 && pbits3.x==1 || globCrd_cntr.x==Nx-1 && pbits3.x==0 ||
          globCrd_cntr.y==0 && pbits3.y==1 || globCrd_cntr.y==Ny-1 && pbits3.y==0 ||
          globCrd_cntr.z==0 && pbits3.z==1 || globCrd_cntr.z==Nz-1 && pbits3.z==0
      ) maybeBC=1;
      for(int iq=0; iq<Cell::Qn; iq++) {
        auto ncell_niq = LBMconsts::compact_fill_cells[iq+(Cell::Qn)*pbits];
        const short shid = shids[ncell_niq.x];
        const short rev_iq = ncell_niq.y;
        sh_index[iq] =  SHMEM_INDEX(rev_iq, shid);

	    //---BounceBack BC
    	if(maybeBC)
        if( globCrd_cntr.x==0 && pbits3.x==1 && ( ncell_niq.x    &1)==0 || globCrd_cntr.x==Nx-1 && pbits3.x==0 && ( ncell_niq.x    &1)==1 ||
            globCrd_cntr.y==0 && pbits3.y==1 && ((ncell_niq.x>>1)&1)==0 || globCrd_cntr.y==Ny-1 && pbits3.y==0 && ((ncell_niq.x>>1)&1)==1 ||
            globCrd_cntr.z==0 && pbits3.z==1 && ((ncell_niq.x>>2)&1)==0 || globCrd_cntr.z==Nz-1 && pbits3.z==0 && ((ncell_niq.x>>2)&1)==1
          )  sh_index[iq] = SHMEM_INDEX(iq, shid_cntr);

        c.f[iq] = fi_sh[ sh_index[iq] ];
      }
      c.collision();
      for(int iq=0; iq<Cell::Qn; iq++) fi_sh[ sh_index[iq] ] = c.f[iq];
      #elif defined UPDATE_USUAL
      //-----------Old variant------------//
      const char3 pbits3 = make_char3(crd_loc.x&1,crd_loc.y&1,crd_loc.z&1);
      const char pbits = pbits3.x|pbits3.y<<1|pbits3.z<<2;
      short sh_index[Qn];
      for(int iq=0; iq<Cell::Qn; iq++) {
        const char4 shifts = LBMconsts::compact_access[iq+(Cell::Qn)*pbits];
        const char3 shCrd = make_char3( (crd_loc.x+shift+shifts.x)%Nbsz, (crd_loc.y+shift+shifts.y)%Nbsz, (crd_loc.z+shift+shifts.z)%Nbsz );
        const short shid = shCrd.x+shCrd.y*Nbsz + shCrd.z*Nbsz*Nbsz;
        const short rev_iq = shifts.w;
        sh_index[iq] =  SHMEM_INDEX(rev_iq, shid);

        //---BounceBack BC
        const char3 shCrd_cntr = make_char3( (crd_loc.x+shift)%Nbsz, (crd_loc.y+shift)%Nbsz, (crd_loc.z+shift)%Nbsz );
        const short shid_cntr = shCrd_cntr.x+shCrd_cntr.y*Nbsz + shCrd_cntr.z*Nbsz*Nbsz;
        const int3 globCrd_cntr = make_int3( (crd_loc.x+gCrd.x)%Nx, (crd_loc.y+gCrd.y)%Ny, (crd_loc.z+gCrd.z)%Nz );
        //const int3 globCrd_near = make_int3( (crd_loc.x+gCrd.x+shifts.x)%Nx, (crd_loc.y+gCrd.y+shifts.y)%Ny, (crd_loc.z+gCrd.z+shifts.z)%Nz );
        if( globCrd_cntr.x==0 && shifts.x<0 || globCrd_cntr.x==Nx-1 && shifts.x>0 ||
            globCrd_cntr.y==0 && shifts.y<0 || globCrd_cntr.y==Ny-1 && shifts.y>0 ||
            globCrd_cntr.z==0 && shifts.z<0 || globCrd_cntr.z==Nz-1 && shifts.z>0
          )  sh_index[iq] = SHMEM_INDEX(iq, shid_cntr);

        c.f[iq] = fi_sh[ sh_index[iq] ];
      }

      //for(int iq=0; iq<Cell::Qn; iq++) c.f[iq]++;
      c.collision();

      for(int iq=0; iq<Cell::Qn; iq++) fi_sh[ sh_index[iq] ] = c.f[iq];
      //for(int iq=0; iq<Cell::Qn; iq++) fi_sh[ SHMEM_INDEX(iq,thid) ] = c.f[iq];
      #endif

    }

    __syncthreads();
    load_store_cells_perimeter(gCrd, fi_sh, shift);
    //load_store_cells_perimeter_fine_tuning(gCrd, fi_sh, shift);
   __syncthreads();
  }
}

__global__ __launch_bounds__(CompStep::Nblk) void compactStepConeFold(int ix_base){
  const int parity=0;
  const int Nbsz = CompStep::Nbsz;
  const int Nbs = CompStep::Nb.x*CompStep::Nb.y*CompStep::Nb.z;
  const int ind1s = blockIdx.x%(Ny/Nbsz)*Nbsz;
  const int ind2s = blockIdx.x/(Ny/Nbsz)*Nbsz;
  const int ix0 = ix_base          +ind1s+ind2s;
  const int iy0 = Ny-CompStep::Nb.y-ind1s;
  const int iz0 = Nz-CompStep::Nb.z-ind2s;

  const int thid = threadIdx.x + threadIdx.y*CompStep::Nb.x + threadIdx.z*CompStep::Nb.x*CompStep::Nb.y;

  const int Qn = Cell::Qn;

  extern __shared__ ftype fi_sh[];
  load_store_datablock(thid, make_int3(ix0,iy0,iz0), fi_sh, RW::Load);
  __syncthreads();

  coneFoldLoop(make_int3(ix0,iy0,iz0), fi_sh);

  const int shift = pars.Nt%CompStep::Nbsz;

  __syncthreads();
  load_store_datablock(thid, make_int3(ix0,iy0,iz0)+make_int3(pars.Nt), fi_sh, RW::Store, shift);

  #if defined DRAW_WAVEFRONT && not defined NOGL
  __syncthreads();
  for(int ith=thid; ith<Nbs; ith+=blockDim.x*blockDim.y*blockDim.z) save_image_arr( make_int3(ix0,iy0,iz0)+make_int3(pars.Nt), ith );
  #endif
}
