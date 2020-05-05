#include "data.cuh"
namespace CompStep{
  static constexpr uint Nbsz = 8;
  static constexpr uint3 Nb = (const uint3){Nbsz,Nbsz,Nbsz};
  static constexpr uint Nblk = Nb.x*Nb.y*Nb.z;
};

enum RW {Load,Store};

template<int parity>__global__ void compactStep();
__global__ void compactStepConeFold(const int ix_base);

__device__ inline static void load_store_datablock(const int3 crd0, ftype* fi_sh, int ls, int shift=0){
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
      const int ix_loc = ind%CompStep::Nb.x;
      const int iy_loc = ind/CompStep::Nb.x%CompStep::Nb.y;
      const int iz_loc = ind/(CompStep::Nb.x*CompStep::Nb.y);
      const int ind_sh = (ix_loc+shift)%CompStep::Nb.x 
                       + (iy_loc+shift)%CompStep::Nb.y*CompStep::Nb.x
                       + (iz_loc+shift)%CompStep::Nb.z*CompStep::Nb.x*CompStep::Nb.y;
      const int gix = ( crd0.x+ix_loc )%Nx;
      const int giy = ( crd0.y+iy_loc )%Ny;
      const int giz = ( crd0.z+iz_loc )%Nz;
      const int iq=iq_l;
      if(ls==RW::Load ) fi_sh[ind_sh*Qn+iq] = pars.data.tiles[gix+giy*Nx+giz*Nx*Ny].f[iq];
      if(ls==RW::Store) pars.data.tiles[gix+giy*Nx+giz*Nx*Ny].f[iq] = fi_sh[ind_sh*Qn+iq];
    }
  }
}

__forceinline__ __device__ void Cell::collision(){
  fast_collision(); return;
  using namespace LBMconsts;
  //rho=0; for(int i=0; i<LBMconsts::Qn; i++) rho+= f[i];
  //for(int i=0; i<Qn; i++) f[i] = w_c[i]*rho; return;
  ftype4 Vrho=make_ftype4(0,0,0,0);
  for(int i=0; i<Qn; i++) Vrho+= f[i]*make_ftype4(e[i].x,e[i].y,e[i].z,1);
  register ftype feq[Qn];
  calcEq(feq, Vrho);
  register ftype _f[Qn];
  const ftype dtau = PPdev.dtau;
  for(int i=0; i<Qn; i++) _f[i] = f[i]-dtau*(f[i]-feq[i]);
  for(int i=0; i<Qn; i++) f[i] = _f[reverseXYZ[i]];
}

__forceinline__ __device__ void Cell::fast_collision(){
  using namespace LBMconsts;
  rho=0; 
  register ftype3 u3=make_ftype3(0,0,0); 
  for(int i=0; i<Qn; i++) rho+= f[i];
  u3.x = E_MATRIX_X(f[0],f[1],f[2],f[3],f[4],f[5],f[6],f[7],f[8],f[9],f[10],f[11],f[12],f[13],f[14],f[15],f[16],f[17],f[18]);
  u3.y = E_MATRIX_Y(f[0],f[1],f[2],f[3],f[4],f[5],f[6],f[7],f[8],f[9],f[10],f[11],f[12],f[13],f[14],f[15],f[16],f[17],f[18]);
  u3.z = E_MATRIX_Z(f[0],f[1],f[2],f[3],f[4],f[5],f[6],f[7],f[8],f[9],f[10],f[11],f[12],f[13],f[14],f[15],f[16],f[17],f[18]);
//   if(rho!=0) u3/=rho;
//  if(rho!=0) { u3.x=__fdividef(u3.x,rho); u3.y=__fdividef(u3.y,rho); u3.z=__fdividef(u3.z,rho); }
  if(rho!=0) { float drho=__fdividef(ftype(1),rho); u3.x=u3.x*drho; u3.y=u3.y*drho; u3.z=u3.z*drho; }
  register ftype feq,frev,eu;
  const register ftype mxw0 = ftype(1) - dot(u3,u3)*ftype(0.5)*dcs2;
  const ftype dtau = PPdev.dtau;
  feq = W0*rho*mxw0; f[0]+= (feq-f[0])*dtau;
  feq = W1*rho*( u3.x*(u3.x*dcs4/2 - dcs2) + mxw0 ); frev  = (feq-f[1] )*dtau+f[1] ;
  feq = W1*rho*( u3.x*(u3.x*dcs4/2 + dcs2) + mxw0 ); f[1]  = (feq-f[2] )*dtau+f[2] ; f[2]=frev;
  feq = W1*rho*( u3.y*(u3.y*dcs4/2 - dcs2) + mxw0 ); frev  = (feq-f[4] )*dtau+f[4] ;
  feq = W1*rho*( u3.y*(u3.y*dcs4/2 + dcs2) + mxw0 ); f[4]  = (feq-f[7] )*dtau+f[7] ; f[7]=frev;
  feq = W1*rho*( u3.z*(u3.z*dcs4/2 - dcs2) + mxw0 ); frev  = (feq-f[10])*dtau+f[10];
  feq = W1*rho*( u3.z*(u3.z*dcs4/2 + dcs2) + mxw0 ); f[10] = (feq-f[13])*dtau+f[13]; f[13]=frev;
  eu=u3.x+u3.y; feq = W2*rho*( eu*(eu*dcs4/2 - dcs2) + mxw0 ); frev  = (feq-f[3 ])*dtau+f[3] ;
  eu=u3.x+u3.y; feq = W2*rho*( eu*(eu*dcs4/2 + dcs2) + mxw0 ); f[3]  = (feq-f[8 ])*dtau+f[8] ; f[8]=frev;
  eu=u3.x-u3.y; feq = W2*rho*( eu*(eu*dcs4/2 + dcs2) + mxw0 ); frev  = (feq-f[5 ])*dtau+f[5] ;
  eu=u3.x-u3.y; feq = W2*rho*( eu*(eu*dcs4/2 - dcs2) + mxw0 ); f[5]  = (feq-f[6 ])*dtau+f[6] ; f[6]=frev;
  eu=u3.x+u3.z; feq = W2*rho*( eu*(eu*dcs4/2 - dcs2) + mxw0 ); frev  = (feq-f[9 ])*dtau+f[9] ;
  eu=u3.x+u3.z; feq = W2*rho*( eu*(eu*dcs4/2 + dcs2) + mxw0 ); f[9]  = (feq-f[14])*dtau+f[14]; f[14]=frev;
  eu=u3.x-u3.z; feq = W2*rho*( eu*(eu*dcs4/2 + dcs2) + mxw0 ); frev  = (feq-f[11])*dtau+f[11] ;
  eu=u3.x-u3.z; feq = W2*rho*( eu*(eu*dcs4/2 - dcs2) + mxw0 ); f[11] = (feq-f[12])*dtau+f[12]; f[12]=frev;
  eu=u3.y+u3.z; feq = W2*rho*( eu*(eu*dcs4/2 - dcs2) + mxw0 ); frev  = (feq-f[15])*dtau+f[15] ;
  eu=u3.y+u3.z; feq = W2*rho*( eu*(eu*dcs4/2 + dcs2) + mxw0 ); f[15] = (feq-f[18])*dtau+f[18]; f[18]=frev;
  eu=u3.y-u3.z; feq = W2*rho*( eu*(eu*dcs4/2 + dcs2) + mxw0 ); frev  = (feq-f[16])*dtau+f[16] ;
  eu=u3.y-u3.z; feq = W2*rho*( eu*(eu*dcs4/2 - dcs2) + mxw0 ); f[16] = (feq-f[17])*dtau+f[17]; f[17]=frev;
}


