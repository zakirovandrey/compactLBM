#pragma once
#include "LBMconsts.cuh"
struct Cell{
  static const int Qn=LBMconsts::Qn;
  ftype f[Qn];
  ftype rho;
  ftype3 vel;
  __host__ __device__ static void calcEq(ftype feq[Qn], const ftype4 Vrho);
  __host__ __device__ void updateRhoVel();
  __device__ void collision();
  __device__ void fast_collision();
  __host__ __device__ void operator=(const ftype v){
    for(int iq=0;iq<Qn;iq++) f[iq]=v;
  }
};

template<int _Ns> struct Tile_t{
  static const int Ns=_Ns;
    static const int Qn4 = (Cell::Qn-1)/4+1;
    static const int Qn2 = (Cell::Qn-1)/2+1;
  union {
    ftype f[Cell::Qn*Ns*Ns*Ns];
    ftype4 f4[Qn4*Ns*Ns*Ns];
    ftype2 f2[Qn2*Ns*Ns*Ns];
  };
  //char slump[ 32 - sizeof(f)/sizeof(ftype) % 32];
  __host__ __device__ Cell construct_cell(const int3 loc_crd) {
    const int Ns3 = Ns*Ns*Ns; 
    Cell c;
    for(int iq=0; iq<Cell::Qn; iq++) c.f[iq]=f[iq + (loc_crd.x+loc_crd.y*Ns+loc_crd.z*Ns*Ns)*Ns3 ];
    return c;
  }
};

//typedef Tile_t<2> Tile;
typedef Tile_t<1> Tile;

struct Data_t{
  Tile* tiles;
  Tile* tilesHost;
  int* it_arr;
  __host__  __device__
  Cell get_cell(const int ix, const int iy, const int iz) {
    Tile* ctile = &tiles[ ix/Tile::Ns + iy/Tile::Ns*(Nx/Tile::Ns) + iz/Tile::Ns*(Nx/Tile::Ns)*(Ny/Tile::Ns) ];
    return ctile->construct_cell( make_int3(ix,iy,iz)%Tile::Ns ); 
  }
  template<int> inline __host__ __device__ Cell get_cell_compact(const int ix, const int iy, const int iz);
  template<int> inline __host__ __device__ void set_cell_compact(const Cell& c, const int ix, const int iy, const int iz);
  void malloc_data(const int Nx, const int Ny, const int Nz);
  void copyHost2Dev();
  void copyDev2Host();
};

#include "data-inl.cu"
