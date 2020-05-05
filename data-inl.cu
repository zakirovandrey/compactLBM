__host__ __device__ inline void Cell::calcEq(ftype feq[Qn], const ftype4 rhoV){
  using namespace LBMconsts;
  const ftype rho=rhoV.w;
  ftype3 u = make_ftype3(rhoV.x,rhoV.y,rhoV.z)/rho;
  if(rho==0) u = make_ftype3(0,0,0);
  const ftype mxw0 = ftype(1) - dot(u,u)*ftype(0.5)*dcs2;
  for(int i=0; i<Qn; i++) {
    ftype3 eidx = make_ftype3(e[i]);
    ftype eu =  dot(eidx,u)*dcs2;
    ftype mxw   = mxw0 + eu + eu*eu*ftype(0.5);
    feq[i] = w[i]*rho*mxw;
  }
}

//this function returns three bits assosiated with XYZ coordinates
//  representing following information:
//  0 -- the required fi with _dir_ direction can be aquired from the current cell
//  1 -- the required fi with _dir_ direction can be aquired from the neighboured cell
//                and fi is also reversed
//   For example, for pbits=011 and returned value 110 the desired fi = neigbour(X+1, Y-1, 0).f[reverseXY]
template<int pbits> inline int3 get_shifts_compact(const int3 dir){
  const int3 cnt = 1-2*make_int3(pbits&1, pbits>>1&1, pbits>>2&2);
  return make_int3((dir.x==cnt.x), (dir.y==cnt.y), (dir.z==cnt.z) );
  //return (dir.x==cnt.x)|(dir.y==cnt.y)<<1|(dir.z==cnt.z)<<2 ;
}

template<int parity> inline __host__ __device__ Cell Data_t::get_cell_compact(const int ix, const int iy, const int iz){
  Cell c;
  static_assert(Tile::Ns==1);
  const char3 pbits3 = make_char3(ix&1,iy&1,iz&1)^parity;
  const char pbits = pbits3.x|pbits3.y<<1|pbits3.z<<2;
  for(int iq=0; iq<Cell::Qn; iq++) {
    const char4 shifts = LBMconsts::compact_access[iq+(Cell::Qn)*pbits];
    //const char3 shift = (nsh>>make_char3(0,1,2)&1)*(1-2*pbits3);
    //if(nsh>>0&1) rev_iq = LBMconsts::reverseX[rev_iq];
    //if(nsh>>1&1) rev_iq = LBMconsts::reverseY[rev_iq];
    //if(nsh>>2&1) rev_iq = LBMconsts::reverseZ[rev_iq];
    const int3 gCrd_ = make_int3(ix+shifts.x, iy+shifts.y, iz+shifts.z);
    const int3 gCrd = make_int3((ix+shifts.x+Nx)%Nx, (iy+shifts.y+Ny)%Ny, (iz+shifts.z+Nz)%Nz);
    const int rev_iq = shifts.w;
    if(Tile::Ns==1) {
      c.f[iq] = tiles[ gCrd.x + gCrd.y*Nx+ gCrd.z*Nx*Ny ].f[rev_iq];
    } else{
      const Tile* ctile = &tiles[ gCrd.x/Tile::Ns + gCrd.y/Tile::Ns*(Nx/Tile::Ns) + gCrd.z/Tile::Ns*(Nx/Tile::Ns)*(Ny/Tile::Ns) ];
      const int3 intileCrd = gCrd%Tile::Ns;
      const int Ns3 = Tile::Ns*Tile::Ns*Tile::Ns; 
      c.f[iq] = ctile->f[rev_iq + (intileCrd.x+intileCrd.y*Tile::Ns+intileCrd.z*Tile::Ns*Tile::Ns)*Ns3 ];
    }
  }
  return c;
}
template<int parity> inline __host__ __device__ void Data_t::set_cell_compact(const Cell& c, const int ix, const int iy, const int iz){
  static_assert(Tile::Ns==1);
  const char3 pbits3 = make_char3(ix&1,iy&1,iz&1)^parity;
  const char pbits = pbits3.x|pbits3.y<<1|pbits3.z<<2;
  for(int iq=0; iq<Cell::Qn; iq++) {
    const char4 shifts = LBMconsts::compact_access[iq+(Cell::Qn)*pbits];
    const int3 gCrd_ = make_int3(ix+shifts.x, iy+shifts.y, iz+shifts.z);
    const int3 gCrd = make_int3((ix+shifts.x+Nx)%Nx, (iy+shifts.y+Ny)%Ny, (iz+shifts.z+Nz)%Nz);
    const int rev_iq = shifts.w;
    if(Tile::Ns==1) {
      tiles[ gCrd.x + gCrd.y*Nx+ gCrd.z*Nx*Ny ].f[rev_iq] = c.f[iq];
    } else{
      Tile* ctile = &tiles[ gCrd.x/Tile::Ns + gCrd.y/Tile::Ns*(Nx/Tile::Ns) + gCrd.z/Tile::Ns*(Nx/Tile::Ns)*(Ny/Tile::Ns) ];
      const int3 intileCrd = gCrd%Tile::Ns;
      const int Ns3 = Tile::Ns*Tile::Ns*Tile::Ns; 
      ctile->f[rev_iq + (intileCrd.x+intileCrd.y*Tile::Ns+intileCrd.z*Tile::Ns*Tile::Ns)*Ns3 ] = c.f[iq];
    }
  }
}
