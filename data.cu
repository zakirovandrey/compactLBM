#include "data.cuh"


__host__ __device__ void Cell::updateRhoVel(){
  using namespace LBMconsts;
  vel = make_ftype3(0,0,0); rho=0;
  for(int iq=0; iq<Qn; iq++) { rho+=f[iq]; vel+= make_ftype3(e[iq])*f[iq]; }
  if(rho!=0) vel/=rho;
}

void Data_t::malloc_data(const int Nx, const int Ny, const int Nz){
  const int Ns3 = Tile::Ns*Tile::Ns*Tile::Ns; 
  const size_t sz = long(Nx)*Ny*Nz/Ns3*sizeof(Tile);
  printf("Total data size = %g GB\n",double(sz)/1024/1024/1024); 
  CHECK_ERROR( cudaMalloc((void**)&tiles, sz ) );
  CHECK_ERROR( cudaMallocHost((void**)&tilesHost, sz ) );
  CHECK_ERROR( cudaMemset(tiles, 0, sz ) );
  CHECK_ERROR( cudaMemset(tilesHost, 0, sz ) );
};
void Data_t::copyHost2Dev(){
  const int Ns3 = Tile::Ns*Tile::Ns*Tile::Ns; 
  const size_t sz = long(Nx)*Ny*Nz/Ns3*sizeof(Tile);
  CHECK_ERROR( cudaMemcpy(tiles, tilesHost, sz, cudaMemcpyHostToDevice ) );
}
void Data_t::copyDev2Host(){
  const int Ns3 = Tile::Ns*Tile::Ns*Tile::Ns; 
  const size_t sz = long(Nx)*Ny*Nz/Ns3*sizeof(Tile);
  CHECK_ERROR( cudaMemcpy(tiles, tilesHost, sz, cudaMemcpyDeviceToHost ) );
}


