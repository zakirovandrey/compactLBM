__device__ inline ftype4 blank_mat(int ix, int iy, int iz){
  const ftype3 r = make_ftype3(ix-Nx/2, (iy-Ny/2), (iz-Nz/2));
  if(length(r)<10) return make_ftype4(0,0,0,2);
  else return make_ftype4(0,0,0,1);
};



