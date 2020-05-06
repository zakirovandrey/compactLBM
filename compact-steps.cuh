#include "data.cuh"
namespace CompStep{
  static constexpr uint Nbsz = 8;
  static constexpr uint3 Nb = (const uint3){Nbsz,Nbsz,Nbsz};
  static constexpr uint Nblk = Nb.x*Nb.y*Nb.z;
};

enum RW {Load,Store};

template<int parity>__global__ void compactStep();
__global__ void compactStepConeFold(const int ix_base);


