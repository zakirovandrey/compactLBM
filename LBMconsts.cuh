#pragma once
#include "params.h"
#include <boost/preprocessor.hpp>
namespace LBMconsts {

enum {Void=1,Fluid=2,Iface=3,Solid=4,Box=5, I_fluid=6, I_void=7, notIface=8};
//enum {Void=1,Fluid=2,Iface=3,I_fluid=4,I_void=5,Solid=6,Box=7};
//static const unsigned IfaceAll = (1<<Iface)|(1<<I_fluid)|(1<<I_void); 
namespace Change{
    enum {I2F=1,I2V};
};

const ftype dx=1.0;
const ftype dy=1.0;
const ftype dz=1.0;
const ftype dt=1.0;

//#define D3Q27
#define D3Q19
// #define D1Q3
// #define D2Q9
// #define D2Q5

template<int B, int ...Btail> constexpr void debug_consts_assert() { static_assert(B, ""); }

#ifdef __CUDA_ARCH__
#define HOST_DEV_CHOOSE __device__
#else
#define HOST_DEV_CHOOSE constexpr
#endif

#define TO_SEQ_ELEM(z, n, text) text[n],

#define TO_SEQ_ELEM_CHECK(z, n, text) int(text.checkval[n]),
#define SEQ_LIST_COLLECTION_CHECK(arg, n) BOOST_PP_REPEAT(n, TO_SEQ_ELEM_CHECK, arg)
   #define SEQ_LIST_COLLECTION(arg, n) BOOST_PP_REPEAT(n, TO_SEQ_ELEM, arg)

#ifdef D1Q3
#define DIM 1
#define QN 3
#define QN_IN_DIM 6   // QN*2^DIM
constexpr const int3 _e[QN] = { { 0, 0, 0}, { 1, 0, 0}, { -1, 0, 0} };
const ftype W0=2./3;
const ftype W1=1./6;
const ftype W2=0;
const ftype W3=0;

#elif defined D3Q27
#define DIM 3
#define QN 27
#define QN_IN_DIM 216   // QN*2^DIM
constexpr const int3 _e[QN] = {
 /*0 -0 */ { 0, 0, 0},
 /*1 -7 */ { 1, 0, 0}, { 0, 1, 0}, { 0, 0, 1}, { 1, 1, 0}, { 1, 0, 1}, {0, 1, 1}, {1, 1, 1},
 /*8 -11*/ {-1, 0, 0}, {-1, 1, 0}, {-1, 0, 1}, {-1, 1, 1},
 /*12-15*/ { 0,-1, 0}, { 0,-1, 1}, { 1,-1, 0}, { 1,-1, 1}, 
 /*16-19*/ { 0, 0,-1}, { 1, 0,-1}, { 0, 1,-1}, { 1, 1,-1},
 /*20-21*/ {-1,-1, 0}, {-1,-1, 1},
 /*22-23*/ {-1, 0,-1}, {-1, 1,-1}, 
 /*24-25*/ { 0,-1,-1}, { 1,-1,-1},
 /*26-26*/ {-1,-1,-1}
};
const ftype W0=8./27;
const ftype W1=2./27;
const ftype W2=1./54;
const ftype W3=1./216;

#elif defined D3Q19
#define DIM 3
#define QN 19
#define QN_IN_DIM 152   // QN*2^DIM
constexpr const int3 _e[QN] = {
 { 0, 0, 0},
 { -1, 0, 0}, { 1, 0, 0}, 
 { -1,-1, 0}, { 0,-1, 0}, { 1,-1, 0}, {-1, 1, 0}, { 0, 1, 0}, { 1, 1, 0},
 { -1, 0,-1}, { 0, 0,-1}, { 1, 0,-1}, {-1, 0, 1}, { 0, 0, 1}, { 1, 0, 1}, 
 {  0,-1,-1},             { 0, 1,-1}, { 0,-1, 1},             { 0, 1, 1}
};
const ftype W0=1./3;
const ftype W1=1./18;
const ftype W2=1./36;
const ftype W3=0;
#define E_MATRIX_X(f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18) -f1 +f2-f3 +f5-f6   +f8-f9+f11-f12+f14
#define E_MATRIX_Y(f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18) -f3 -f4-f5+f6+f7+f8 -f15+f16-f17+f18
#define E_MATRIX_Z(f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18) -f9-f10-f11+f12+f13+f14-f15-f16+f17+f18

#elif defined D3Q13
#define DIM 3
#define QN 13
#define QN_IN_DIM 104   // QN*2^DIM
constexpr const int3 _e[QN] = {
 { 0, 0, 0},
 { -1,-1, 0}, { 1,-1, 0}, {-1, 1, 0}, { 1, 1, 0},
 { -1, 0,-1}, { 1, 0,-1}, {-1, 0, 1}, { 1, 0, 1}, 
 {  0,-1,-1}, { 0, 1,-1}, { 0,-1, 1}, { 0, 1, 1}
};
const ftype W0=1./3;
const ftype W1=1./18;
const ftype W2=1./36;
const ftype W3=0;
#define E_MATRIX_X(f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18) -f1 +f2-f3 +f5-f6   +f8-f9+f11-f12+f14
#define E_MATRIX_Y(f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18) -f3 -f4-f5+f6+f7+f8 -f15+f16-f17+f18
#define E_MATRIX_Z(f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18) -f9-f10-f11+f12+f13+f14-f15-f16+f17+f18

#elif defined D2Q5
#define DIM 2
#define QN 5
#define QN_IN_DIM 20   // QN*2^DIM
constexpr const int3 _e[QN] = {
  /* 0 */ { 0, 0, 0}, 
  /* 1 */ { 1, 0, 0}, 
  /* 2 */ { 0, 1, 0}, 
  /* 3 */ {-1, 0, 0}, 
  /* 4 */ { 0,-1, 0}, 
};
const ftype W0=0.5;
const ftype W1=0.125;
const ftype W2=0;
const ftype W3=0;

#elif defined D2Q9
#define DIM 2
#define QN 9
#define QN_IN_DIM 36   // QN*2^DIM
constexpr const int3 _e[QN] = {
  /* 0 */ { 0, 0, 0}, 
  /* 1 */ { 1, 0, 0}, 
  /* 2 */ { 0, 1, 0}, 
  /* 3 */ { 1, 1, 0}, 
  /* 4 */ {-1, 0, 0}, 
  /* 5 */ {-1, 1, 0}, 
  /* 6 */ { 0,-1, 0}, 
  /* 7 */ { 1,-1, 0},
  /* 8 */ {-1,-1, 0}
};
const ftype W0=0.5;
const ftype W1=0.125;
const ftype W2=0;
const ftype W3=0;

#endif

const int Qn=QN;

HOST_DEV_CHOOSE const int3 e[Qn] = { SEQ_LIST_COLLECTION(_e,QN) };

template<class T, const int Narr=Qn> struct Consts_Wrap {
  constexpr Consts_Wrap():arr() {}
  constexpr T operator[](int iq) const { return arr[iq]; }
  T arr[Narr];
};
constexpr const struct _W: public Consts_Wrap<ftype> {
  constexpr _W():Consts_Wrap<ftype>() {
    for(int i=0; i<Qn; i++) {
      const auto sum = abs(_e[i].x)+abs(_e[i].y)+abs(_e[i].z);
      const ftype wchoices[] = {W0, W1, W2, W3};
      arr[i] = wchoices[sum];
    }
  }
} _w;
template<int DIRX,int DIRY,int DIRZ> struct _Reverse: public Consts_Wrap<char> {
  constexpr _Reverse():Consts_Wrap<char>() {
    for(int i=0; i<Qn; i++) for(int j=0; j<Qn; j++) {
      const bool matchX = _e[i].x==_e[j].x && DIRX==0 || _e[i].x==-_e[j].x && DIRX;
      const bool matchY = _e[i].y==_e[j].y && DIRY==0 || _e[i].y==-_e[j].y && DIRY;
      const bool matchZ = _e[i].z==_e[j].z && DIRZ==0 || _e[i].z==-_e[j].z && DIRZ;
      if(matchX && matchY && matchZ) { arr[i]=j; break; }
    }
  }
};
constexpr const _Reverse<1,0,0> _reverse100;
constexpr const _Reverse<0,1,0> _reverse010;
constexpr const _Reverse<0,0,1> _reverse001;
constexpr const _Reverse<1,1,1> _reverse111;

//Generates three bits assosiated with XYZ coordinates
//  representing following information:
//  0 -- the required fi with _dir_ direction can be aquired from the current cell
//  1 -- the required fi with _dir_ direction can be aquired from the neighboured cell
//                and fi is also reversed
//   For example, for pbits=011 and returned value 110 the desired fi = neigbour(X+1, Y-1, 0).f[reverseXY]
template<int pbits> struct _CompactNsh: public Consts_Wrap<char> {
  constexpr _CompactNsh():Consts_Wrap<char>() {
    for(int i=0; i<Qn; i++) {
      const auto dir = _e[i];
      constexpr const int3 cnt = { 1-2*(pbits&1), 1-2*(pbits>>1&1), 1-2*(pbits>>2&1) };
      arr[i] = (dir.x==cnt.x)|(dir.y==cnt.y)<<1|(dir.z==cnt.z)<<2;
    }
  }
};
constexpr const _CompactNsh<0> _compact_nsh_0;
constexpr const _CompactNsh<1> _compact_nsh_1;
constexpr const _CompactNsh<2> _compact_nsh_2;
constexpr const _CompactNsh<3> _compact_nsh_3;

//Generates neighbourCell_f_iq index
template<int pbits> struct _CompactIndFi: public Consts_Wrap<uchar> {
  constexpr _CompactIndFi():Consts_Wrap<uchar>() {
    constexpr const _Reverse<1,0,0> revX;
    constexpr const _Reverse<0,1,0> revY;
    constexpr const _Reverse<0,0,1> revZ;
    for(int i=0; i<Qn; i++) {
      const auto dir = _e[i];
      const int3 cnt = { 1-2*(pbits&1), 1-2*(pbits>>1&1), 1-2*(pbits>>2&1) };
      arr[i]=i;
      if(dir.x==cnt.x) arr[i]=revX[arr[i]];
      if(dir.y==cnt.y) arr[i]=revY[arr[i]];
      if(dir.z==cnt.z) arr[i]=revZ[arr[i]];
      // c++ cannot evaluate this template as consts 
      // arr[i] = _Reverse< (int)(dir.x==cnt.x), (int)(dir.y==cnt.y), (int)(dir.z==cnt.z) >()[i];
    }
  }
};
constexpr const _CompactIndFi<7> ccccc;

/*constexpr const struct _CompactAccess: public Consts_Wrap<char4, Qn*(1<<DIM) > {
  constexpr _CompactAccess(): Consts_Wrap<char4, Qn*(1<<DIM)>() {
    for(int ind=0; ind<Qn*(1<<DIM); ind++) {
      const int pbits = ind/Qn;
      const int iq = ind%Qn;
      const int3 cnt = { 1-2*(pbits&1), 1-2*(pbits>>1&1), 1-2*(pbits>>2&1) };
      const auto dir = _e[iq];
      arr[ind].x = (dir.x==cnt.x)*(1-2*(pbits>>0&1));
      arr[ind].y = (dir.y==cnt.y)*(1-2*(pbits>>1&1));
      arr[ind].z = (dir.z==cnt.z)*(1-2*(pbits>>2&1));
      arr[ind].w = _Reverse< (int)(dir.x==cnt.x), (int)(dir.y==cnt.y), (int)(dir.z==cnt.z) >()[iq];
    }
  }
} _compact_access;*/
constexpr const struct _CompactAccess: public Consts_Wrap<char4, Qn*(1<<DIM) > {
  char checkval[Qn*(1<<DIM)];
  constexpr _CompactAccess(): Consts_Wrap<char4, Qn*(1<<DIM)>(),checkval() {
    constexpr const _Reverse<1,0,0> revX;
    constexpr const _Reverse<0,1,0> revY;
    constexpr const _Reverse<0,0,1> revZ;
    for(int pbits=0,i=0; pbits<(1<<DIM); pbits++) {
      const int3 cnt = { 1-2*(pbits&1), 1-2*(pbits>>1&1), 1-2*(pbits>>2&1) };
      for(int iq=0; iq<Qn; iq++,i++) {
        const auto dir = _e[iq];
        arr[i].x = (dir.x==cnt.x)*(1-2*(pbits>>0&1));
        arr[i].y = (dir.y==cnt.y)*(1-2*(pbits>>1&1));
        arr[i].z = (dir.z==cnt.z)*(1-2*(pbits>>2&1));
        int n_iq = iq;
        if(dir.x==cnt.x) n_iq=revX[n_iq];
        if(dir.y==cnt.y) n_iq=revY[n_iq];
        if(dir.z==cnt.z) n_iq=revZ[n_iq];
        arr[i].w = n_iq;
        // c++ cannot evaluate this template as consts 
        // arr[i].w = _Reverse< (int)(dir.x==cnt.x), (int)(dir.y==cnt.y), (int)(dir.z==cnt.z) >()[iq];
        checkval[i]=arr[i].w;
      }
    }
  }
} _compact_access;

constexpr const struct _CompactFillCells: public Consts_Wrap<uchar2, Qn*(1<<DIM) > {
  uchar checkval[Qn*(1<<DIM)];
  constexpr _CompactFillCells(): Consts_Wrap<uchar2, Qn*(1<<DIM)>(),checkval() {
    constexpr const _Reverse<1,0,0> revX;
    constexpr const _Reverse<0,1,0> revY;
    constexpr const _Reverse<0,0,1> revZ;
    for(int icell=0,i=0; icell<(1<<DIM); icell++) {
      const int3 cnt = { 1-2*(icell&1), 1-2*(icell>>1&1), 1-2*(icell>>2&1) };
      for(int iq=0; iq<Qn; iq++,i++) {
        const auto dir = _e[iq];
        const int3 wr_cell = { icell>>0&1^(dir.x==cnt.x), icell>>1&1^(dir.y==cnt.y), icell>>2&1^(dir.z==cnt.z) };
        arr[i].x = wr_cell.x | wr_cell.y<<1 | wr_cell.z<<2;
        int n_iq = iq;
        if(dir.x==cnt.x) n_iq=revX[n_iq];
        if(dir.y==cnt.y) n_iq=revY[n_iq];
        if(dir.z==cnt.z) n_iq=revZ[n_iq];
        arr[i].y = n_iq;
        // c++ cannot evaluate this template as consts 
        // arr[i].w = _Reverse< (int)(dir.x==cnt.x), (int)(dir.y==cnt.y), (int)(dir.z==cnt.z) >()[iq];
        checkval[i]=arr[i].x;
      }
    }
  }
} _compact_fill_cells;

//=====Pattern of How to use _compact_fill_cells:=====
//  for(iq:{0...Qn}) for(icell:{0...8}) {
//      uchar2 (ncell,niq) =  _compact_fill_cells[iq+icell*Qn];
//      local_real_cell[ncell].f[niq] = Group_of_2x2x2cells_in_glob_mem.get_fi[icell,iq];
//  }

//TODO remake everything to std::array
//#include "utility"
//auto ints = std::make_index_sequence<int, Qn>{};

HOST_DEV_CHOOSE const ftype w[Qn] = { SEQ_LIST_COLLECTION(_w, QN) } ;
HOST_DEV_CHOOSE const int reverseX[Qn]   = { SEQ_LIST_COLLECTION(_reverse100, QN) } ;
HOST_DEV_CHOOSE const int reverseY[Qn]   = { SEQ_LIST_COLLECTION(_reverse010, QN) } ;
HOST_DEV_CHOOSE const int reverseZ[Qn]   = { SEQ_LIST_COLLECTION(_reverse001, QN) } ;
HOST_DEV_CHOOSE const int reverseXYZ[Qn] = { SEQ_LIST_COLLECTION(_reverse111, QN) } ;
HOST_DEV_CHOOSE const char4 compact_access[Qn*(1<<DIM)] = { SEQ_LIST_COLLECTION(_compact_access, QN_IN_DIM) } ;
HOST_DEV_CHOOSE const uchar2 compact_fill_cells[Qn*(1<<DIM)] = { SEQ_LIST_COLLECTION(_compact_fill_cells, QN_IN_DIM) } ;

template void debug_consts_assert<1, int(_w[0]*1000),int(_w[1]*1000),int(_w[2]*1000),int(_w[3]*1000),int(_w[4]*1000),int(_w[5]*1000),int(_w[6]*1000),int(_w[7]*1000) >(); 
//template void debug_consts_assert<0,SEQ_LIST_COLLECTION_CHECK(_w,QN)>(); 
//template void debug_consts_assert<0,SEQ_LIST_COLLECTION(_reverse111,QN)-11111>(); 
//template void debug_consts_assert<0,SEQ_LIST_COLLECTION(ccccc, QN)-11111>(); 
template void debug_consts_assert<1, SEQ_LIST_COLLECTION_CHECK(_compact_access, QN_IN_DIM)-111111>(); 
template void debug_consts_assert<1, SEQ_LIST_COLLECTION_CHECK(_compact_fill_cells, QN_IN_DIM)-111111>(); 



//const long int No=(Qn+1)/2;
const long int No=1;

const ftype cs2 = dx*dx/3.;
const ftype dcs2 = 1./cs2;
const ftype dcs4 = 1./(cs2*cs2);

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

// #ifdef __CUDA_ARCH__
// #define e_c e_const
// #define w_c w_const
// #else
// #define e_c e_host
// #define w_c w_host
// #endif
};
