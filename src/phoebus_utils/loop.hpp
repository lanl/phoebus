#ifndef PHOEBUS_UTILS_LOOP_HPP_
#define PHOEBUS_UTILS_LOOP_HPP_

#include "geometry/geometry_utils.hpp"

namespace loop {

namespace Flatten {

#define FLATTEN2(m,n) (m + n + (m > n ? (m+1)/3 + (m+1)/4 : (n+1)/3 + (n+1)/4))
template <int m, int n, int size>
KOKKOS_FORCEINLINE_FUNCTION int Flatten2() {
  PARTHENON_DEBUG_REQUIRE(0 <= m && 0 <= n && m < size && n < size, "bounds");
  return FLATTEN2(m,n);
}
#undef FLATTEN2

}

// compille time unroll a 1-d loop
// for (; i < imax; i++)
template <int i, int imax, bool loop=true>
struct UnrolledLoop {
  template <typename F>
  KOKKOS_FORCEINLINE_FUNCTION
  static void exec(const F &func) {
    func(i);
    UnrolledLoop<i+1,imax,(i+1<imax)>::exec(func);
  }
};

// compille time unroll a 2-d loop
// for (; i < imax; i++) for (; j < jmax; j++)
template <int i, int imax, int j, int jmax, bool loop=true>
struct UnrolledLoop2 {
  template <typename F>
  KOKKOS_FORCEINLINE_FUNCTION
  static void exec(const F &func) {
    func(i,j);
    UnrolledLoop2<i+(j+1)/jmax,imax,(j+1)%jmax, jmax, (i+(j+1)/jmax<imax)>::exec(func);
  }
};

// compille time unroll a 2-d loop
// for (; i < imax; i++) for (; j < jmax; j++)
template <int i, int imax, int j, int jmax, int ioff=0, int joff=0, bool loop=true>
struct UnrolledLoop2_flatidx {
  template <typename F>
  KOKKOS_FORCEINLINE_FUNCTION
  static void exec(const F &func) {
    const int offset = Flatten::Flatten2<i+ioff,j+joff,imax>();
    func(i,j,offset);
    UnrolledLoop2_flatidx<i+(j+1)/jmax,imax,(j+1)%jmax, jmax, ioff, joff, (i+(j+1)/jmax<imax)>::exec(func);
  }
};

// compille time unroll a 3-d loop
// for (; i < imax; i++) for (; j < jmax; j++) for (; k < kmax; k++)
template <int i, int imax, int j, int jmax, int k, int kmax, bool loop=true>
struct UnrolledLoop3 {
  template <typename F>
  KOKKOS_FORCEINLINE_FUNCTION
  static void exec(const F &func) {
    func(i,j,k);
    UnrolledLoop3<i+((j+1==jmax) && (k+1==kmax)), imax,
            (j+(k+1)/kmax)%jmax, jmax,
            (k+1)%kmax, kmax,
            (i+((j+1==jmax) && (k+1==kmax)))<imax>::exec(func);
  }
};

// compille time unroll a 3-d loop
// for (; i < imax; i++) for (; j < jmax; j++) for (; k < kmax; k++)
template <int i, int imax, int j, int jmax, int k, int kmax, int ioff, int joff, bool loop=true>
struct UnrolledLoop3_flatidx {
  template <typename F>
  KOKKOS_FORCEINLINE_FUNCTION
  static void exec(const F &func) {
    const int offset = Flatten::Flatten2<i+ioff,j+joff,imax>();
    func(i,j,k,offset);
    UnrolledLoop3_flatidx<i+((j+1==jmax) && (k+1==kmax)), imax,
            (j+(k+1)/kmax)%jmax, jmax,
            (k+1)%kmax, kmax, ioff, joff,
            (i+((j+1==jmax) && (k+1==kmax)))<imax>::exec(func);
  }
};

template <int start, int stop>
struct UnrolledLoop<start,stop,false> {
  template <typename F>
  static void exec(const F &func);
};

template <int i, int imax, int j, int jmax>
struct UnrolledLoop2<i,imax,j,jmax,false> {
  template <typename F>
  static void exec(const F &func);
};

template <int i, int imax, int j, int jmax, int ioff, int joff>
struct UnrolledLoop2_flatidx<i,imax,j,jmax,ioff,joff,false> {
  template <typename F>
  static void exec(const F &func);
};

template <int i, int imax, int j, int jmax, int k, int kmax>
struct UnrolledLoop3<i,imax,j,jmax,k,kmax,false> {
  template <typename F>
  static void exec(const F &func);
};

template <int i, int imax, int j, int jmax, int k, int kmax, int ioff, int joff>
struct UnrolledLoop3_flatidx<i,imax,j,jmax,k,kmax,ioff,joff,false> {
  template <typename F>
  static void exec(const F &func);
};

template <int start, int stop>
template <typename F>
void UnrolledLoop<start,stop,false>::exec(const F &func) {}

template <int i, int imax, int j, int jmax>
template <typename F>
void UnrolledLoop2<i,imax,j,jmax,false>::exec(const F &func) {}

template <int i, int imax, int j, int jmax, int ioff, int joff>
template <typename F>
void UnrolledLoop2_flatidx<i,imax,j,jmax,ioff,joff,false>::exec(const F &func) {}

template <int i, int imax, int j, int jmax, int k, int kmax>
template <typename F>
void UnrolledLoop3<i,imax,j,jmax,k,kmax,false>::exec(const F &func) {}

template <int i, int imax, int j, int jmax, int k, int kmax, int ioff, int joff>
template <typename F>
void UnrolledLoop3_flatidx<i,imax,j,jmax,k,kmax,ioff,joff,false>::exec(const F &func) {}

// define some convenient shorthands
template <typename F>
KOKKOS_FORCEINLINE_FUNCTION
void SpaceLoop(const F &func) {
  UnrolledLoop<0,3>::exec(func);
}
template <typename F>
KOKKOS_FORCEINLINE_FUNCTION
void SpacetimeLoop(const F &func) {
  UnrolledLoop<0,4>::exec(func);
}
template <typename F>
KOKKOS_FORCEINLINE_FUNCTION
void SpaceLoop2(const F &func) {
  UnrolledLoop2<0,3,0,3>::exec(func);
}
template <typename F, int ioff=0, int joff=0>
KOKKOS_FORCEINLINE_FUNCTION
void SpaceLoop2_flatidx(const F &func) {
  UnrolledLoop2_flatidx<0,3,0,3,ioff,joff>::exec(func);
}
template <typename F>
KOKKOS_FORCEINLINE_FUNCTION
void SpacetimeLoop2(const F &func) {
  UnrolledLoop2<0,4,0,4>::exec(func);
}
template <typename F, int ioff=0, int joff=0>
KOKKOS_FORCEINLINE_FUNCTION
void SpacetimeLoop2_flatidx(const F &func) {
  UnrolledLoop2_flatidx<0,4,0,4,ioff,joff>::exec(func);
}
template <typename F>
KOKKOS_FORCEINLINE_FUNCTION
void SpaceLoop3(const F &func) {
  UnrolledLoop3<0,3,0,3,0,3>::exec(func);
}
template <typename F>
KOKKOS_FORCEINLINE_FUNCTION
void SpacetimeLoop3(const F &func) {
  UnrolledLoop3<0,4,0,4,0,4>::exec(func);
}
template <typename F, int ioff=0, int joff=0>
KOKKOS_FORCEINLINE_FUNCTION
void SpacetimeLoop3_flatidx(const F &func) {
  UnrolledLoop3_flatidx<0,4,0,4,0,4,ioff,joff>::exec(func);
}

}

#endif // PHOEBUS_UTILS_LOOP_HPP