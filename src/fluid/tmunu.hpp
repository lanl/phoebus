#ifndef FLUID_TMUNU_HPP_
#define FLUID_TMUNU_HPP_

#include "geometry/geometry.hpp"

namespace fluid {

  template<typename Pack>
  class StressEnergyTensor {
  public:
    static constexpr int ND = Geometry::CoordinateSystem::NDFULL;

    KOKKOS_INLINE_FUNCTION
    StressEnergyTensor() = default;

    KOKKOS_INLINE_FUNCTION
    void operator()(int b, int k, int j, int i, Real T[ND][ND]);
    void operator()(int k, int j, int i, Real T[ND][ND]);
  private:
    Geometry::CoordinateSystem system_;
    
  };

}

#endif // FLUID_TMUNU_HPP_
