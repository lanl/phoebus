// © 2021. Triad National Security, LLC. All rights reserved.  This
// program was produced under U.S. Government contract
// 89233218CNA000001 for Los Alamos National Laboratory (LANL), which
// is operated by Triad National Security, LLC for the U.S.
// Department of Energy/National Nuclear Security Administration. All
// rights in the program are reserved by Triad National Security, LLC,
// and the U.S. Department of Energy/National Nuclear Security
// Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works,
// distribute copies to the public, perform publicly and display
// publicly, and to permit others to do so.

#include <memory>

#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>

#include "gr1d.hpp"

using namespace parthenon::package::prelude;

namespace GR1D {
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto gr1d = std::make_shared<StateDescriptor>("GR1D");
  Params &params = gr1d->AllParams();

  bool enable_gr1d = pin->GetOrAddBoolean("GR1D", "enabled", false);
  params.Add("enable_gr1d", enable_gr1d);
  if (!enable_gr1d) return gr1d; // Short-circuit with nothing

  // TODO(JMM): Ghost zones or one-sided differences/BCs?
  int npoints = pin->GetOrAddInteger("GR1D", "npoints", 100);
  params.Add("npoints", npoints);

  // Rin and Rout are not necessarily the same as
  // bounds for the fluid
  Real rin = 0;
  Real rout = pin->GetOrAddReal("GR1D", "rout", 100);
  params.Add("rin", rin);
  params.Add("rout", rout);

  // These are registered in Params, not as variables,
  // because they have unique shapes are 1-copy
  Grids grids;
  grids.a     = Grids::MetricGrid_t("GR1D a",     2, npoints);
  grids.K_rr  = Grids::MetricGrid_t("GR1D K^r_r", 2, npoints);
  grids.alpha = Grids::MetricGrid_t("GR1D alpha", 2, npoints);
  grids.aleph = Grids::MetricGrid_t("GR1D aleph", 2, npoints);
  grids.rho   = Grids::MatterGrid_t("GR1D rho",   npoints);
  grids.j_r   = Grids::MatterGrid_t("GR1D j^r",   npoints);
  grids.trcS  = Grids::MatterGrid_t("GR1D S",     npoints);
  params.Add("grids", grids);

  // The radius object, returns radius vs index and index vs radius
  Radius radius(rin, rout, npoints);
  params.Add("radius", radius);
  
  return gr1d;
}
} // namespace GR1D
