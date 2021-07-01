#include "fixup.hpp"

namespace fixup {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto fix = std::make_shared<StateDescriptor>("fixup");
  Params &params = fix->AllParams();

  bool enable_flux_fixup = pin->GetOrAddBoolean("fixup", "enable_flux_fixup", false);
  params.Add("enable_flux_fixup", enable_flux_fixup);
  bool enable_fixup = pin->GetOrAddBoolean("fixup", "enable_fixup", false);
  params.Add("enable_fixup", enable_fixup);
  bool enable_floors = pin->GetOrAddBoolean("fixup", "enable_floors", false);
  params.Add("enable_floors", enable_floors);

  if (enable_floors) {
    std::string floor_type = pin->GetString("fixup", "floor_type");
    if (floor_type == "ConstantRhoSie") {
      Real rho0 = pin->GetOrAddReal("fixup", "rho0", 0.0);
      Real sie0 = pin->GetOrAddReal("fixup", "sie0", 0.0);
      params.Add("floor", Floors(constant_rho_sie_floor_tag, rho0, sie0));
    } else if (floor_type == "ExpX1RhoSie") {
      Real rho0 = pin->GetOrAddReal("fixup", "rho0", 0.0);
      Real sie0 = pin->GetOrAddReal("fixup", "sie0", 0.0);
      Real rp = pin->GetOrAddReal("fixup", "rho_exp", -2.0);
      Real sp = pin->GetOrAddReal("fixup", "sie_exp", -1.0);
      params.Add("floor", Floors(exp_x1_rho_sie_floor_tag, rho0, sie0, rp, sp));
    } else {
      PARTHENON_FAIL("invalid <fixup>/floor_type input");
    }
  } else {
    params.Add("floor", Floors());
  }

  return fix;
}

TaskStatus FixFailures(MeshBlockData<Real> *rc) {

  namespace p = fluid_prim;
  namespace c = fluid_cons;
  auto *pmb = rc->GetParentPointer().get();
  return TaskStatus::complete;

  std::vector<std::string> vars({p::density, c::density, p::velocity,
                                 c::momentum, p::energy, c::energy});

  PackIndexMap imap;
  auto v = rc->PackVariables(vars, imap);
  const int prho = imap[p::density].first;
  const int crho = imap[c::density].first;
  const int pvel_lo = imap[p::velocity].first;
  const int pvel_hi = imap[p::velocity].second;
  const int cmom_lo = imap[c::momentum].first;
  const int cmom_hi = imap[c::momentum].second;
  const int peng = imap[p::energy].first;
  const int ceng = imap[c::energy].first;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  auto coords = pmb->coords;
  //  const Real Rh = std::log(2.0);

  auto gpkg = pmb->packages.Get("geometry");
  const Real a = gpkg->Param<Real>("a");
  const Real reh = 1. + sqrt(1. - a * a);
  const Real x1eh = std::log(reh); // TODO(BRR) still coordinate dependent     

  parthenon::par_for(
    DEFAULT_LOOP_PATTERN, "Fix velocity inside horizon", DevExecSpace(),
    kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int k, const int j, const int i) {
      if (coords.x1v(i) <= x1eh) {
        v(crho,k,j,i) *= 0.99;//std::min(v(cmom_lo,k,j,i), 0.0);
      }
    }
  );

  return TaskStatus::complete;
}

TaskStatus NothingEscapes(MeshBlockData<Real> *rc) {
  auto *pmb = rc->GetParentPointer().get();
  if (!pmb->packages.Get("fixup")->Param<bool>("enable_flux_fixup")) return TaskStatus::complete;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto coords = pmb->coords;
  //  const Real Rh = std::log(2.0);

  auto gpkg = pmb->packages.Get("geometry");
  const Real a = gpkg->Param<Real>("a");
  const Real reh = 1. + sqrt(1. - a * a);
  const Real x1eh = std::log(reh); // TODO(BRR) still coordinate dependent     

  auto flux = rc->PackVariablesAndFluxes({fluid_cons::density, fluid_cons::energy, fluid_cons::momentum, fluid_prim::pressure},
                                         {fluid_cons::density, fluid_cons::energy, fluid_cons::momentum});

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "FixFluxes", DevExecSpace(),
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,                                                      \
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        if (coords.x1f(i) <= x1eh+1.e-8) {
          for (int l = 0; l < 2; l++) {
            flux.flux(1,l,k,j,i) = std::min(flux.flux(1,l,k,j,i), 0.0);
          }
        }
      });

  return TaskStatus::complete;
}


} // namespace fixup
