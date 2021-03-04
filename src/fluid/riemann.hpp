#ifndef FLUID_RIEMANN_HPP_
#define FLUID_RIEMANN_HPP_

#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

#include "compile_constants.hpp"
#include "geometry/geometry.hpp"
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/variables.hpp"


namespace riemann {

enum class solver {LLF, HLL};

class FluxState {
 public:
  FluxState(MeshBlockData<Real> *rc) : FluxState(rc, PackIndexMap()) {}

  static void ReconVars(std::vector<std::string> &vars) {
    for (const auto &v : vars) {
      recon_vars.push_back(v);
    }
  }
  static void ReconVars(const std::string &var) {
    recon_vars.push_back(var);
  }
  static void FluxVars(std::vector<std::string> &vars) {
    for (const auto &v : vars) {
      flux_vars.push_back(v);
    }
  }
  static void FluxVars(const std::string &var) {
    flux_vars.push_back(var);
  }

  static std::vector<std::string> ReconVars() {
    return recon_vars;
  }
  static std::vector<std::string> FluxVars() {
    return flux_vars;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  int NumConserved() const {
    return ncons;
  }
  
  KOKKOS_FUNCTION
  void prim_to_flux(const int d, const int k, const int j, const int i,
                    const ParArrayND<Real> &q, Real &vm, Real &vp, Real *U, Real *F) const;

  const VariableFluxPack<Real> v;
  const ParArrayND<Real> ql;
  const ParArrayND<Real> qr;
  const Geometry::CoordinateSystem geom;
 private:
  const int prho, pvel_lo, pvel_hi, peng, pb_lo, pb_hi, pye, prs, gm1;
  const int crho, cmom_lo, cmom_hi, ceng, cb_lo, cb_hi, cye, ncons;
  static std::vector<std::string> recon_vars, flux_vars;
  FluxState(MeshBlockData<Real> *rc, PackIndexMap imap)
    : v(rc->PackVariablesAndFluxes(ReconVars(), FluxVars(), imap)),
      ql(rc->Get("ql").data),
      qr(rc->Get("qr").data),
      geom(Geometry::GetCoordinateSystem(rc)),
      prho(imap[primitive_variables::density].first),
      pvel_lo(imap[primitive_variables::velocity].first),
      pvel_hi(imap[primitive_variables::velocity].second),
      peng(imap[primitive_variables::energy].first),
      pb_lo(imap[primitive_variables::bfield].first),
      pb_hi(imap[primitive_variables::bfield].second),
      pye(imap[primitive_variables::ye].second),
      prs(imap[primitive_variables::pressure].first),
      gm1(imap[primitive_variables::gamma1].first),
      crho(imap[conserved_variables::density].first),
      cmom_lo(imap[conserved_variables::momentum].first),
      cmom_hi(imap[conserved_variables::momentum].second),
      ceng(imap[conserved_variables::energy].first),
      cb_lo(imap[conserved_variables::bfield].first),
      cb_hi(imap[conserved_variables::bfield].second),
      cye(imap[conserved_variables::ye].first),
      ncons(5 + (pb_hi-pb_lo+1) + (cye>0)) {
    PARTHENON_REQUIRE_THROWS(ncons <= NCONS_MAX, "ncons exceeds NCONS_MAX.  Reconfigure to increase NCONS_MAX.");
  }



  KOKKOS_FORCEINLINE_FUNCTION
  int Delta(const int i, const int j) const {
    return i==j;
  }
};

KOKKOS_FUNCTION
Real llf(const FluxState &fs, const int d, const int k, const int j, const int i);
KOKKOS_FUNCTION
Real hll(const FluxState &fs, const int d, const int k, const int j, const int i);

} // namespace riemann

#endif // FLUID_RIEMANN_HPP_