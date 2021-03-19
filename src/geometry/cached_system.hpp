#ifndef GEOMETRY_CACHED_SYSTEM_HPP_
#define GEOMETRY_CACHED_SYSTEM_HPP_

#include <array>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

// Parthenon includes
#include <coordinates/coordinates.hpp>
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

// phoebus includes
#include "geometry/geometry_defaults.hpp"
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/cell_locations.hpp"

using namespace parthenon::package::prelude;
using parthenon::ParArray1D;

namespace Geometry {

// A geometry type that "caches" an analytic coordinate system on the
// grid. Quantities such as the metric, connection coefficients, etc.,
// are stored at cell centers and cell faces. If a request is made for
// a node or for off-grid, the original analytic system is used.
const std::vector<std::string> GEOMETRY_CACHED_VARS = {
    "g.c.alpha",   "g.c.bcon",  "g.c.gcov",  "g.c.gamcon",
    "g.c.detgam",  "g.c.dg",    "g.c.coord",

    "g.f1.alpha",  "g.f1.bcon", "g.f1.gcov", "g.f1.gamcon",
    "g.f1.detgam", "g.f1.dg",

    "g.f2.alpha",  "g.f2.bcon", "g.f2.gcov", "g.f2.gamcon",
    "g.f2.detgam", "g.f2.dg",

    "g.f3.alpha",  "g.f3.bcon", "g.f3.gcov", "g.f3.gamcon",
    "g.f3.detgam", "g.f3.dg",

    "g.n.alpha",   "g.n.bcon",  "g.n.gcov",  "g.n.gamcon",
    "g.n.detgam",  "g.n.dg",    "g.n.coord"};
const std::vector<std::string> GEOMETRY_LOC_NAMES = {"c", "f1", "f2", "f3",
                                                     "n"};

// Support class. Accesses a pack on cells or faces as appropriate
namespace Impl {
struct GeomPackIndices {
  int alpha, bcon, gcov, gamcon, detgam, dg;
};
template <typename T> class LocArray {
public:
  KOKKOS_INLINE_FUNCTION
  T &operator[](int i) { return a_[i]; }
  KOKKOS_INLINE_FUNCTION
  T &operator[](CellLocation loc) { return (*this)[icast_(loc)]; }
  KOKKOS_INLINE_FUNCTION
  const T &operator[](int i) const { return a_[i]; }
  KOKKOS_INLINE_FUNCTION
  const T &operator[](CellLocation loc) const { return (*this)[icast_(loc)]; }

private:
  KOKKOS_INLINE_FUNCTION
  int icast_(CellLocation loc) const { return static_cast<int>(loc); }
  std::array<T, NUM_CELL_LOCATIONS> a_;
};
} // namespace Impl

// TODO(JMM): Store corners when available if needed
template <typename Pack, typename System> class Cached {
public:
  Cached() = default;
  template <typename Data> Cached(Data *rc, const System &s) : s_(s) {
    PackIndexMap imap;
    pack_ = rc->PackVariables(GEOMETRY_CACHED_VARS, imap);
    int i = 0;
    for (auto &loc : GEOMETRY_LOC_NAMES) {
      PARTHENON_DEBUG_REQUIRE(imap["g." + loc + ".alpha"].second >= 0,
                              "variable exists");
      idx_[i].alpha = imap["g." + loc + ".alpha"].first;
      idx_[i].bcon = imap["g." + loc + ".bcon"].first;
      idx_[i].gcov = imap["g." + loc + ".gcov"].first;
      idx_[i].gamcon = imap["g." + loc + ".gamcon"].first;
      idx_[i].detgam = imap["g." + loc + ".detgam"].first;
      idx_[i].dg = imap["g." + loc + ".dg"].first;
      i++;
    }
    icoord_c_ = imap["g.c.coord"].first;
    icoord_n_ = imap["g.n.coord"].first;
  }
  KOKKOS_INLINE_FUNCTION
  Real Lapse(Real X0, Real X1, Real X2, Real X3) const {
    return s_.Lapse(X0, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real Lapse(CellLocation loc, int b, int k, int j, int i) const {
    return pack_(b, idx_[loc].alpha, k, j, i);
  }
  KOKKOS_INLINE_FUNCTION
  Real Lapse(CellLocation loc, int k, int j, int i) const {
    return Lapse(loc, b_, k, j, i);
  }
  KOKKOS_INLINE_FUNCTION
  void ContravariantShift(Real X0, Real X1, Real X2, Real X3,
                          Real beta[NDSPACE]) const {
    s_.ContravariantShift(X0, X1, X2, X3, beta);
  }
  KOKKOS_INLINE_FUNCTION
  void ContravariantShift(CellLocation loc, int b, int k, int j, int i,
                          Real beta[NDSPACE]) const {
    SPACELOOP(d) { beta[d] = pack_(b, idx_[loc].bcon + d, k, j, i); }
  }
  KOKKOS_INLINE_FUNCTION
  void ContravariantShift(CellLocation loc, int k, int j, int i,
                          Real beta[NDSPACE]) const {
    ContravariantShift(loc, b_, k, j, i, beta);
  }
  KOKKOS_INLINE_FUNCTION
  void Metric(Real X0, Real X1, Real X2, Real X3,
              Real gamma[NDSPACE][NDSPACE]) const {
    return s_.Metric(X0, X1, X2, X3, gamma);
  }
  KOKKOS_INLINE_FUNCTION
  void Metric(CellLocation loc, int b, int k, int j, int i,
              Real gamma[NDSPACE][NDSPACE]) const {
    SPACELOOP2(m, n) { // gamma_{ij} = g_{ij}
      int offst = Utils::Flatten2(m + 1, n + 1, NDFULL);
      gamma[m][n] = pack_(b, idx_[loc].gcov + offst, k, j, i);
    }
  }
  KOKKOS_INLINE_FUNCTION
  void Metric(CellLocation loc, int k, int j, int i,
              Real gamma[NDSPACE][NDSPACE]) const {
    Metric(loc, b_, k, j, i, gamma);
  }
  KOKKOS_INLINE_FUNCTION
  void MetricInverse(Real X0, Real X1, Real X2, Real X3,
                     Real gamma[NDSPACE][NDSPACE]) const {
    return s_.MetricInverse(X0, X1, X2, X3, gamma);
  }
  KOKKOS_INLINE_FUNCTION
  void MetricInverse(CellLocation loc, int b, int k, int j, int i,
                     Real gamma[NDSPACE][NDSPACE]) const {
    SPACELOOP2(m, n) {
      int offst = Utils::Flatten2(m, n, NDSPACE);
      gamma[m][n] = pack_(b, idx_[loc].gamcon + offst, k, j, i);
    }
  }
  KOKKOS_INLINE_FUNCTION
  void MetricInverse(CellLocation loc, int k, int j, int i,
                     Real gamma[NDSPACE][NDSPACE]) const {
    MetricInverse(loc, b_, k, j, i, gamma);
  }
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetric(Real X0, Real X1, Real X2, Real X3,
                       Real g[NDFULL][NDFULL]) const {
    return s_.SpacetimeMetric(X0, X1, X2, X3, g);
  }
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetric(CellLocation loc, int b, int k, int j, int i,
                       Real g[NDFULL][NDFULL]) const {
    SPACETIMELOOP2(mu, nu) {
      int offst = Utils::Flatten2(mu, nu, NDFULL);
      g[mu][nu] = pack_(b, idx_[loc].gcov + offst, k, j, i);
    }
  }
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetric(CellLocation loc, int k, int j, int i,
                       Real g[NDFULL][NDFULL]) const {
    SpacetimeMetric(loc, b_, k, j, i, g);
  }
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetricInverse(Real X0, Real X1, Real X2, Real X3,
                              Real g[NDFULL][NDFULL]) const {
    return s_.SpacetimeMetricInverse(X0, X1, X2, X3, g);
  }
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetricInverse(CellLocation loc, int b, int k, int j, int i,
                              Real g[NDFULL][NDFULL]) const {
    auto &idx = idx_[loc];
    Real alpha2 = Lapse(loc, b, k, j, i);
    alpha2 *= alpha2;
    g[0][0] = -Utils::ratio(1, alpha2);
    SPACELOOP(mu) {
      g[mu + 1][0] = g[0][mu + 1] =
          Utils::ratio(pack_(b, idx.bcon + mu, k, j, i), alpha2);
    }
    SPACELOOP2(mu, nu) {
      int offst = Utils::Flatten2(mu, nu, NDSPACE);
      g[mu + 1][nu + 1] = pack_(b, idx.gamcon + offst, k, j, i);
      g[mu + 1][nu + 1] -= Utils::ratio(g[0][mu + 1] * g[0][nu + 1], alpha2);
    }
  }
  KOKKOS_INLINE_FUNCTION
  void SpacetimeMetricInverse(CellLocation loc, int k, int j, int i,
                              Real g[NDFULL][NDFULL]) const {
    SpacetimeMetricInverse(loc, k, j, i, g);
  }
  KOKKOS_INLINE_FUNCTION
  Real DetGamma(Real X0, Real X1, Real X2, Real X3) const {
    return s_.DetGamma(X0, X1, X2, X3);
  }
  KOKKOS_INLINE_FUNCTION
  Real DetGamma(CellLocation loc, int b, int k, int j, int i) const {
    return pack_(b, idx_[loc].detgam, k, j, i);
  }
  KOKKOS_INLINE_FUNCTION
  Real DetGamma(CellLocation loc, int k, int j, int i) const {
    return DetGamma(loc, b_, k, j, i);
  }
  template <typename... Args>
  KOKKOS_INLINE_FUNCTION Real DetG(Args... args) const {
    return Lapse(std::forward<Args>(args)...) *
           DetGamma(std::forward<Args>(args)...);
  }
  KOKKOS_INLINE_FUNCTION
  void ConnectionCoefficient(Real X0, Real X1, Real X2, Real X3,
                             Real Gamma[NDFULL][NDFULL][NDFULL]) const {
    return s_.ConnectionCoefficient(X0, X1, X2, X3, Gamma);
  }
  KOKKOS_INLINE_FUNCTION
  void ConnectionCoefficient(CellLocation loc, int k, int j, int i,
                             Real Gamma[NDFULL][NDFULL][NDFULL]) const {
    Utils::SetConnectionCoeffByFD(*this, Gamma, loc, k, j, i);
  }
  KOKKOS_INLINE_FUNCTION
  void ConnectionCoefficient(CellLocation loc, int b, int k, int j, int i,
                             Real Gamma[NDFULL][NDFULL][NDFULL]) const {
    Utils::SetConnectionCoeffByFD(*this, Gamma, loc, b, k, j, i);
  }
  KOKKOS_INLINE_FUNCTION
  void MetricDerivative(Real X0, Real X1, Real X2, Real X3,
                        Real dg[NDFULL][NDFULL][NDFULL]) const {
    // return s_.MetricDerivative(X0, X1, X2, X3, dg);
  }
  KOKKOS_INLINE_FUNCTION
  void MetricDerivative(CellLocation loc, int b, int k, int j, int i,
                        Real dg[NDFULL][NDFULL][NDFULL]) const {

    SPACETIMELOOP3(mu, nu, sigma) {
      int offst = (sigma - 1) * Utils::SymSize(NDFULL) +
                  Utils::Flatten2(mu, nu, NDFULL);
      dg[mu][nu][sigma] =
          (sigma == 0) ? 0 : pack_(b, idx_[loc].dg + offst, k, j, i);
    }
  }
  KOKKOS_INLINE_FUNCTION
  void MetricDerivative(CellLocation loc, int k, int j, int i,
                        Real dg[NDFULL][NDFULL][NDFULL]) const {
    MetricDerivative(loc, b_, k, j, i, dg);
  }
  KOKKOS_INLINE_FUNCTION
  void GradLnAlpha(Real X0, Real X1, Real X2, Real X3, Real da[NDFULL]) const {
    return s_.GradLnAlpha(X0, X1, X2, X3, da);
  }
  KOKKOS_INLINE_FUNCTION
  void GradLnAlpha(CellLocation loc, int b, int k, int j, int i,
                   Real da[NDFULL]) const {
    // (d/dx) ln(alpha) = (1/alpha) (d/dx)alpha = (1/alpha) alpha^3 (d/dx)g^{00}
    // because alpha = (-g^{00})^{-1/2}
    Real alpha2 = Lapse(loc, b, k, j, i);
    alpha2 *= alpha2;
    SPACETIMELOOP(mu) {
      int offset =
          (mu - 1) * Utils::SymSize(NDFULL) + Utils::Flatten2(0, 0, NDFULL);
      Real dg00 = pack_(b, idx_[loc].dg + offset, k, j, i);
      da[mu] = 0.5 * alpha2 * dg00;
    }
  }
  KOKKOS_INLINE_FUNCTION
  void GradLnAlpha(CellLocation loc, int k, int j, int i,
                   Real da[NDFULL]) const {
    return GradLnAlpha(loc, b_, k, j, i, da);
  }
  KOKKOS_INLINE_FUNCTION
  void Coords(Real X0, Real X1, Real X2, Real X3, Real C[NDFULL]) const {
    return s_.Coords(X0, X1, X2, X3, C);
  }
  KOKKOS_INLINE_FUNCTION
  void Coords(CellLocation loc, int b, int k, int j, int i,
              Real C[NDFULL]) const {
    if ((loc == CellLocation::Cent) || (loc == CellLocation::Corn)) {
      C[0] = X0_;
      int icoord = (loc == CellLocation::Cent) ? icoord_c_ : icoord_n_;
      SPACELOOP(d) { C[d + 1] = pack_(b, icoord + d, k, j, i); }
    } else {
      s_.Coords(loc, b, k, j, i, C);
    }
  }
  KOKKOS_INLINE_FUNCTION
  void Coords(CellLocation loc, int k, int j, int i, Real C[NDFULL]) const {
    if ((loc == CellLocation::Cent) || (loc == CellLocation::Corn)) {
      C[0] = X0_;
      int icoord = (loc == CellLocation::Cent) ? icoord_c_ : icoord_n_;
      SPACELOOP(d) { C[d + 1] = pack_(b_, icoord + d, k, j, i); }
    } else {
      s_.Coords(loc, k, j, i, C);
    }
  }

private:
  System s_;
  int icoord_c_; // don't bother with face coords for now.
  int icoord_n_;
  Pack pack_;
  Impl::LocArray<Impl::GeomPackIndices> idx_;
  static constexpr Real X0_ = 0;
  static constexpr int b_ = 0;
};

template <typename System>
using CachedOverMeshBlock = Cached<parthenon::VariablePack<Real>, System>;
template <typename System>
using CachedOverMesh = Cached<parthenon::MeshBlockVarPack<Real>, System>;

template <typename System>
void InitializeCachedCoordinateSystem(ParameterInput *pin,
                                      StateDescriptor *geometry) {
  using parthenon::MetadataFlag;

  Initialize<System>(pin, geometry);

  // Request fields for cache
  Utils::MeshBlockShape dims(pin);
  std::vector<int> shape;
  Metadata m;
  std::vector<MetadataFlag> flags_c = {Metadata::Cell, Metadata::Derived,
                                       Metadata::OneCopy};
  std::vector<MetadataFlag> flags_o = {Metadata::Derived, Metadata::OneCopy};

  std::vector<int> var_sizes = {1,
                                NDSPACE,
                                Utils::SymSize(NDFULL),
                                Utils::SymSize(NDSPACE),
                                1,
                                NDSPACE * Utils::SymSize(NDFULL)};
  std::vector<std::string> var_names = {"alpha",  "bcon", "gcov", "gamcon",
                                        "detgam", "dg",   "coord"};
  PARTHENON_REQUIRE_THROWS(var_sizes.size() == var_names.size(),
                           "Same number of variables as sizes");
  // Cell variables
  for (int i = 0; i < var_sizes.size(); ++i) {
    shape = {var_sizes[i]};
    m = Metadata(flags_c, shape);
    geometry->AddField("g.c." + var_names[i], m);
  }
  // face variables
  for (int d = 1; d <= 3; ++d) {
    for (int i = 0; i < var_sizes.size(); ++i) {
      shape = {dims.nx1, dims.nx2, dims.nx3, var_sizes[i]};
      shape[d - 1] += 1;
      m = Metadata(flags_o, shape);
      geometry->AddField("g." + GEOMETRY_LOC_NAMES[d] + var_names[i], m);
    }
  }
  // node variables
  for (int i = 0; i < var_sizes.size(); ++i) {
    shape = {dims.nx1 + 1, dims.nx2 + 1, dims.nx3 + 1, var_sizes[i]};
    m = Metadata(flags_o, shape);
    geometry->AddField("g.n" + var_names[i], m);
  }
}

template <typename System>
CachedOverMeshBlock<System> GetCachedCoordinateSystem(MeshBlockData<Real> *rc) {
  auto system = GetCoordinateSystem<System>(rc);
  return CachedOverMeshBlock<System>(rc, system);
}
template <typename System>
CachedOverMesh<System> GetCachedCoordinateSystem(MeshData<Real> *rc) {
  auto system = GetCoordinateSystem<System>(rc);
  return CachedOverMesh<System>(rc, system);
}
template <typename System>
void SetCachedCoordinateSystem(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetParentPointer();
  auto system = GetCoordinateSystem<System>(rc);

  Impl::LocArray<Impl::GeomPackIndices> idx;
  PackIndexMap imap;
  auto pack = rc->PackVariables(GEOMETRY_CACHED_VARS, imap);
  int i = 0;
  for (auto &loc : GEOMETRY_LOC_NAMES) {
    PARTHENON_DEBUG_REQUIRE(imap["g." + loc + ".alpha"].second >= 0,
                            "Variable exists");
    idx[i].alpha = imap["g." + loc + ".alpha"].first;
    idx[i].bcon = imap["g." + loc + ".bcon"].first;
    idx[i].gcov = imap["g." + loc + ".gcov"].first;
    idx[i].gamcon = imap["g." + loc + ".gamcon"].first;
    idx[i].detgam = imap["g." + loc + ".detgam"].first;
    idx[i].dg = imap["g." + loc + ".dg"].first;
    i++;
  }

  int icoord_c = imap["g.c.coord"].first;
  int icoord_n = imap["g.n.coord"].first;

  auto lamb =
      KOKKOS_LAMBDA(const int k, const int j, const int i, CellLocation loc) {
    Real beta[NDSPACE];
    Real gcov[NDFULL][NDFULL];
    Real gamcon[NDSPACE][NDSPACE];
    Real dg[NDFULL][NDFULL][NDFULL];
    pack(idx[loc].alpha, k, j, i) = system.Lapse(loc, k, j, i);
    system.ContravariantShift(loc, k, j, i, beta);
    SPACELOOP(d) { pack(idx[loc].bcon + d, k, j, i) = beta[d]; }
    system.SpacetimeMetric(loc, k, j, i, gcov);
    int lin = 0;
    for (int mu = 0; mu < NDFULL; ++mu) {
      for (int nu = mu; nu < NDFULL; ++nu) {
        pack(idx[loc].gcov + lin, k, j, i) = gcov[mu][nu];
        lin++;
      }
    }
    system.MetricInverse(loc, k, j, i, gamcon);
    lin = 0;
    for (int mu = 0; mu < NDSPACE; ++mu) {
      for (int nu = mu; nu < NDSPACE; ++nu) {
        pack(idx[loc].gamcon + lin, k, j, i) = gamcon[mu][nu];
        lin++;
      }
    }
    pack(idx[loc].detgam, k, j, i) = system.DetGamma(loc, k, j, i);
    system.MetricDerivative(loc, k, j, i, dg);
    lin = 0;
    for (int sigma = 1; sigma < NDFULL; ++sigma) {
      for (int mu = 0; mu < NDSPACE; ++mu) {
        for (int nu = mu; nu < NDSPACE; ++nu) {
          pack(idx[loc].dg + lin, k, j, i) = dg[mu][nu][sigma];
          lin++;
        }
      }
    }

    if ((loc == CellLocation::Cent) || (loc == CellLocation::Corn)) {
      Real C[NDFULL];
      int icoord = (loc == CellLocation::Cent) ? icoord_c : icoord_n;
      system.Coords(loc, k, j, i, C);
      SPACETIMELOOP(mu) pack(icoord + mu, k, j, i) = C[mu];
    }
  };

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  pmb->par_for(
      "SetGeometry::Set Cached data, Cent", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        lamb(k, j, i, CellLocation::Cent);
      });
  pmb->par_for(
      "SetGeometry::Set Cached data, Face1", kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e + 1, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        lamb(k, j, i, CellLocation::Face1);
      });
  pmb->par_for(
      "SetGeometry::Set Cached data, Face2", kb.s, kb.e, jb.s, jb.e + 1, ib.s,
      ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        lamb(k, j, i, CellLocation::Face2);
      });
  pmb->par_for(
      "SetGeometry::Set Cached data, Face3", kb.s, kb.e + 1, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        lamb(k, j, i, CellLocation::Face3);
      });
  pmb->par_for(
      "SetGeometry::Set Cached data, Corn", kb.s, kb.e + 1, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        lamb(k, j, i, CellLocation::Corn);
      });
}

} // namespace Geometry

#endif // GEOMETRY_CACHED_SYSTEM_HPP_
