// parthenon::ParArray2D<Real> © 2021. Triad National Security, LLC. All rights reserved.
// This
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

// Parthenon
#include <globals.hpp>

#include "monopole_gr/monopole_gr_base.hpp"
#include "pgen/pgen.hpp"
#include "phoebus_utils/root_find.hpp"
#include "phoebus_utils/unit_conversions.hpp"
#include "utils/error_checking.hpp"
#include <cmath>

// Spiner
#include <spiner/interpolation.hpp>

namespace standing_accretion_shock {

parthenon::constants::PhysicalConstants<parthenon::constants::CGS> pc;
using Microphysics::EOS::EOS;

constexpr int NSASW = 8;
constexpr int R = 0;
constexpr int RHO = 1;
constexpr int VR = 2;
constexpr int EPS = 3;
constexpr int PRES = 4;
constexpr int HEAT = 5;
constexpr int S = 6;
constexpr int OMEGA = 7;

using State_t = parthenon::ParArray2D<Real>;
using State_host_t = typename parthenon::ParArray2D<Real>::HostMirror;
using Radius = Spiner::RegularGrid1D;

std::pair<int, int> Get1DProfileNumZones(const std::string model_filename);

template <typename ParArray2D>
void Get1DProfileData(const std::string model_filename, const int num_zones,
                      const int num_comments, const int num_vars, ParArray2D &state_raw);

KOKKOS_FUNCTION
Real ucon_norm(Real ucon[4], Real gcov[4][4]);

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  PARTHENON_REQUIRE(
      typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::SphericalKerrSchild),
      "Problem \"standing_accretion_shock\" requires \"SphericalKerrSchild\" geometry!");

  auto &rc = pmb->meshblock_data.Get();

  PackIndexMap imap;
  auto v = rc->PackVariables(
      {fluid_prim::density, fluid_prim::velocity, fluid_prim::energy, fluid_prim::bfield,
       fluid_prim::ye, fluid_prim::pressure, fluid_prim::temperature, fluid_prim::gamma1},
      imap);

  const int irho = imap[fluid_prim::density].first;
  const int ivlo = imap[fluid_prim::velocity].first;
  const int ivhi = imap[fluid_prim::velocity].second;
  const int ieng = imap[fluid_prim::energy].first;
  const int ib_lo = imap[fluid_prim::bfield].first;
  const int ib_hi = imap[fluid_prim::bfield].second;
  const int iye = imap[fluid_prim::ye].second;
  const int iprs = imap[fluid_prim::pressure].first;
  const int itmp = imap[fluid_prim::temperature].first;
  const int igm1 = imap[fluid_prim::gamma1].first;

  const std::string eos_type = pin->GetString("eos", "type");
  PARTHENON_REQUIRE_THROWS(
      eos_type == "IdealGas" || eos_type == "StellarCollapse",
      "Standing Accretion Shock setup only works with Ideal Gas or Stellar Collapse EOS");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto &coords = pmb->coords;
  auto &unit_conv =
      pmb->packages.Get("phoebus")->Param<phoebus::UnitConversions>("unit_conv");
  auto eos = pmb->packages.Get("eos")->Param<Microphysics::EOS::EOS>("d.EOS");
  const Real a = pin->GetReal("geometry", "a");
  auto geom = Geometry::GetCoordinateSystem(rc.get());

  // info about model file
  std::string model_filename =
      pin->GetOrAddString("standing_accretion_shock", "model_filename", "file.txt");

  std::pair<int, int> p = Get1DProfileNumZones(model_filename);
  const int num_zones = p.first - 1;
  const int num_comments = p.second;

  printf("1D model read in with %d number of zones. \n", num_zones);
  printf("1D model read in with %d number of comments. \n", num_comments);

  // Kokkos views for loading in data
  State_t state_raw_d("state raw", NSASW, num_zones);
  State_host_t state_raw_h = Kokkos::create_mirror_view(state_raw_d);

  Get1DProfileData(model_filename, num_zones, num_comments, NSASW, state_raw_h);

  Kokkos::deep_copy(state_raw_d, state_raw_h);

  const Real rhomin = 1.e-16;
  const Real epsmin = 1.e-16;

  Real rin = state_raw_h(R, 0);
  Real rout = state_raw_h(R, num_zones - 1);

  printf("rin  = %.14e (code units)\n", rin);
  printf("rout  = %.14e (code units)\n", rout);

  Radius raw_radius(rin, rout, num_zones);

  pmb->par_for(
      "Phoebus::ProblemGenerator::StandingAccrectionShock", kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real x1 = coords.Xc<1>(k, j, i);
        const Real x2 = coords.Xc<2>(k, j, i);
        const Real x3 = coords.Xc<3>(k, j, i);

        Real r = std::abs(x1);
        Real eos_lambda[2] = {0.};

        if (iye > 0) {
          v(iye, k, j, i) = 0.5;
          eos_lambda[0] = v(iye, k, j, i);
        }

        Real rho =
            std::max(rhomin, MonopoleGR::Interpolate(r, state_raw_d, raw_radius, RHO));
        Real eps =
            std::max(epsmin, MonopoleGR::Interpolate(r, state_raw_d, raw_radius, EPS));
        Real vr = std::max(-1., MonopoleGR::Interpolate(r, state_raw_d, raw_radius, VR));

        v(irho, k, j, i) = rho;
        v(ieng, k, j, i) = rho * eps;
        v(iprs, k, j, i) = eos.PressureFromDensityInternalEnergy(rho, eps, eos_lambda);
        v(igm1, k, j, i) =
            eos.BulkModulusFromDensityInternalEnergy(rho, eps, eos_lambda) /
            v(iprs, k, j, i);

        Real ucon[] = {0.0, vr, 0.0, 0.0};
        Real gcov[4][4];
        geom.SpacetimeMetric(CellLocation::Cent, k, j, i, gcov);
        ucon[0] = ucon_norm(ucon, gcov);

        const Real lapse = geom.Lapse(CellLocation::Cent, k, j, i);
        Real beta[3];
        geom.ContravariantShift(CellLocation::Cent, k, j, i, beta);
        Real W = lapse * ucon[0];

        for (int d = 0; d < 3; d++) {
          v(ivlo + d, k, j, i) = ucon[d + 1] + W * beta[d] / lapse;
        }
      });
  fluid::PrimitiveToConserved(rc.get());
}

std::pair<int, int> Get1DProfileNumZones(const std::string model_filename) {

  // open file
  std::ifstream inputfile(model_filename);
  const std::string whitespace(" \t\n\r");

  // error check
  if (!inputfile.is_open()) {
    std::cout << model_filename << " not found :( \n.";
    exit(-1);
  }

  int nz = 0;
  int nc = 0;
  std::string line;
  std::string comment1("#");
  std::string comment2("//");

  // get number of zones from 1d file
  while (!inputfile.eof()) {

    getline(inputfile, line);

    std::size_t first_nonws = line.find_first_not_of(whitespace);

    // skip empty lines
    if (first_nonws == std::string::npos) {
      continue;
    }

    // skip comments
    if (line.find(comment1) == first_nonws || line.find(comment2) == first_nonws) {
      nc++;
      continue;
    }

    nz++;
  }

  inputfile.close();

  return std::make_pair(nz + 1, nc);
}

template <typename ParArray2D>
void Get1DProfileData(const std::string model_filename, const int num_zones,
                      const int num_comments, const int num_vars, ParArray2D &state_raw) {

  std::ifstream inputfile(model_filename);

  Real val = 0;
  std::string line;
  int line_num = 0;

  // skip comment lines
  while (line_num < num_comments) {
    getline(inputfile, line);
    line_num++;
  }

  // read file into model_1d
  for (int i = 0; i < num_zones; i++) // number of zones
  {
    for (int j = 0; j < num_vars; j++) //  number of vars
    {
      inputfile >> val;
      state_raw(j, i) = val;
    }
  }

  printf("Read 1D profile  into state array.\n");

  inputfile.close();
}

KOKKOS_FUNCTION
Real ucon_norm(Real ucon[4], Real gcov[4][4]) {
  Real AA = gcov[0][0];
  Real BB = 2. * (gcov[0][1] * ucon[1] + gcov[0][2] * ucon[2] + gcov[0][3] * ucon[3]);
  Real CC = 1. + gcov[1][1] * ucon[1] * ucon[1] + gcov[2][2] * ucon[2] * ucon[2] +
            gcov[3][3] * ucon[3] * ucon[3] +
            2. * (gcov[1][2] * ucon[1] * ucon[2] + gcov[1][3] * ucon[1] * ucon[3] +
                  gcov[2][3] * ucon[2] * ucon[3]);
  Real discr = BB * BB - 4. * AA * CC;
  if (discr < 0) printf("discr = %g   %g %g %g\n", discr, AA, BB, CC);
  PARTHENON_REQUIRE(discr >= 0, "discr < 0");
  return (-BB - std::sqrt(discr)) / (2. * AA);
}

} // namespace standing_accretion_shock
