#include <cstdio>
#include <memory>
#include <vector>

#include "analysis/analysis.hpp"
#include "analysis/history.hpp"
#include "ascii_reader.hpp"
#include "geometry/geometry.hpp"
#include "microphysics/eos_phoebus/eos_phoebus.hpp"
#include "monopole_gr/monopole_gr.hpp"
#include "pgen/pgen.hpp"
#include "phoebus_utils/reduction.hpp"
#include "phoebus_utils/unit_conversions.hpp"
#include "progenitordata.hpp"

namespace Progenitor {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto progenitor_pkg = std::make_shared<StateDescriptor>("progenitor");
  Params &params = progenitor_pkg->AllParams();
  auto mutable_param = parthenon::Params::Mutability::Mutable;

  bool enabled = pin->GetOrAddBoolean("progenitor", "enabled", false);
  params.Add("enabled", enabled);
  if (!enabled) return progenitor_pkg;

  // Read the table
  std::vector<double> mydata[11];
  const std::string table = pin->GetString("progenitor", "tablepath");
  AsciiReader::readtable(table, mydata);
  const int npoints = mydata[0].size();

  // DataBoxes for interpolation functions
  Spiner::DataBox<Real> r(npoints);
  Spiner::DataBox<Real> mass_density(npoints);
  Spiner::DataBox<Real> temp(npoints);
  Spiner::DataBox<Real> Ye(npoints);
  Spiner::DataBox<Real> specific_internal_energy(npoints);
  Spiner::DataBox<Real> velocity(npoints);
  Spiner::DataBox<Real> pressure(npoints);

  Spiner::DataBox<Real> adm_density(npoints);
  Spiner::DataBox<Real> adm_momentum(npoints);
  Spiner::DataBox<Real> S_adm(npoints);
  Spiner::DataBox<Real> Srr_adm(npoints);

  // Fill in the variables
  // Requires same grid for primitive and adm quantities in the input table
  for (int i = 0; i < npoints; ++i) {
    r(i) = mydata[AsciiReader::COLUMNS::rad][i];

    mass_density(i) = mydata[AsciiReader::COLUMNS::rho][i];
    temp(i) = mydata[AsciiReader::COLUMNS::temp][i];
    Ye(i) = mydata[AsciiReader::COLUMNS::ye][i];
    specific_internal_energy(i) = mydata[AsciiReader::COLUMNS::eps][i];
    velocity(i) = mydata[AsciiReader::COLUMNS::vel][i];
    pressure(i) = mydata[AsciiReader::COLUMNS::press][i];

    adm_density(i) = mydata[AsciiReader::COLUMNS::rho_adm][i];
    adm_momentum(i) = mydata[AsciiReader::COLUMNS::P_adm][i];
    S_adm(i) = mydata[AsciiReader::COLUMNS::S_adm][i];
    Srr_adm(i) = mydata[AsciiReader::COLUMNS::Srr_adm][i];
  } // for loop (i)

  // Interpolation functions
  mass_density.setRange(0, r(0), r(npoints - 1), npoints);
  temp.setRange(0, r(0), r(npoints - 1), npoints);
  Ye.setRange(0, r(0), r(npoints - 1), npoints);
  specific_internal_energy.setRange(0, r(0), r(npoints - 1), npoints);
  velocity.setRange(0, r(0), r(npoints - 1), npoints);
  pressure.setRange(0, r(0), r(npoints - 1), npoints);

  adm_density.setRange(0, r(0), r(npoints - 1), npoints);
  adm_momentum.setRange(0, r(0), r(npoints - 1), npoints);
  S_adm.setRange(0, r(0), r(npoints - 1), npoints);
  Srr_adm.setRange(0, r(0), r(npoints - 1), npoints);

  // Get on device
  auto mass_density_dev = mass_density.getOnDevice();
  auto temp_dev = temp.getOnDevice();
  auto Ye_dev = Ye.getOnDevice();
  auto specific_internal_energy_dev = specific_internal_energy.getOnDevice();
  auto velocity_dev = velocity.getOnDevice();
  auto pressure_dev = pressure.getOnDevice();

  auto adm_density_dev = adm_density.getOnDevice();
  auto adm_momentum_dev = adm_momentum.getOnDevice();
  auto S_adm_dev = S_adm.getOnDevice();
  auto Srr_adm_dev = Srr_adm.getOnDevice();

  // Post-processing params
  Real outside_pns_threshold =
      pin->GetOrAddReal("progenitor", "outside_pns_threshold",
                        2.42e-5); // corresponds to entropy > 3 kb/baryon
  Real inside_pns_threshold = pin->GetOrAddReal("progenitor", "inside_pns_threshold",
                                                0.008); // corresponds to r < 80 km
  Real net_heat_threshold = pin->GetOrAddReal("progenitor", "net_heat_threshold",
                                              1e-8); // corresponds to r < 80 km
  auto mdot_radii = pin->GetOrAddVector("progenitor", "mdot_radii",
                                        std::vector<Real>{400}); // default 400km
  // progenitor status params
  bool post_bounce_model = pin->GetOrAddBoolean("progenitor", "post_bounce_model", false);

  // unit conversions
  UnitConversions units(pin);
  CodeConstants consts(units);
  Real LengthCGSToCode = units.GetLengthCGSToCode();
  Real DensityCGSToCode = units.GetMassDensityCGSToCode();

  // Add Params
  params.Add("mass_density", mass_density);
  params.Add("temp", temp);
  params.Add("Ye", Ye);
  params.Add("specific_internal_energy", specific_internal_energy);
  params.Add("velocity", velocity);
  params.Add("pressure", pressure);

  params.Add("adm_density", adm_density);
  params.Add("adm_momentum", adm_momentum);
  params.Add("S_adm", S_adm);
  params.Add("Srr_adm", Srr_adm);

  params.Add("mass_density_dev", mass_density_dev);
  params.Add("temp_dev", temp_dev);
  params.Add("Ye_dev", Ye_dev);
  params.Add("specific_internal_energy_dev", specific_internal_energy_dev);
  params.Add("velocity_dev", velocity_dev);
  params.Add("pressure_dev", pressure_dev);

  params.Add("adm_density_dev", adm_density_dev);
  params.Add("adm_momentum_dev", adm_momentum_dev);
  params.Add("S_adm_dev", S_adm_dev);
  params.Add("Srr_adm_dev", Srr_adm_dev);

  // criterion for bounce in ccsne (O'Connor & Ott 2010)
  params.Add("bounce_density", Constants::BOUNCE_DENS * DensityCGSToCode);
  // set if bounce occurs
  params.Add("post_bounce", false, mutable_param);
  // initial model state
  params.Add("post_bounce_model", post_bounce_model);

  params.Add("outside_pns_threshold", outside_pns_threshold);
  params.Add("inside_pns_threshold", inside_pns_threshold);
  params.Add("net_heat_threshold", net_heat_threshold);
  params.Add("mdot_radii", mdot_radii);

  if (!post_bounce_model) {
    progenitor_pkg->PostStepDiagnosticsMesh = PostStepDiagnostics;
  }

  // Reductions
  auto HstSum = parthenon::UserHistoryOperation::sum;
  auto HstMax = parthenon::UserHistoryOperation::max;
  using History::ReduceInGain;
  using History::ReduceOneVar;
  using parthenon::HistoryOutputVar;
  parthenon::HstVar_list hst_vars = {};
  auto Mgain = [](MeshData<Real> *md) {
    return ReduceInGain<fluid_cons::density>(md, 1, 0);
  };
  auto Qgain = [](MeshData<Real> *md) {
    return ReduceInGain<internal_variables::GcovHeat>(md, 0, 0) -
           ReduceInGain<internal_variables::GcovCool>(md, 0, 0);
  };
  for (auto rc : mdot_radii) {
    auto rc_code = rc * 1e5 * LengthCGSToCode;
    auto Mdot = [rc_code](MeshData<Real> *md) {
      return History::CalculateMdot(md, rc_code, false);
    };
    hst_vars.emplace_back(
        HistoryOutputVar(HstSum, Mdot, "Mdot at r = " + std::to_string(int(rc)) + "km"));
  }

  Real x1max = pin->GetReal("parthenon/mesh", "x1max");
  auto Mdot_gain = [x1max](MeshData<Real> *md) {
    return History::CalculateMdot(md, x1max, true);
  };

  const bool rad_active = pin->GetBoolean("physics", "rad");
  if (rad_active) {
    const bool do_gain_calc = pin->GetOrAddBoolean("radiation", "do_gain_calc", false);
    if (do_gain_calc) {
      hst_vars.emplace_back(HistoryOutputVar(HstSum, Mgain, "Mgain"));
      hst_vars.emplace_back(HistoryOutputVar(HstSum, Qgain, "total net heat"));
      hst_vars.emplace_back(HistoryOutputVar(HstSum, Mdot_gain, "Mdot gain"));
    }
  }
  params.Add(parthenon::hist_param_key, hst_vars);

  return progenitor_pkg;
} // Initialize

TaskStatus GetProgenitorState(MeshData<Real> *md, Real simtime) {

  // bounds
  const auto ib = md->GetBoundsI(IndexDomain::interior);
  const auto jb = md->GetBoundsJ(IndexDomain::interior);
  const auto kb = md->GetBoundsK(IndexDomain::interior);

  // namespaces, etc.
  using parthenon::MakePackDescriptor;
  namespace p = fluid_prim;
  auto *pmb = md->GetParentPointer();
  Mesh *pmesh = md->GetMeshPointer();

  // could re-calc entropy if needed, but requires an extra eos retrieval + call
  auto &resolved_pkgs = pmesh->resolved_packages;
  static auto desc = MakePackDescriptor<p::density, p::entropy>(resolved_pkgs.get());

  auto v = desc.GetPack(md);
  const int nblocks = v.GetNBlocks();

  // reading in parameters/conditions from package
  auto progen = pmb->packages.Get("progenitor").get(); // actual progenitor package
  const auto progenitor_enabled = progen->Param<bool>("enabled");
  auto post_bounce = progen->MutableParam<bool>("post_bounce");

  // we only want to write this file once at bounce!
  if (progenitor_enabled && !(*post_bounce)) {
    const Real bounce_density_crit = progen->Param<Real>("bounce_density");

    Real max_density, min_entropy, bounce_time;
    typename Kokkos::MinMax<Real>::value_type minmax; // is this right?

    parthenon::par_reduce(
        parthenon::LoopPatternMDRange(),
        "Calculates max density and min entropy (pre-bounce) in SNe.", DevExecSpace(), 0,
        nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                      typename Kokkos::MinMax<Real>::value_type &lminmax) {
          // checking for maximum density and minimum entropy
          lminmax.min_val =
              (v(b, p::entropy(), k, j, i) < lminmax.min_val ? v(b, p::entropy(), k, j, i)
                                                             : lminmax.min_val);
          lminmax.max_val =
              (v(b, p::density(), k, j, i) > lminmax.max_val ? v(b, p::density(), k, j, i)
                                                             : lminmax.max_val);
        },
        Kokkos::MinMax<Real>(minmax));

// now we do an MPI reduce if enabled; otherwise get the values directly.
#ifdef MPI_PARALLEL
    MPI_Allreduce(&minmax.max_val, &max_density, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&minmax.min_val, &min_entropy, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#else
    max_density = minmax.max_val;
    min_entropy = minmax.min_val;
#endif // MPI_PARALLEL

    // this could be updated with entropy criteria later (not necessarily needed)
    if (max_density >= bounce_density_crit) {

      bounce_time = simtime; // capture bounce time
      // update in params for continuity (also in case of restart?)
      progen->UpdateParam<bool>("post_bounce", true);

      // let's get some nice (human readable) units
      Mesh *pm = md->GetMeshPointer();
      auto uc = pm->packages.Get("phoebus")->Param<UnitConversions>("unit_conv");

      // output data file, in code units and cgs units
      FILE *fout;
      fout = fopen("bounce.dat", "w"); // questionable naming...
      fprintf(fout, "%-26s  %.14e  %.14e\n", "time [s]", bounce_time,
              uc.GetTimeCodeToCGS() * bounce_time);
      fprintf(fout, "%-26s  %.14e  %.14e\n", "central density [g/cc]", max_density,
              uc.GetMassDensityCodeToCGS() * max_density);
      fprintf(fout, "%-26s  %.14e  %.14e\n", "central entropy [erg/K/g]", min_entropy,
              uc.GetEntropyCodeToCGS() * min_entropy);
      fclose(fout);
    }

    // todo: do we want a file even if bounce isn't reached? just for progen stats?
  }

  return TaskStatus::complete;

} // GetProgenitorState

TaskStatus PostStepDiagnostics(const parthenon::SimTime &time, MeshData<Real> *md) {

  // we'll call our progenitor diagnostics in here for each cycle
  return GetProgenitorState(md, time.time);

} // PostStepDiagnostics

} // namespace Progenitor
