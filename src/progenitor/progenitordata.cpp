#include <memory>
#include <vector>

#include "analysis/analysis.hpp"
#include "analysis/history.hpp"
#include "ascii_reader.hpp"
#include "geometry/geometry.hpp"
#include "microphysics/eos_phoebus/eos_phoebus.hpp"
#include "monopole_gr/monopole_gr.hpp"
#include "pgen/pgen.hpp"
#include "progenitordata.hpp"

namespace Progenitor {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto progenitor_pkg = std::make_shared<StateDescriptor>("progenitor");
  Params &params = progenitor_pkg->AllParams();

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

  // Reductions
  const Real rc = pin->GetReal("analysis", "radius_cutoff_mdot");
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
  auto Mdot400 = [rc](MeshData<Real> *md) { return History::CalculateMdot(md, rc, 0); };

  Real x1max = pin->GetReal("parthenon/mesh", "x1max");
  auto Mdot_gain = [x1max](MeshData<Real> *md) {
    return History::CalculateMdot(md, x1max, 1);
  };
  hst_vars.emplace_back(HistoryOutputVar(HstSum, Mgain, "Mgain"));
  hst_vars.emplace_back(HistoryOutputVar(HstSum, Qgain, "total net heat"));
  hst_vars.emplace_back(HistoryOutputVar(HstSum, Mdot400, "Mdot at 400km"));
  hst_vars.emplace_back(HistoryOutputVar(HstSum, Mdot_gain, "Mdot gain"));
  params.Add(parthenon::hist_param_key, hst_vars);

  return progenitor_pkg;
} // Initialize

} // namespace Progenitor
