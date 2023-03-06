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


#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "geometry/geometry.hpp"
#include "monopole_gr/monopole_gr.hpp"
#include "pgen/pgen.hpp"
#include "phoebus_utils/ascii_reader.hpp"


// Homologously collapsing star.
namespace homologous {
  void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
    std::cout << "Entered into ProbGen" <<std::endl;
    using namespace AsciiReader;
    PARTHENON_REQUIRE(
		    typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::MonopoleSph), 
      "Problem \"homologous\" requires \"MonopoleSph\" \" geometry!");
  auto &rc = pmb->meshblock_data.Get();
  auto geom=Geometry::GetCoordinateSystem(rc.get());
    
  PackIndexMap imap;
  auto v = rc->PackVariables(
      {fluid_prim::density, fluid_prim::velocity, fluid_prim::energy, fluid_prim::bfield,
       fluid_prim::ye, fluid_prim::pressure, fluid_prim::temperature, fluid_prim::gamma1},
      imap);
  std::cout<< "PackVariablesCompleted"<<std::endl;
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


  
  const Real Cv = pin->GetOrAddReal("eos", "Cv", 1.0);
  const Real gamma = pin->GetOrAddReal("eos", "gamma", 1.3334);
  //const bool spherical = pin->GetOrAddReal("homologous", "spherical_coords", true);
  const std::string table = pin ->GetString("homologous","table");
  auto &coords = pmb->coords;
  auto pmesh = pmb->pmy_mesh;
  int ndim = pmesh->ndim;

  //const Real v_inner = (4. / 3.) * M_PI * std::pow(rinner, 3.);
  //const Real uinner = Eexp / v_inner;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);



  auto eos = pmb->packages.Get("eos")->Param<singularity::EOS>("d.EOS"); 
  auto emin = pmb->packages.Get("eos")->Param<Real>("sie_min");
  auto emax = pmb->packages.Get("eos")->Param<Real>("sie_max");

  std::cout << "about to read table" <<std::endl;
  //Initial Data
  std::vector<double> mydata[11];
  readtable(table,mydata);
  const int npoints = (mydata[0].size()-1)/2+1;

  //MonopoleGR
  auto monopole_pkg = pmb->packages.Get("monopole_gr");
  auto matter = monopole_pkg->Param<MonopoleGR::Matter_t>("matter");
  auto matter_h = monopole_pkg->Param<MonopoleGR::Matter_host_t>("matter_h");
  for (int i = 0; i < npoints; ++i){
    matter_h(MonopoleGR::Matter::RHO,i)=mydata[COLUMNS::rho_adm][2*i];
    matter_h(MonopoleGR::Matter::J_R,i)=mydata[COLUMNS::P_adm][2*i];
    matter_h(MonopoleGR::Matter::trcS,i)=mydata[COLUMNS::S_adm][2*i];
    matter_h(MonopoleGR::Matter::Srr,i)=mydata[COLUMNS::Srr_adm][2*i];
  }
  Kokkos::deep_copy(matter,matter_h);
  MonopoleGR::IntegrateHypersurface(monopole_pkg.get());
  MonopoleGR::LinearSolveForAlpha(monopole_pkg.get());
  MonopoleGR::SpacetimeToDevice(monopole_pkg.get());
  MonopoleGR::DumpToTxt("phoebus-metric-setup.dat",monopole_pkg.get());

  // Primitive Variables
  auto MoveVectorToDevice=[](std::vector<double> &vec){
    Kokkos::View<Real*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> hm (vec.data(),vec.size());
    return Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace::memory_space(),hm);
  };
  auto r_data_d=MoveVectorToDevice(mydata[COLUMNS::rad]);
  auto rho_d=MoveVectorToDevice(mydata[COLUMNS::rho]);
  auto eps_d=MoveVectorToDevice(mydata[COLUMNS::eps]);
  auto vel_d=MoveVectorToDevice(mydata[COLUMNS::vel]);

  pmb->par_for(
      "Phoebus::ProblemGenerator::homologous", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real r = coords.x1v(i);
	int ii = i-ib.s;
	std::cout << r << " , " << r_data_d[2*ii+1] << std::endl;
	PARTHENON_REQUIRE(std::abs(r - r_data_d[2*ii+1]) < 1e-5, "cell centers line up between table and phoebus");

        Real lambda[2];
        if (iye > 0) {
          v(iye, k, j, i) = 0.5;
          lambda[0] = v(iye, k, j, i);
        }

        const Real T = eos.TemperatureFromDensityInternalEnergy(rho_d(2*ii+1), eps_d(2*ii+1), lambda);
        const Real P = eos.PressureFromDensityInternalEnergy(rho_d(2*ii+1), eps_d(2*ii+1), lambda);

        v(irho, k, j, i) = rho_d(2*ii+1);
	v(ivlo, k, j, i) = vel_d(2*ii+1);
	v(iprs, k, j, i) = P;
        v(ieng, k, j, i) = rho_d(2*ii+1)*eps_d(2*ii+1);
        v(itmp, k, j, i) = T;
        v(igm1, k, j, i) = eos.BulkModulusFromDensityTemperature(
                               v(irho, k, j, i), v(itmp, k, j, i), lambda) /
                           v(iprs, k, j, i);
	Real Gammacov[3][3]={0};
	Real vcon[3]={v(ivlo, k, j, i), v(ivlo+1, k, j, i), v(ivlo+2, k, j, i)};
	geom.Metric(CellLocation::Cent, k, j, i, Gammacov);
	Real Gamma = phoebus::GetLorentzFactor(vcon, Gammacov);
	v(ivlo, k, j, i) *= Gamma;
      });
  auto nb = IndexRange{0, v.GetDim(4)-1};
  pmb->par_for_bndry(
  "Phoebus::ProblemGenerator::homologous::inner_x1",
  nb, IndexDomain::inner_x1, false,
  KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
    v(l, k, j, i) = v(l, k, j, ib.s);
  });
  pmb->par_for_bndry(
  "Phoebus::ProblemGenerator::homologous::outer_x1",
  nb, IndexDomain::outer_x1, false,
  KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
    v(l, k, j, i) = v(l, k, j, ib.e);
  });
  fluid::PrimitiveToConserved(rc.get());
}

} // namespace homologous
