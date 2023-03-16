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
#include "microphysics/eos_phoebus/eos_phoebus.hpp"
#include "pgen/pgen.hpp"
#include "phoebus_utils/ascii_reader.hpp"
#include "progenitor/progenitor.hpp"

// Homologously collapsing star.
namespace homologous {
  void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
    std::cout << "Entered into ProbGen" <<std::endl;
  const bool is_monopole_cart =
      (typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::MonopoleCart));
  const bool is_monopole_sph =
      (typeid(PHOEBUS_GEOMETRY) == typeid(Geometry::MonopoleSph));
  PARTHENON_REQUIRE(is_monopole_cart || is_monopole_sph, "Monopole GR Required");
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


  // DO WE NEED THOSE?
  const Real Cv = pin->GetOrAddReal("eos", "Cv", 1.0);
  const Real gamma = pin->GetOrAddReal("eos", "gamma", 1.3334);
  //const bool spherical = pin->GetOrAddReal("homologous", "spherical_coords", true);
  
  auto &coords = pmb->coords;
  auto pmesh = pmb->pmy_mesh;
  int ndim = pmesh->ndim;

  //const Real v_inner = (4. / 3.) * M_PI * std::pow(rinner, 3.);
  //const Real uinner = Eexp / v_inner;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);



  auto eos = pmb->packages.Get("eos")->Param<Microphysics::EOS::EOS>("d.EOS"); 
  auto emin = pmb->packages.Get("eos")->Param<Real>("sie_min");
  auto emax = pmb->packages.Get("eos")->Param<Real>("sie_max");

  std::cout << "about to Interpolate table" <<std::endl;
  // Interpolated Initial Data
  auto progenitor_pkg = pmb->packages.Get("progenitor");
  auto mass_density= progenitor_pkg->Param<Spiner::DataBox>("mass_density");
  auto temp= progenitor_pkg->Param<Spiner::DataBox>("temp");
  auto Ye= progenitor_pkg->Param<Spiner::DataBox>("Ye");
  auto specific_internal_energy= progenitor_pkg->Param<Spiner::DataBox>("specific_internal_energy");
  auto velocity= progenitor_pkg->Param<Spiner::DataBox>("velocity");
  auto pressure= progenitor_pkg->Param<Spiner::DataBox>("pressure");
  

  auto adm_density= progenitor_pkg->Param<Spiner::DataBox>("adm_density");
  auto adm_momentum= progenitor_pkg->Param<Spiner::DataBox>("adm_momentum");
  auto S_adm= progenitor_pkg->Param<Spiner::DataBox>("S_adm");
  auto Srr_adm= progenitor_pkg->Param<Spiner::DataBox>("Srr_adm");


  auto mass_density_dev= progenitor_pkg->Param<Spiner::DataBox>("mass_density_dev");
  auto temp_dev= progenitor_pkg->Param<Spiner::DataBox>("temp_dev");
  auto Ye_dev= progenitor_pkg->Param<Spiner::DataBox>("Ye_dev");
  auto specific_internal_energy_dev= progenitor_pkg->Param<Spiner::DataBox>("specific_internal_energy_dev");
  auto velocity_dev= progenitor_pkg->Param<Spiner::DataBox>("velocity_dev");
  auto pressure_dev = progenitor_pkg->Param<Spiner::DataBox>("pressure_dev");


  auto adm_density_dev= progenitor_pkg->Param<Spiner::DataBox>("adm_density_dev");
  auto adm_momentum_dev= progenitor_pkg->Param<Spiner::DataBox>("adm_momentum_dev");
  auto S_adm_dev= progenitor_pkg->Param<Spiner::DataBox>("S_adm_dev");
  auto Srr_adm_dev= progenitor_pkg->Param<Spiner::DataBox>("Srr_adm_dev");
  
  
  //MonopoleGR
  static bool monopole_initialized = false;
  if (!monopole_initialized) {
  auto monopole_pkg = pmb->packages.Get("monopole_gr");
  auto matter = monopole_pkg->Param<MonopoleGR::Matter_t>("matter");
  auto matter_h = monopole_pkg->Param<MonopoleGR::Matter_host_t>("matter_h");
  auto npoints = monopole_pkg->Param<int>("npoints");
  auto rad = monopole_pkg->Param<MonopoleGR::Radius>("radius");
  
  
  
  for (int i = 0; i < npoints; ++i){
    std::cout<<"rad="<<rad.x(i)<<std::endl;
    matter_h(MonopoleGR::Matter::RHO,i)=adm_density.interpToReal(rad.x(i)); 
    matter_h(MonopoleGR::Matter::J_R,i)=adm_momentum.interpToReal(rad.x(i));
    matter_h(MonopoleGR::Matter::trcS,i)=S_adm.interpToReal(rad.x(i));
    matter_h(MonopoleGR::Matter::Srr,i)=Srr_adm.interpToReal(rad.x(i));
  }
  Kokkos::deep_copy(matter,matter_h);
  MonopoleGR::IntegrateHypersurface(monopole_pkg.get());
  MonopoleGR::LinearSolveForAlpha(monopole_pkg.get());
  MonopoleGR::SpacetimeToDevice(monopole_pkg.get());
  MonopoleGR::DumpToTxt("phoebus-metric-setup.dat",monopole_pkg.get());
  monopole_initialized = true;
  } // if statement
  
  // Primitive Variables
  using Transformation_t = Geometry::SphericalToCartesian;
  auto const &geom_pkg = pmb->packages.Get("geometry");
  auto transform = Geometry::GetTransformation<Transformation_t>(geom_pkg.get());
  pmb->par_for(
      "Phoebus::ProblemGenerator::homologous", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
      Real r;
      const Real x1 = coords.Xc<1>(k, j, i);
      const Real x2 = coords.Xc<2>(k, j, i);
      const Real x3 = coords.Xc<3>(k, j, i);
      Real C[3];
      Real s2c[3][3], c2s[3][3];

      if (is_monopole_sph) {
          r = std::abs(x1);
        }
      else { // Cartesian
          r = std::sqrt(x1 * x1 + x2 * x2 + x3 * x3);
	  transform(x1, x2, x3, C, c2s, s2c);
        }
	
        Real lambda[2];
        if (iye > 0) {
          v(iye, k, j, i) = 0.5;
          lambda[0] = v(iye, k, j, i);
        }

        const Real T = eos.TemperatureFromDensityInternalEnergy(mass_density_dev.interpToReal(r), specific_internal_energy_dev.interpToReal(r), lambda);
        const Real P = eos.PressureFromDensityInternalEnergy(mass_density_dev.interpToReal(r), specific_internal_energy_dev.interpToReal(r), lambda);

	Real vel_vec_in[3]={0};
	Real vel_vec_out[3]={0};
	vel_vec_in[0] = velocity_dev.interpToReal(r);
	if (is_monopole_cart){
	  for ( int i=0; i<3; ++i){
	    for (int j=0; j<3; ++j){
	      vel_vec_out[i]+=s2c[i][j]*vel_vec_in[j];
	    }
	  }
	}
	else {
	  vel_vec_out[0] = vel_vec_in[0];
	}
	
        v(irho, k, j, i) = mass_density_dev.interpToReal(r);
        for (int d = 0 ; d < 3; ++d){
	  v(ivlo+d, k, j, i) = vel_vec_out[d];
	}
	v(iprs, k, j, i) = P;
        v(ieng, k, j, i) = mass_density_dev.interpToReal(r)*specific_internal_energy_dev.interpToReal(r);
        v(itmp, k, j, i) = T;
        v(igm1, k, j, i) = eos.BulkModulusFromDensityTemperature(
                               v(irho, k, j, i), v(itmp, k, j, i), lambda) /
                           v(iprs, k, j, i);
	Real Gammacov[3][3]={0};
	Real vcon[3]={v(ivlo, k, j, i), v(ivlo+1, k, j, i), v(ivlo+2, k, j, i)};
	geom.Metric(CellLocation::Cent, k, j, i, Gammacov);
	Real Gamma = phoebus::GetLorentzFactor(vcon, Gammacov);
	for (int d = 0 ; d < 3; ++d){
	 v(ivlo+d, k, j, i) *= Gamma;
	}
	
      });

  fluid::PrimitiveToConserved(rc.get());
}

} // namespace homologous
