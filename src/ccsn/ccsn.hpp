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

#ifndef CCSN_CCSN_HPP_
#define CCSN_CCSN_HPP_

// Parthenon
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

namespace CCSN {
constexpr int NCCSN = 8;

using State_t = parthenon::ParArray2D<Real>;
using State_host_t = typename parthenon::ParArray2D<Real>::HostMirror;

constexpr int RHO = 0;
constexpr int V = 1;
constexpr int EPS = 2;
constexpr int YE = 3;
constexpr int P = 4;
constexpr int TEMP = 5;
constexpr int grav = 6;
constexpr int entr = 7;

KOKKOS_INLINE_FUNCTION 
Real Get1DProfile(std::string model_filename){

    // open file					    
    std::ifstream inputfile(model_filename);

    // error check    
    if (!inputfile.is_open()) 
        std::cout<<"Error opening file",model_filename;

    const int num_vars = 9; // 8 + radius
    int num_zones = 0;
    std::string line;

    // get number of zones from 1d file
    while(!inputfile.eof()) {
            getline(inputfile, line);
            num_zones ++;
    }

    Real model_1d[num_vars][num_zones];

    // read file into model_1d     
    for (int k = 0; k < num_vars; k++) // number of vars
    {
        for (int j = 0; j < num_zones; j++) //  number of zones
        {
          inputfile >> model_1d[k][j];
        }
    }					  

    std::cout << "Read in file " << model_filename << " 8 variables and " << num_zones << " number of zones.";

    inputfile.close();
    return model_1d[num_vars][num_zones];
}

KOKKOS_INLINE_FUNCTION
Real Interp1DProfile(const Real model_1d[NCCSN][num_zones], const int npoints, const Real r, const Real dr){

    Real model_1d_interp[NCCSN][npoints];

    for (int k = 0, k < num_vars; k++)
    {
	for (int j = 1, j < npoints; j++)  //call interp on a point by point basis here
	{
		// move hunt call here??
	    call quadraticInterp(model_1d[0][:],       // radius from 1D model
			    model_1d[k+1][:],          // k-th variable from 1D model
			    num_zones,                 // total number of zones for 1D model
			    r[j],      // (center) r[j] of phoebus grid want var value at 
			    model_1d_interp[k+1][j]);  // resulting zone averaged var interp value at r[j] 
	}
     }

    return model_1d_interp[NCCSN][npoints];
}

KOKKOS_INLINE_FUNCTION
Real quadraticInterp(const Real input_radius[num_zones], const Real model_1d[num_zones], const int num_zones, const Real r_c, Real var_interp){ 
        Real var_l;
	Real var_c;
	Real var_r;


        call huntInterp(input_radius,num_zones,r_c,low);
       
        

	return 
}

KOKKOS_INLINE_FUNCTION
int huntInterp(const Real input_radius[num_zones], const int num_zones, const Real radius_want, int low){
	step = 1;

	
	return
}




std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

//this task will change to something that inits the CCSN problem
TaskStatus InitializeCCSN(StateDescriptor *ccsnpkg, StateDescriptor *monopolepkg,
                        StateDescriptor *eospkg);

} // namespace CCSN

#endif // CCSN_CCSN_HPP_
