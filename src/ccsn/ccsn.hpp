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
constexpr int NCCSN = 9;

// NVAR:NUM_ZONES
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
std::pair<int, int> Get1DProfileNumZones(const std::string model_filename){

    // open file
    std::ifstream inputfile("input.txt"); // works when harcoded and lives in /bin

    // error check    
    if (!inputfile.is_open()){ 
        std::cout <<  model_filename << " not found :( \n.";
    	exit(-1);
    }

    int nz = 0;
    int nc = 0;
    std::string line;

    // get number of zones from 1d file
    while(!inputfile.eof()){ 
            getline(inputfile, line);
	    if( line.find("//") != std::string::npos ){ 
		nc ++;
	    }
	    else{
            nz ++;
	    }
    }

    inputfile.close();

    return std::make_pair(nz-nc, nc);
}

template <typename HostArray2D>
KOKKOS_INLINE_FUNCTION
void Get1DProfileData(const std::string model_filename, const int num_zones,const int num_comments, HostArray2D &ccsn_state_raw_h){

    std::ifstream inputfile("input.txt"); // works when harcoded and lives in /bin

    const int num_vars = 9; // 8 + radius
    Real val = 0;
    std::string line;

    // read file into model_1d
    for (int i = num_comments; i < num_zones; i++) // number of vars
    {
        for (int j = 0; j < num_vars; j++) //  number of zones
        {
          inputfile >> val;
	  ccsn_state_raw_h(i-num_comments,j) = val;
        }
    }

    std::cout << "Read 1D profile into state array.\n";

    inputfile.close();

    /*
    for (int i = 0; i < num_zones; i++)
    {
        for (int j = 0; j < num_vars; j++)
        {
		std::cout << model_1d[i][j] << "\t";
        }
	std::cout<<std::endl;
    }
    */

    //std::cout << "first cell of radius read in as " << ccsn_state_raw_h(0,0) << "\n.";

}


/*
KOKKOS_INLINE_FUNCTION
Real Interp1DProfile(const Real model_1d[NCCSN][num_zones], const int npoints, const Real r, const Real dr){

    Real model_1d_interp[NCCSN][npoints];

    for (int k = 0, k < num_vars; k++)
    {
	for (int j = 1, j < npoints; j++)  //call interp on a point by point basis here
	{
	    call hunt(input_radius,num_zones,r_c,jlo);
	    //call quadraticInterp(model_1d[0][:],       // radius from 1D model
	    //		    model_1d[k+1][:],          // k-th variable from 1D model
	    //		    num_zones,                 // total number of zones for 1D model
	    //		    r[j],      // (center) r[j] of phoebus grid want var value at 
	    //		    model_1d_interp[k+1][j]);  // resulting zone averaged var interp value at r[j] 
	}
     }

    return model_1d_interp[NCCSN][npoints];
}

KOKKOS_INLINE_FUNCTION
Real quadraticInterp(const Real input_radius[num_zones], const Real model_1d[num_zones], const int num_zones, const Real r_c, Real var_interp){ 
        Real var_l;
	Real var_c;
	Real var_r;
        

	return 
}

// hunt routine - Numerical Recipes in C 2nd Ed
// --------------------------------------------
// Given an array xx[1..n], and given a value x, returns a value jlo such that x is between
// xx[jlo] and xx[jlo+1]. xx[1..n] must be monotonic, either increasing or decreasing.
// jlo=0 or jlo=n is returned to indicate that x is out of range. jlo on input is taken as the
// initial guess for jlo on output. 
KOKKOS_INLINE_FUNCTION
int hunt(const Real xx[num_zones], const int n, const Real x, unsigned long *jlo){
	unsigned long jm,jhi,inc;
	int ascnd;
	ascnd=(xx[n] >= xx[1]);
	if (*jlo <= 0 || *jlo > n) {
	    *jlo=0;
	    jhi=n+1;
	} else {
	    inc=1;
	    if (x >= xx[*jlo] == ascnd) {
		if (*jlo == n) return;
		jhi=(*jlo)+1;
		while (x >= xx[jhi] == ascnd) {
		    *jlo=jhi;
		    inc += inc;
		    jhi=(*jlo)+inc;
		    if (jhi > n) {
			jhi=n+1;
			break;
		    }
		}
	    } else {
		if (*jlo == 1) {
		    *jlo=0;
		    return;
		}
		jhi=(*jlo)--;
		while (x < xx[*jlo] == ascnd) {
		    jhi=(*jlo);
		    inc <<= 1;
		    if (inc >= jhi) {
			*jlo=0;
			break;
		    }
		    else *jlo=jhi-inc;
		}
	    }
        }
	while (jhi-(*jlo) != 1) {
	    jm=(jhi+(*jlo)) >> 1;
	    if (x >= xx[jm] == ascnd)
		*jlo=jm;
	    else
		jhi=jm;
	}
	if (x == xx[n]) *jlo=n-1;
	if (x == xx[1]) *jlo=1;
}

*/

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

//this task will change to something that inits the CCSN problem
TaskStatus InitializeCCSN(StateDescriptor *ccsnpkg, StateDescriptor *monopolepkg,
                        StateDescriptor *eospkg);

} // namespace CCSN

#endif // CCSN_CCSN_HPP_
