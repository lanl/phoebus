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

#ifndef MICROPHYSICS_EOS_EOS_HPP_
#define MICROPHYSICS_EOS_EOS_HPP_

#include <memory>
#include <parthenon/package.hpp>

#include <singularity-eos/eos/eos.hpp>

using namespace parthenon::package::prelude;

namespace Microphysics {

//  using MyEOS=singularity::impl::Variant<IdealGas>;

namespace EOS {

#ifdef SPINER_USE_HDF
using UnmodifiedEOS =
    singularity::Variant<singularity::IdealGas, singularity::SpinerEOSDependsRhoT,
                         singularity::SpinerEOSDependsRhoSie,
                         singularity::StellarCollapse>;
#else
using UnmodifiedEOS = singularity::Variant<singularity::IdealGas>;
#endif

using EOS = UnmodifiedEOS;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
} // namespace EOS

} // namespace Microphysics

#endif // MICROPHYSICS_EOS_EOS_HPP_
