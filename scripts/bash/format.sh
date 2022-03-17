#!/bin/bash

#------------------------------------------------------------------------------
# © 2022. Triad National Security, LLC. All rights reserved.  This
# program was produced under U.S. Government contract 89233218CNA000001
# for Los Alamos National Laboratory (LANL), which is operated by Triad
# National Security, LLC for the U.S.  Department of Energy/National
# Nuclear Security Administration. All rights in the program are
# reserved by Triad National Security, LLC, and the U.S. Department of
# Energy/National Nuclear Security Administration. The Government is
# granted for itself and others acting on its behalf a nonexclusive,
# paid-up, irrevocable worldwide license in this material to reproduce,
# prepare derivative works, distribute copies to the public, perform
# publicly and display publicly, and to permit others to do so.
#------------------------------------------------------------------------------


: ${CFM:=$(command -v clang-format)}

if ! command -v ${CFM} &> /dev/null; then
    >&2 echo "Error: No clang format found!"
    exit 1
fi

# clang format major version
TARGET_CF_VRSN=12
CF_VRSN=$(${CFM} --version | cut -d ' ' -f 4 | cut -d '.' -f 1)

if [ "${CF_VRSN}" != "${TARGET_CF_VRSN}" ]; then
    >&2 echo "Warning! Your clang format version ${CF_VRSN} is not the same as the pinned version ${TARGET_CF_VRSN}."
    >&2 echo "Results may be unstable."
fi

REPO=$(git rev-parse --show-toplevel)
for f in $(git grep --cached -Il res -- :/*.hpp :/*.cpp); do
    ${CFM} -i ${REPO}/${f}
done
