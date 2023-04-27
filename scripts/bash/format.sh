#!/bin/bash

#------------------------------------------------------------------------------
# © 2022-2023. Triad National Security, LLC. All rights reserved.  This
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


: ${CFM:=clang-format}
: ${PFM:=black}
: ${VERBOSE:=0}

if ! command -v ${CFM} &> /dev/null; then
    >&2 echo "Error: No clang format found! Looked for ${CFM}"
    exit 1
else
    CFM=$(command -v ${CFM})
    echo "Clang format found: ${CFM}"
fi

# clang format major version
TARGET_CF_VRSN=12
CF_VRSN=$(${CFM} --version)
echo "Note we assume clang format version ${TARGET_CF_VRSN}."
echo "You are using ${CF_VRSN}."
echo "If these differ, results may not be stable."

echo "Formatting C++ files..."
REPO=$(git rev-parse --show-toplevel)
for f in $(git grep --untracked -ail res -- :/*.hpp :/*.cpp); do
    if [ ${VERBOSE} -ge 1 ]; then
       echo ${f}
    fi
    ${CFM} -i ${REPO}/${f}
done
echo "...Done"

# format python files
if ! command -v ${PFM} &> /dev/null; then
    >&2 echo "Error: No version of black found! Looked for ${PFM}"
    exit 1
else
    PFM=$(command -v ${PFM})
    echo "black Python formatter found: ${PFM}"
fi

echo "Formatting Python files..."
#directories=("tst/" "script/")
#for dir in "${directories[@]}"; do
#    find "$dir" -name "*.py" -exec ${PFM} {} \;
#done
REPO=$(git rev-parse --show-toplevel)
for f in $(git grep --untracked -ail res -- :/*.py); do
    if [ ${VERBOSE} -ge 1 ]; then
       echo ${f}
    fi
    ${PFM} ${REPO}/${f}
done
echo "...Done"
