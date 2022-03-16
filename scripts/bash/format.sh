#!/bin/bash

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