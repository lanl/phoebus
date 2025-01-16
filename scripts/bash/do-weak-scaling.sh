#!/bin/bash

# Set modules and environment variables here
module purge
module load gcc/10.3.0
module load openmpi/4.1.1
module load hdf5-parallel/1.10.7
module load cmake/3.20.3

TIMING_FILE_NAME="timings.dat"

EXEC=./phoebus # executable
INP=torus.pin # input deck
PART=standard # partition
NT_P_NODE=4 # cores per node

# Base grid (smallest size)
NXBASE=64
NX=(${NXBASE} ${NXBASE} ${NXBASE})

# Meshblock sizes to test
MESHBLOCK_SIZES=(8 16 32 64)

# Pack sizes to test
PACK_SIZES=(-1 1 2 4 8 16)

# Walltime needed
WALLTIME="00:30:00"

# Array of powers of 2 from which we will compute core counts
CORE_P_START=0
CORE_P_STOP=10
CORE_POWERS=$(seq ${CORE_P_START} ${CORE_P_STOP})

echo "Saving timing to ${TIMING_FILE_NAME}"
echo "# NX1 NX2 NX3 MB PACK_SIZE ranks Zone-cycles/wallsecond"
echo "# NX1 NX2 NX3 MB PACK_SIZE ranks Zone-cycles/wallsecond" > ${TIMING_FILE_NAME}

# loop
for p in ${CORE_POWERS}; do
    # core count
    count=$((2 ** ${p}))
    echo "Core count = ${count}"
    # node count
    nodes=$((${count} > ${NT_P_NODE} ? ${count}/${NT_P_NODE} : 1))
    echo "Node count = ${nodes}"
    # update base grid. Double appropriately
    d=$(((${p} - 1) % 3))
    if [ ${d} -ge 0 ]; then
        NX[${d}]=$((${d} < 0 ? ${NX[${d}]} : 2 * ${NX[${d}]}))
    fi
    echo "Grid size = ${NX[@]}"
    for nxb in ${MESHBLOCK_SIZES[@]}; do
        echo "Meshblock size = ${nxb}"
        for pack in ${PACK_SIZES[@]}; do
            echo "Pack size = ${pack}"
            # Submit job and run it
            outfile=$(printf "scale-%d-%d-%d.out" ${count} ${nxb} ${pack})
            errfile=$(printf "scale-%d-%d-%d.err" ${count} ${nxb} ${pack})
            echo "saving to output file ${outfile}"
            echo "saving to error file ${errfile}"
            # Some pretty gross subtle weirdness here. Piping the output doesn't work as intended.
            # Either phoebus's input handling redirects the | and > characters, or slurm ignores it or something.
            # So I use slurm's output handling with the --output and --error flags
            # This DOES seem to work as intended.
            ARGS="${EXEC} -i ${INP} parthenon/mesh/nx1=${NX[0]} parthenon/mesh/nx2=${NX[1]} parthenon/mesh/nx3=${NX[2]} parthenon/meshblock/nx1=${nxb} parthenon/meshblock/nx2=${nxb} parthenon/meshblock/nx3=${nxb} parthenon/mesh/pack_size=${pack}"
            CMD="srun --output=${outfile} --error=${errfile} --time=${WALLTIME} --partition ${PART} -n ${count} --nodes ${nodes} ${ARGS}"
            echo ${CMD}
            ${CMD}
            # Pull out performance, and save it to text file.
            # This assumes average performance = average over performance/step.
            # This assumption may be incorrect if you run with I/O.
            zc=$(grep 'zone-cycles/wallsecond = ' ${outfile} | cut -d '=' -f 2 | xargs)
            echo "${NX[0]} ${NX[1]} ${NX[2]} ${nxb} ${pack} ${count} ${zc}"
            echo "${NX[0]} ${NX[1]} ${NX[2]} ${nxb} ${pack} ${count} ${zc}" >> ${TIMING_FILE_NAME}
        done
    done
done
