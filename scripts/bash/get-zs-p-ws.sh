#!/bin/bash

# Used to pull out the zone-cycles/wallsecond from a phoebus or parthenon
# output file. Can be combined with other tools for bash scripting
# usage
# ZC=$(get-zs-p-ws.sh torus.out)
# echo ${ZC}

echo $(grep 'zone-cycles/wallsecond = ' ${1} | cut -d '=' -f 2 | xargs)
