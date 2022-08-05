#!/bin/bash

echo $(grep 'zone-cycles/wallsecond = ' ${1} | cut -d '=' -f 2 | xargs)
