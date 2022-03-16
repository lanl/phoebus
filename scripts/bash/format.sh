#!/bin/bash

for f in $(git grep --cached -Il res -- :/*.hpp :/*.cpp); do
    clang-format -i $(git rev-parse --show-toplevel)/${f}
done
