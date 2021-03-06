#!/bin/bash

build(){
    rm -rf build/
    mkdir build
    cd build
    cmake ..
    make -j4
}

main() {
    build
    ./ceres-curve-fitting 
    echo "plotting with python"
    python3 ../main.py ceres-output.txt
}

main
