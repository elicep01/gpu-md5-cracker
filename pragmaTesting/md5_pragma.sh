#!/usr/bin/env bash
#SBATCH -p instruction
#SBATCH -J md5_pragma
#SBATCH -o md5_pragma.out
#SBATCH -e md5_pragma.err
#SBATCH -c 16

module load nvidia/cuda/11.8.0
g++ md5_pragma.cpp -O3 -fopenmp -std=c++17 -o md5_pragma
./md5_pragma amijan
