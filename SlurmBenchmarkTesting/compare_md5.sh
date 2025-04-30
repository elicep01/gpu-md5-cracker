#!/usr/bin/env bash
#SBATCH -p instruction
#SBATCH -t 0-00:10:00
#SBATCH -J md5_compare
#SBATCH -o md5_compare-%j.out
#SBATCH -e md5_compare-%j.err
#SBATCH -c 8
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1

# 1) Load modules
module purge
module load gcc/13.2.0
module load nvidia/cuda/11.8.0

# 2) Debug: show what .cu/.cpp files you actually have
echo ">>> Source files in $(pwd):"
ls -1 *.cpp *.cu || true
echo

# 3) Patch out illegal breaks in md5_cpu.cpp
sed -i \
  -e 's/if (found.load(std::memory_order_relaxed)) break;/if (found.load(std::memory_order_relaxed)) continue;/' \
  -e 's/            break;/            continue;/' \
  md5_cpu.cpp

# 5) Compile the CPU cracker
echo ">>> Compiling CPU cracker (md5_cpu.cpp)…"
g++ md5_cpu.cpp -O3 -fopenmp -std=c++17 -o cpu_crack

# 6) Compile the GPU cracker
GPU_SRC=main.cu
if [[ ! -f "$GPU_SRC" ]]; then
  echo "ERROR: GPU source '$GPU_SRC' not found!" >&2
  exit 1
fi

echo ">>> Compiling GPU cracker ($GPU_SRC)…"
nvcc "$GPU_SRC" -O3 -std=c++17 \
     -maxrregcount=32 \
     --ptxas-options=-v \
     -arch=sm_70 \
     -o gpu_crack

# 7) Choose the target for "aaaaaaa" (MD5("aaaaaaa"))
TARGET="e0c9035898dd52fc65c41454cec9c4d261"  # MD5("aaaaaaa")

# 8) Run CPU version
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
echo "=== CPU run (${OMP_NUM_THREADS} threads) ==="
cpu_out=$(./cpu_crack $TARGET -t $OMP_NUM_THREADS)
echo "$cpu_out"
cpu_time=$(echo "$cpu_out" | awk '/CPU time/ {print $4}')

# 9) Run GPU version
echo "=== GPU run (blocks=1024, threads=256) ==="
gpu_out=$(./gpu_crack $TARGET)
echo "$gpu_out"
gpu_time=$(echo "$gpu_out" | awk '/GPU elapsed/ {print $3}')

# 10) Summary
echo
echo ">>> Summary:"
echo "CPU time: ${cpu_time} s"
echo "GPU time: ${gpu_time} s"
awk -v C=$cpu_time -v G=$gpu_time 'BEGIN { printf("Speedup: %.2fx\n", C/G) }'
