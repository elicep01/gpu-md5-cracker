#!/usr/bin/env bash
#SBATCH -p instruction            # Specify partition (adjust to match available partition)
#SBATCH -t 0-00:30:00             # Time limit (increase if needed)
#SBATCH -J md5_compare            # Job name
#SBATCH -o md5_compare-%j.out     # Output file
#SBATCH -e md5_compare-%j.err     # Error file
#SBATCH -c 16                     # Request 16 CPU cores
#SBATCH --mem=16GB                # Request 16GB of memory
#SBATCH --gres=gpu:1              # Request 1 GPU

set -euo pipefail

# 1) Load modules
echo ">>> Loading modules..."
module purge
module load gcc/13.2.0
module load nvidia/cuda/11.8.0

echo
echo ">>> Patching md5_cpu_6.cpp..."
# 2) Patch illegal 'break' statements in md5_cpu_6.cpp
if [[ -f md5_cpu_6.cpp ]]; then
    sed -i \
        -e 's/if (found.load(std::memory_order_relaxed)) break;/if (found.load(std::memory_order_relaxed)) continue;/' \
        -e 's/            break;/            continue;/' \
        md5_cpu_6.cpp
else
    echo "ERROR: md5_cpu_6.cpp not found!" >&2
    exit 1
fi

echo
echo ">>> Compiling CPU cracker (md5_cpu_6.cpp)..."
# 3) Compile CPU cracker
g++ md5_cpu_6.cpp -O3 -fopenmp -std=c++17 -o cpu_crack

echo
echo ">>> Compiling GPU cracker (gpu_6.cu)..."
# 4) Compile GPU cracker
GPU_SRC=gpu_6.cu
if [[ ! -f "$GPU_SRC" ]]; then
    echo "ERROR: GPU source '$GPU_SRC' not found!" >&2
    exit 1
fi
nvcc "$GPU_SRC" -O3 -std=c++17 \
     -maxrregcount=32 \
     --ptxas-options=-v \
     -arch=sm_70 \
     -o gpu_crack

echo
# 5) Generate random 6-character alphanumeric password (Bash RNG, non-blocking)
# build array of valid characters
pw_chars=( {A..Z} {a..z} {0..9} )
RAND_PW=""
for i in {1..6}; do
    RAND_PW+="${pw_chars[RANDOM % ${#pw_chars[@]}]}"
done
# compute its MD5 hash
echo ">>> Generated random password: $RAND_PW"
TARGET=$(printf "%s" "$RAND_PW" | md5sum | awk '{print $1}')
echo ">>> Using target hash: $TARGET (MD5 of '$RAND_PW')"

echo "=== GPU run (blocks=1024, threads=256) ==="
# 6) Run GPU version FIRST
gpu_out=$(./gpu_crack "$TARGET")
echo "$gpu_out"
gpu_time=$(echo "$gpu_out" | grep -oP 'GPU elapsed\s*:\s*\K[0-9.]+' || echo "")
[[ -z "$gpu_time" ]] && echo "WARNING: Failed to extract GPU time."

echo
echo "=== CPU run (16 threads) ==="
# 7) Run CPU version
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-16}
cpu_out=$(./cpu_crack "$TARGET" -t "$OMP_NUM_THREADS")
echo "$cpu_out"
cpu_time=$(echo "$cpu_out" | grep -oP 'CPU time\s*\(.*?\)?:\s*\K[0-9.]+' || echo "")
[[ -z "$cpu_time" ]] && echo "WARNING: Failed to extract CPU time."

echo
echo ">>> Summary:"
echo "GPU time: ${gpu_time:-N/A} s"
echo "CPU time: ${cpu_time:-N/A} s"
if [[ -n "$cpu_time" && -n "$gpu_time" && "$gpu_time" != "0" ]]; then
    awk -v G="$gpu_time" -v C="$cpu_time" 'BEGIN { printf("Speedup (CPU/GPU): %.2fx\n", C/G) }'
else
    echo "Speedup: N/A (missing timing data)"
fi
