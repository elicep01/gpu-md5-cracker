#!/usr/bin/env bash
#SBATCH -p instruction            # Partition
#SBATCH -t 0-00:10:00             # Time limit
#SBATCH -J gpu_compare            # Job name
#SBATCH -o gpu_compare-%j.out     # Stdout
#SBATCH -e gpu_compare-%j.err     # Stderr
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH -c 16                      # CPU cores for compilation
#SBATCH --mem=16GB                 # Memory

set -euo pipefail

echo ">>> Loading modules..."
module purge
module load gcc/13.2.0
module load nvidia/cuda/11.8.0

# Compile raw-digest GPU cracker
echo
echo ">>> Compiling brute7 GPU cracker (main.cu)..."
nvcc main.cu -O3 -std=c++17 \
     -maxrregcount=32 \
     --ptxas-options=-v \
     -arch=sm_70 \
     -o gpu_crack

# Compile Thrust-based GPU cracker
if [[ -f md5_thrust.cu ]]; then
    echo
    echo ">>> Compiling Thrust GPU cracker (md5_thrust.cu)..."
    nvcc md5_thrust.cu -O3 -std=c++17 \
         -maxrregcount=32 \
         --ptxas-options=-v \
         -arch=sm_70 \
         -o md5_thrust
else
    echo
    echo "Skipping Thrust compilation: 'md5_thrust.cu' not found."
fi

# Generate random 7-character password (Bash RNG)
pw_chars=( {A..Z} {a..z} {0..9} )
RAND_PW=""
for i in {1..7}; do
    RAND_PW+="${pw_chars[RANDOM % ${#pw_chars[@]}]}"
done
# Compute its MD5 hash
TARGET=$(printf "%s" "$RAND_PW" | md5sum | awk '{print $1}')
echo
echo ">>> Test password: $RAND_PW"
echo ">>> Target hash: $TARGET"

# 1) Run brute7
echo
echo "=== Running brute7 GPU cracker ==="
start=$(date +%s.%N)
gpu_out=$(./gpu_crack "$TARGET")
end=$(date +%s.%N)
gpu_time=$(awk "BEGIN {print $end - $start}")
echo "$gpu_out"
echo "brute7 elapsed: ${gpu_time} s"

# 2) Run Thrust variant
echo
echo "=== Running Thrust GPU cracker ==="
start=$(date +%s.%N)
thrust_out=$(./md5_thrust "$TARGET")
end=$(date +%s.%N)
thrust_time=$(awk "BEGIN {print $end - $start}")
echo "$thrust_out"
echo "md5_thrust elapsed: ${thrust_time} s"

# Summary
echo
echo ">>> Summary:"
echo "brute7 time   : ${gpu_time} s"
echo "thrust time   : ${thrust_time} s"
if awk "BEGIN {exit !($thrust_time > 0)}"; then
    speedup=$(awk "BEGIN {printf \"%.2f\", ${gpu_time}/${thrust_time}}")
    echo "Speedup (brute7/thrust): ${speedup}x"
fi
