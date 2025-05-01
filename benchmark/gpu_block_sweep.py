"""
gpu_block_sweep.py
------------------
Re-compiles md5cracker with different CUDA block sizes and records
raw GPU throughput (Ghash/s) for each build.

Outputs
-------
results/YYYY-MM-DD_HH-MM_gpu_block_sweep.csv
results/YYYY-MM-DD_HH-MM_gpu_block_sweep.png
"""

import os, subprocess, re, csv, datetime, pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ paths
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXE  = os.path.join(REPO, "src", "md5cracker.exe")
RESULTS_DIR = os.path.join(REPO, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------------ test case
DIGEST  = "ab56b4d92b40713acc5af89985d4b786"   # "abcde"
LENGTH  = "5"
BLOCK_SIZES = [64, 128, 256, 512, 1024]

# ------------------------------------------------------------------ helpers
def build(block):
    """Re-compile md5cracker.exe with a given BLOCK_SIZE value."""
    cmd = [
        "nvcc", "-O3", "-std=c++17", "-Xcompiler", "/openmp",
        f"-DBLOCK_SIZE={block}",
        "src/main.cu", "src/md5_cpu.cpp", "src/cli_driver.cpp",
        "-o", EXE
    ]
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=REPO)   # show nvcc output if it fails

def run():
    """Run the GPU cracker once and parse ‘… Ghash/s’ from stdout."""
    out = subprocess.check_output(
        [EXE, "-d", DIGEST, "-l", LENGTH, "-m", "gpu"],
        text=True, cwd=REPO
    )
    m = re.search(r"([\d.]+)\s+Ghash/s", out)
    return float(m.group(1)) if m else 0.0

# ------------------------------------------------------------------ sweep
rows = []
for b in BLOCK_SIZES:
    print(f"\n=== BLOCK_SIZE={b} ===")
    build(b)
    ghs = run()
    rows.append((b, ghs))
    print(f"{b:4d} threads → {ghs:.2f} Ghash/s")

# ------------------------------------------------------------------ save CSV
stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
csv_path = os.path.join(RESULTS_DIR, f"{stamp}_gpu_block_sweep.csv")
with open(csv_path, "w", newline="") as f:
    csv.writer(f).writerows([("block", "ghash")] + rows)
print("\nCSV  →", csv_path)

# ------------------------------------------------------------------ plot PNG
df = pd.DataFrame(rows, columns=["block", "ghash"])
plt.figure()
plt.plot(df.block, df.ghash, "-o")
plt.xlabel("Threads per block");  plt.ylabel("Ghash/s")
plt.title(f"GPU block-size sweep (len {LENGTH})")
plt.grid(True)
png_path = csv_path.replace(".csv", ".png")
plt.savefig(png_path, dpi=150)
plt.close()
print("Plot →", png_path)
