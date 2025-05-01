#!/usr/bin/env python3
"""
Run CPU and GPU brute-force for lengths 1-7, store CSV + plots.
"""

import os, sys, subprocess, datetime, csv, platform, shutil, re, time
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR   = os.path.join(REPO_ROOT, "src")
EXE       = os.path.join(SRC_DIR, "md5cracker.exe")
RESULTS   = os.path.join(REPO_ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

DIGESTS = {
    1: "0cc175b9c0f1b6a831c399e269772661",   # "a"
    2: "187ef4436122d1cc2f40dc2b92f0eba0",   # "ab"
    3: "900150983cd24fb0d6963f7d28e17f72",   # "abc"
    4: "e2fc714c4727ee9395f324cd2e7f331f",   # "abcd"
    5: "ab56b4d92b40713acc5af89985d4b786",   # "abcde"
    6: "e80b5017098950fc58aad83c8c14978e",   # "abcdef"
    7: "7ac66c0f148de9519b8bd264312c4d64",   # "abcdefg"
}

def build():
    if os.path.exists(EXE): return
    print(">> Building md5cracker.exe …")
    cmd = ["nvcc","-O3","-std=c++17","-Xcompiler","/openmp",
           os.path.join("src","main.cu"),
           os.path.join("src","md5_cpu.cpp"),
           os.path.join("src","cli_driver.cpp"),
           "-o", EXE]
    subprocess.check_call(cmd, cwd=REPO_ROOT)

def run(mode, length, threads=""):
    digest = DIGESTS[length]
    cmd = [EXE, "-d", digest, "-l", str(length), "-m", mode]
    if mode == "cpu" and threads:
        cmd += ["-t", str(threads)]
    t0 = time.time()
    out = subprocess.check_output(cmd, text=True)
    dur = time.time() - t0

    # GPU line: "[GPU] abcde  (0.163104 s, 5.61685 Ghash/s)"
    # CPU line: "[CPU] abcde  (0.326693 s)"
    m_sec  = re.search(r"\(([\d.]+) s", out)
    m_gh   = re.search(r", ([\d.]+) Ghash/s", out)
    sec = float(m_sec.group(1)) if m_sec else dur
    ghs = float(m_gh.group(1))  if m_gh else 0.0
    return sec, ghs

def main():
    build()
    hw_cpu = platform.processor()
    gpu_name = subprocess.check_output(
        ["nvidia-smi","--query-gpu=name","--format=csv,noheader"],
        text=True, stderr=subprocess.DEVNULL).splitlines()[0].strip()

    threads = os.cpu_count()
    rows = []
    print(f"🔧  CPU  ({hw_cpu})  logical cores: {threads}")
    print(f"🖥️  GPU  ({gpu_name})\n")

    for L in range(1,8):
        sec_cpu, _     = run("cpu", L, threads)
        sec_gpu, ghash = run("gpu", L)
        spd = sec_cpu / sec_gpu
        rows.append(dict(length=L, cpu_s=sec_cpu, gpu_s=sec_gpu,
                         speedup=spd, ghash=ghash))
        print(f"L={L}  CPU {sec_cpu:.3f}s   GPU {sec_gpu:.3f}s   "
              f"{spd:4.1f}×   {ghash:.2f} Ghash/s")

    # Save CSV
    today = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    csv_path = os.path.join(RESULTS, f"{today}_bench.csv")
    with open(csv_path,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\n✅  Results saved to {csv_path}")

    # Make plots
    df = pd.DataFrame(rows)
    fig1 = plt.figure()
    plt.plot(df['length'], df['cpu_s'], '-o', label="CPU")
    plt.plot(df['length'], df['gpu_s'], '-o', label="GPU")
    plt.xlabel("Password length"); plt.ylabel("Time (s)")
    plt.title("Brute-force time vs length")
    plt.legend(); plt.grid(True)
    png1 = csv_path.replace(".csv","_time.png")
    plt.savefig(png1, dpi=150)
    plt.close(fig1)

    fig2 = plt.figure()
    plt.bar(df['length'], df['speedup'])
    plt.xlabel("Password length"); plt.ylabel("GPU speed-up (×)")
    plt.title("GPU / CPU speed-up")
    plt.grid(True, axis="y")
    png2 = csv_path.replace(".csv","_speedup.png")
    plt.savefig(png2, dpi=150)
    plt.close(fig2)

    print(f"📈  Plots saved to {png1} and {png2}")

if __name__ == "__main__":
    main()
