#!/usr/bin/env python3
"""
throughput.py – Hash/s micro-benchmark
* CPU: length-4 (≈15 M combos) – always finishes <1 s on 16 threads
* GPU: length-5 (≈916 M combos) – finishes in ~0.15 s on RTX 3060
Outputs CSV + PNG bar-chart in results/
"""
import os, subprocess, csv, datetime, re, pandas as pd, matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXE  = os.path.join(REPO, "src", "md5cracker.exe")
RES  = os.path.join(REPO, "results"); os.makedirs(RES, exist_ok=True)

DIGEST = "ffffffffffffffffffffffffffffffff"    # never found
SETTINGS = {        # per-mode parameters
    "cpu": dict(l=4, extra=["-t", str(os.cpu_count())], timeout=None),
    "gpu": dict(l=5, extra=[],                         timeout=30),
}

def run(mode):
    p = SETTINGS[mode]
    cmd = [EXE, "-d", DIGEST, "-l", str(p["l"]), "-m", mode] + p["extra"]
    proc = subprocess.run(cmd, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=p["timeout"])
    out = proc.stdout
    # GPU prints “… 42.0 Ghash/s”
    m_rate = re.search(r"([\d.]+)\s+Ghash/s", out)
    if m_rate:
        return float(m_rate.group(1))
    # CPU prints “… (0.007 s)”
    m_sec  = re.search(r"\(([\d.]+)\s*s\)", out)
    if m_sec:
        combos = 62 ** p["l"]
        return combos / float(m_sec.group(1)) / 1e9
    print(f"[warn] couldn’t parse {mode} output:\n{out.strip()}")
    return 0.0

def main():
    cpu_gh = run("cpu")
    gpu_gh = run("gpu")

    ts   = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    csvf = os.path.join(RES, f"{ts}_throughput.csv")
    with open(csvf, "w", newline="") as f:
        csv.writer(f).writerows([("mode","ghash_per_s"),
                                 ("cpu", cpu_gh), ("gpu", gpu_gh)])

    print(f"CPU: {cpu_gh:.2f} Ghash/s   GPU: {gpu_gh:.2f} Ghash/s")
    print("CSV  →", csvf)

    # bar chart
    df = pd.DataFrame(dict(mode=["CPU","GPU"], ghash=[cpu_gh, gpu_gh]))
    plt.figure()
    plt.bar(df["mode"], df["ghash"])
    plt.ylabel("Giga-hashes per second")
    plt.title("Raw throughput  (CPU len 4, GPU len 5)")
    plt.grid(axis="y")
    pngf = csvf.replace(".csv", ".png")
    plt.savefig(pngf, dpi=150, bbox_inches="tight")
    plt.close()
    print("Plot →", pngf)

if __name__ == "__main__":
    main()
