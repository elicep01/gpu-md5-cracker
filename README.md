```markdown
# GPU MD5 Cracker 

A high-performance, GPU-accelerated MD5 hash cracker built using CUDA.  
This project is part of our final course project for **ECE 759: High-Performance Computing (Spring 2025)** at UW–Madison.

---

## Project Overview

The goal of this project is to design and implement a parallelized MD5 hash attack using **CUDA** to efficiently brute-force 7-character alphanumeric passwords. Modern GPUs can drastically accelerate hash computation through data-parallelism, and this project demonstrates that speedup against a CPU baseline.

---

## Team Members

- Elice Priyadarshini (epriyadarshi@wisc.edu)  
- Michael Pan (mp mpan4@wisc.edu)  
- Saketh Katta (katta3@wisc.edu)  
- Ankit Mohapatra (amohapatra4@wisc.edu)  
- Fahad Touseef (touseef@wisc.edu)  
- K M Jamiul Haque (khaque@wisc.edu)  

---

## Tech Stack

- **CUDA Toolkit** (v12.x)  
- **C++ / CUDA C**  
- **Thrust & CUB** (for parallel primitives)  
- **OpenMP** (CPU baseline)  
- **Python** (matplotlib & pandas for plotting/benchmarking)  
- **NVIDIA Nsight Compute** (profiling & roofline analysis)  

---

## Project Structure

```text
gpu-md5-cracker-main/
├── README.md               # This file
├── analysis.md             # In-depth methodology & performance analysis
├── src/
│   ├── main.cu             # CUDA MD5 cracker + optimizations
│   ├── md5_cpu.cpp         # CPU baseline (OpenMP)
│   └── md5_cpu.h           # CPU helper definitions
├── bin/
│   ├── gpu_crack.exe       # Compiled GPU executable
│   └── cpu_crack.exe       # Compiled CPU executable
├── benchmark/
│   ├── run.bat             # Batch script to collect timings
│   ├── results.csv         # GPU vs. CPU timing data
│   └── plot_results.py     # Python script to generate bar charts

```

---

## Building

### GPU executable

```bash
cd src
nvcc -O3 -arch=sm_86 main.cu -o ../bin/gpu_crack.exe

# (Optional) Enforce ≤32 registers/thread:
nvcc -O3 -arch=sm_86 -maxrregcount=32 main.cu -o ../bin/gpu_crack.exe
```

### CPU executable

```bash
cd src
cl /O2 /openmp md5_cpu.cpp -o ../bin/cpu_crack.exe
```

> **Note:** Ensure OpenMP and standard C++ libraries are installed.

---

## Usage

### GPU mode

```bash
bin\gpu_crack.exe <32-char MD5 hex>
```

### CPU mode (with optional thread count)

```bash
bin\cpu_crack.exe <32-char MD5 hex> -t 8
```

#### Example

```bash
bin\gpu_crack.exe 5d793fc5b00a2348c3fb9ab59e5ca98a
```

---

## Benchmarking & Profiling

1. **Collect timings**  
   ```bash
   cd benchmark
   run.bat 5d793fc5b00a2348c3fb9ab59e5ca98a 8
   ```
   Outputs `results.csv`.

2. **Plot results**  
   ```bash
   python plot_results.py
   ```

3. **GPU profiling (Nsight Compute)**  
   ```bash
   cd src
   ncu -o brute7_report --set full --target-processes all ../bin/gpu_crack.exe 5d793fc5b00a2348c3fb9ab59e5ca98a
   ncu-ui brute7_report.ncu-rep   # Open GUI for roofline & timeline
   ```

---

## Contributing

Contributions, issues, and feature requests are welcome!  
Please fork the repository and submit a pull request.

---

## License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.
```