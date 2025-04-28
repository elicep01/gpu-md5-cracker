## Analysis of GPU-Accelerated MD5 Cracker

This document provides an in-depth analysis of our CUDA-based MD5 brute-force cracker developed for **ECE 759: High-Performance Computing (Spring 2025)** at UW–Madison.  
It outlines the baseline implementation, optimization strategies, performance evaluations, and insights gained from profiling using **NVIDIA Nsight Compute**.

---

## Baseline Implementation

### Objective

The primary goal was to implement a brute-force MD5 hash cracker capable of efficiently searching the entire **7-character alphanumeric password space** (62⁷ combinations) using GPU acceleration.

---

### Initial Features

- **CUDA Kernel (`brute7`)**:  
  Each thread computes the MD5 hash of a unique password guess.

- **Early Exit Mechanism**:  
  Utilizes a global `volatile` flag and `atomicCAS` to signal when the correct password is found, allowing other threads to terminate early.

- **Host-Side Output**:  
  Only the host prints the result to avoid per-thread I/O overhead.

- **Performance Measurement**:  
  Basic timing and throughput calculations are included.

---

## Optimization Strategies

To enhance performance, several GPU-specific optimizations were implemented:

### 1. Constant Memory for Lookup Tables

- **Implementation**:  
  Moved the `K[64]` array, containing sine-based constants used in the MD5 algorithm, into **constant memory** (`__constant__ __device__`).

- **Benefit**:  
  Constant memory offers faster access times and is optimized for broadcast to all threads, improving overall kernel performance.

---

### 2. Launch Bounds and Register Limiting

- **Implementation**:  
  Added `__launch_bounds__(256, 4)` to the kernel declaration and compiled with `-maxrregcount=32`.

- **Benefit**:  
  Helps the compiler optimize register usage, potentially increasing occupancy and reducing register spilling.

---

### 3. Memory Optimization

- **Implementation**:  
  Replaced per-thread local arrays (e.g., `char msg[64]`) with on-the-fly computation directly into registers.

- **Benefit**:  
  Reduces local memory usage, which decreases register pressure and improves warp occupancy.

---

### 4. Loop Unrolling

- **Implementation**:  
  Applied `#pragma unroll 64` to the main MD5 processing loop inside the kernel.

- **Benefit**:  
  Encourages the compiler to unroll loops, reducing loop overhead and increasing instruction-level parallelism.

---

## Performance Evaluation

### Benchmarking Results

| Configuration  | Time (s) | Throughput (Ghash/s) |
|----------------|----------|----------------------|
| Baseline GPU   | 0.740    | 1.39                 |
| Optimized GPU  | 0.188    | 5.47                 |

- **Speedup**:  
  The optimized version achieved approximately **4× speedup** over the baseline.

---

### Profiling with NVIDIA Nsight Compute

- **Roofline Analysis**:  
  Used Nsight Compute to generate roofline charts, identifying the kernel's position relative to hardware limits.

- **Findings**:
  - **Arithmetic Intensity**:  
    Increased due to reduced memory accesses and improved computation-to-memory ratio.
  
  - **Occupancy**:  
    Improved by limiting register usage and optimizing memory access patterns.
  
  - **Bottlenecks**:  
    Initial profiling indicated memory-bound performance, which was alleviated through the optimizations.

---

## Insights and Future Work

### Key Takeaways

- **Memory Hierarchy Utilization**:  
  Efficient use of constant memory and elimination of unnecessary local arrays significantly impact performance.

- **Compiler Directives**:  
  Proper use of `__launch_bounds__` and loop unrolling can guide the compiler to generate highly efficient code.

- **Profiling Tools**:  
  Nsight Compute was invaluable for identifying performance bottlenecks and guiding optimization efforts.

---

### Potential Improvements

- **Shared Memory Usage**:  
  Explore using shared memory for intermediate computations to further reduce global memory accesses.

- **Dynamic Parallelism**:  
  Investigate dynamic parallelism (device kernels launched by device code) to manage irregular workloads.

- **Algorithmic Enhancements**:  
  Implement more advanced password generation strategies (e.g., mask-based approaches) to intelligently prune the search space.

---

## Conclusion

Through targeted optimizations and detailed profiling, we significantly improved the performance of our GPU-accelerated MD5 cracker.  
This project demonstrates the importance of understanding GPU architecture and utilizing available tools to achieve high-performance computing goals.

---
