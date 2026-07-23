# JuGNLSE Benchmarking Methodology

## 1. Introduction

This document outlines the methodology for benchmarking the performance of JuGNLSE, a Julia-based Generalized Nonlinear Schrödinger Equation (GNLSE) solver. The primary goal is to provide a rigorous and reproducible comparison against established industry standards such as `gnlse-python`, Laserfun, and other reference codes (e.g., Travers' code).

## 2. Gold Standard Scenario: Supercontinuum Generation in a Photonic Crystal Fiber (PCF)

To ensure a fair and consistent comparison, a "Gold Standard" simulation scenario has been defined. This scenario focuses on supercontinuum generation in a photonic crystal fiber (PCF) operating in the anomalous dispersion regime. The parameters are derived from a well-documented example in `gnlse-python` to facilitate direct comparison.

### 2.1. Physical and Numerical Parameters

The following parameters will be used for all benchmark simulations:

**Numerical Parameters:**
*   **Temporal Resolution (`N_POINTS`)**: $2^{14}$ points
*   **Time Window (`TIME_WINDOW`)**: 12.5 ps
*   **Save Points Along Fiber (`Z_SAVES`)**: 200 points

**Physical Parameters:**
*   **Central Wavelength (`WAVELENGTH`)**: 835 nm
*   **Fiber Length (`FIBER_LENGTH`)**: 0.15 m
*   **Nonlinearity Coefficient ($\\gamma$) (`NONLINEARITY`)**: 0.11 W⁻¹m⁻¹
*   **Raman Scattering Model**: Equivalent to `gnlse-python`'s `raman_blowwood` model. (Specific implementation in JuGNLSE should mimic this behavior).
*   **Self-Steepening**: Enabled (`SELF_STEEPENING = True`)
*   **Loss (`LOSS`)**: 0 dB/m
*   **Dispersion Coefficients ($\\beta_n$)**: Taylor expansion coefficients up to $\\beta_{10}$ at 835 nm:
    *   $\beta_2$: -11.830e-3 ps²m⁻¹
    *   $\beta_3$: 8.1038e-5 ps³m⁻¹
    *   $\beta_4$: -9.5205e-8 ps⁴m⁻¹
    *   $\beta_5$: 2.0737e-10 ps⁵m⁻¹
    *   $\beta_6$: -5.3943e-13 ps⁶m⁻¹
    *   $\beta_7$: 1.3486e-15 ps⁷m⁻¹
    *   $\beta_8$: -2.5495e-18 ps⁸m⁻¹
    *   $\beta_9$: 3.0524e-21 ps⁹m⁻¹
    *   $\beta_{10}$: -1.7140e-24 ps¹⁰m⁻¹

**Input Pulse Parameters (Sech Envelope):**
*   **Peak Power (`PEAK_POWER`)**: 10000 W
*   **Pulse Duration (FWHM) (`DURATION`)**: 0.050 ps (sech-shaped)

## 3. Benchmarking Implementation (JuGNLSE)

### 3.1. Benchmark Script

The benchmark will be implemented in `test/benchmark.jl`. This script will:
1.  Load necessary JuGNLSE modules and `BenchmarkTools.jl`.
2.  Define the "Gold Standard" parameters.
3.  Perform a "warm-up" run of the simulation to ensure Julia's Just-In-Time (JIT) compiler has optimized the code, thereby providing more accurate performance measurements.
4.  Execute the core simulation using `@benchmarkable` from `BenchmarkTools.jl` to collect timing and memory allocation data.
    *   `samples`: Number of times the benchmark expression is evaluated (e.g., 10).
    *   `evals`: Number of times the benchmark expression is evaluated per sample (e.g., 1).

### 3.2. Performance Metrics

The following metrics will be collected for JuGNLSE:
*   **Minimum Time**: The fastest execution time recorded.
*   **Mean Time**: The average execution time.
*   **Median Time**: The median execution time.
*   **Total Allocs**: Total memory allocations (bytes).
*   **GC Time**: Time spent in garbage collection.

## 4. Comparative Methodology

### 4.1. Comparison with Reference Codes

Results from JuGNLSE will be compared against:
*   **`gnlse-python`**: Direct comparison using the identical "Gold Standard" parameters.
*   **Laserfun (if applicable)**: If a compatible scenario can be configured.
*   **Travers' code (if applicable)**: If a compatible scenario can be configured.

Comparison will involve both quantitative performance metrics (execution time, memory usage) and qualitative accuracy assessment (spectral and temporal profiles of the supercontinuum).

### 4.2. Reporting Format

Benchmark results will be presented in a clear, standardized format within this `docs/benchmarks.md` file (or linked sub-documents). Each benchmark entry will include:

*   **Date of Benchmark**: `YYYY-MM-DD`
*   **JuGNLSE Version/Commit**: Git hash or version number.
*   **Hardware/Software Environment**: CPU, RAM, OS, Julia version, relevant library versions.
*   **Simulation Parameters**: A reiteration of the "Gold Standard" parameters used.
*   **Performance Results (JuGNLSE)**:
    *   Minimum Time: `X.XX s`
    *   Mean Time: `X.XX s`
    *   Median Time: `X.XX s`
    *   Total Allocs: `X.XX MiB`
    *   GC Time: `X.XX %`
*   **Comparative Results (vs. `gnlse-python`, etc.)**:
    *   Reference Code Version: (e.g., `gnlse-python vX.X.X`)
    *   Reference Code Performance: (Similar metrics as above, if available).
    *   Performance Ratio: (e.g., `JuGNLSE is Yx faster/slower than gnlse-python`).
*   **Accuracy Assessment (Qualitative)**:
    *   Comparison plots (spectral and temporal profiles) showing agreement/discrepancies.
    *   Discussion of any observed differences in output.

This structured approach ensures that future benchmarks can be easily added and compared, maintaining a consistent record of JuGNLSE's performance and accuracy evolution.
