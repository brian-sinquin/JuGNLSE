# Benchmark Suite for JuGNLSE.jl

This directory contains comprehensive performance benchmarks for JuGNLSE.jl.

## Running Benchmarks

### Full Benchmark Suite

Run all benchmarks and generate a detailed report:

```bash
julia benchmark/run_benchmarks.jl
```

This will:
- Benchmark all core components
- Compare SSFM vs RK4IP methods
- Analyze scaling with grid size
- Test different physics configurations
- Save results to `benchmark_results.txt`

**Time:** ~5-10 minutes depending on your system

### Quick Benchmarks

For faster testing during development:

```julia
using JuGNLSE, BenchmarkTools

# Benchmark a single solve
fiber = FiberParams(0.1, 0.11, [0.0, 0.0, -11.83e-27], 0.0, 835e-9)
grid = create_grid(2^10, 10e-12, 835e-9)
pulse = sech_pulse(grid, 50e-15, 5000.0, 835e-9)
params = SimParams(fiber=fiber, N=2^10, n_saves=20)

@benchmark solve($pulse, $params, method=:ssfm, progress=false)
```

## Benchmark Categories

### 1. Component Benchmarks
- Grid creation
- Pulse generation (sech, Gaussian)
- FFT operations
- Operator construction (dispersion, Raman)

### 2. Method Comparison
- SSFM vs RK4IP performance
- Accuracy vs speed tradeoffs
- Memory usage comparison

### 3. Scaling Analysis
- Performance vs grid size
- Time per grid point
- Memory scaling

### 4. Physics Effects
- Linear propagation only
- Kerr nonlinearity
- Raman scattering
- Full physics (Raman + shock)

## Performance Targets

Based on modern CPU (Intel i7/i9, AMD Ryzen):

| Operation | Target | Notes |
|-----------|--------|-------|
| Grid creation (2^12) | < 10 ms | One-time setup |
| Pulse generation | < 50 ms | One-time setup |
| FFT (2^12) | < 1 ms | Multiple per step |
| SSFM solve (2^11, 50 steps) | < 500 ms | Fastest method |
| RK4IP solve (2^11, 50 steps) | < 3 s | More accurate |

## Expected Performance

Typical results on modern hardware:

```
Grid creation (2^12):      ~2-5 ms
Pulse generation:          ~10-20 ms
FFT (2^12):                ~0.2-0.5 ms
Dispersion operator:       ~1-3 ms
Raman response:            ~5-15 ms
SSFM solve (2^11):         ~200-400 ms
RK4IP solve (2^11):        ~1-2 s
Full physics (2^11):       ~3-5 s
```

## Interpreting Results

### Timing
- **< 1 ms**: Excellent, sub-millisecond operations
- **1-100 ms**: Good, interactive response
- **100 ms - 1 s**: Acceptable for batch processing
- **> 1 s**: May need optimization for large problems

### Memory
- Look for unexpected allocations
- Pre-allocated arrays should minimize allocations
- FFT plans should be reused

### Scaling
- FFT should scale as O(N log N)
- Overall solve should scale roughly linearly with n_saves
- Memory should scale linearly with N and n_saves

## Profiling

For detailed profiling:

```julia
using Profile, ProfileView

fiber = FiberParams(0.15, 0.11, [0.0, 0.0, -11.83e-27], 0.0, 835e-9)
grid = create_grid(2^11, 10e-12, 835e-9)
pulse = sech_pulse(grid, 50e-15, 10000.0, 835e-9)
params = SimParams(fiber=fiber, N=2^11, n_saves=50)

# Profile
@profile solve(pulse, params, method=:rk4ip, progress=false)

# View results
ProfileView.view()
```

## Optimization Tips

If benchmarks are slower than expected:

1. **Check Julia version**: Use Julia 1.6 or newer
2. **Enable CPU optimizations**: Start Julia with `julia -O3`
3. **Check for type instabilities**: Use `@code_warntype`
4. **Pre-compile**: Let code run once before benchmarking
5. **Reduce allocations**: Look for unnecessary array copies
6. **Use in-place operations**: Functions with `!` suffix
7. **Check grid size**: Powers of 2 are optimal for FFT

## Comparing Results

To compare against previous runs:

```bash
# Save current results
cp benchmark_results.txt benchmark_results_$(date +%Y%m%d).txt

# Run new benchmarks
julia benchmark/run_benchmarks.jl

# Compare
diff benchmark_results_*.txt
```

## Contributing Benchmarks

When adding new features, please:

1. Add relevant benchmarks in `run_benchmarks.jl`
2. Update performance targets if needed
3. Document expected performance
4. Check for performance regressions

## System Information

Record your system info when sharing results:

```julia
using Pkg
Pkg.status()  # Package versions

versioninfo()  # Julia and system info

# CPU info
if Sys.islinux()
    run(`cat /proc/cpuinfo`)
elseif Sys.iswindows()
    run(`wmic cpu get name`)
end
```
