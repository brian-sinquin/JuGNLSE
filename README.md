# JuGNLSE.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://brian-sinquin.github.io/JuGNLSE.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://brian-sinquin.github.io/JuGNLSE.jl/dev/)
[![Build Status](https://github.com/brian-sinquin/JuGNLSE.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/brian-sinquin/JuGNLSE.jl/actions/workflows/CI.yml?query=branch%3Amaster)

A high-performance Julia package for solving the **Generalized Nonlinear Schrödinger Equation (GNLSE)** for modeling ultrafast nonlinear pulse propagation in optical fibers.

## Features

- **Multiple Integration Methods**
  - RK4IP (Runge-Kutta 4th order in Interaction Picture) with adaptive stepping
  - Split-Step Fourier Method (SSFM)
  
- **Comprehensive Physics**
  - Higher-order dispersion (arbitrary order via Taylor expansion)
  - Kerr nonlinearity (self-phase modulation)
  - Self-steepening (shock term)
  - Raman scattering with three models:
    - Blow-Wood (1989)
    - Lin-Agrawal (2006) with Boson peak
    - Hollenbeck-Cantrell (2002)
  - Frequency-dependent loss

- **Flexible Pulse Shapes**
  - Hyperbolic secant (sech)
  - Gaussian
  - Continuous wave (CW)
  - Custom envelopes

- **High Performance**
  - Type-stable implementations
  - Pre-planned FFTs for efficiency
  - Optimized for Julia's JIT compiler
  - ~5-10x faster than Python implementations

## Installation

```julia
using Pkg
Pkg.add("JuGNLSE")
```

Or in development mode from GitHub:

```julia
using Pkg
Pkg.develop(url="https://github.com/brian-sinquin/JuGNLSE.jl")
```

## Quick Start

```julia
using JuGNLSE

# Define fiber parameters
fiber = FiberParams(
    0.15,                              # Fiber length [m]
    0.11,                              # Nonlinear coefficient γ [W⁻¹m⁻¹]
    [0.0, 0.0, -11.83e-27, 8.13e-41], # Dispersion coefficients βₙ [psⁿ/m]
    0.0,                               # Loss α [dB/km]
    835e-9                             # Center wavelength λ₀ [m]
)

# Create time-frequency grid
grid = create_grid(
    2^12,      # 4096 points
    10e-12,    # 10 ps time window
    835e-9     # 835 nm center wavelength
)

# Create input pulse (50 fs sech, 10 kW peak power)
pulse = sech_pulse(grid, 50e-15, 10000.0, 835e-9)

# Setup simulation parameters
params = SimParams(
    fiber=fiber,
    N=2^12,
    n_saves=200,              # Number of output points
    raman=true,               # Include Raman effect
    shock=true,               # Include self-steepening
    raman_model=Hollenbeck(), # Raman response model
    fr=0.18                   # Raman fraction
)

# Solve GNLSE
results = solve(pulse, params, method=:rk4ip)

# Access results
z = results.z           # Propagation distances [m]
At = results.At         # Time-domain evolution [N × n_saves]
Aw = results.Aw         # Frequency-domain evolution [N × n_saves]
```

## Examples

### Soliton Propagation

```julia
# First-order soliton (N=1)
fiber = FiberParams(1.0, 0.11, [0.0, 0.0, -11.83e-27], 0.0, 835e-9)
grid = create_grid(2^12, 20e-12, 835e-9)

# Calculate soliton peak power: P₀ = |β₂| / (γ T₀²)
beta2 = -11.83e-27
T0 = 50e-15 / (2 * asinh(1))  # Convert FWHM to T₀
P0 = abs(beta2) / (fiber.gamma * T0^2)

pulse = sech_pulse(grid, 50e-15, P0, 835e-9)
params = SimParams(fiber=fiber, raman=false, shock=false)

results = solve(pulse, params)
```

### Supercontinuum Generation

```julia
# High-power pulse in photonic crystal fiber
fiber = FiberParams(
    0.15,                              # 15 cm
    0.11,                              # High nonlinearity
    [0.0, 0.0, -11.83e-27, 8.13e-41],
    0.0,
    835e-9
)

grid = create_grid(2^13, 12e-12, 835e-9)  # Larger grid for SC
pulse = sech_pulse(grid, 50e-15, 10000.0, 835e-9)

params = SimParams(
    fiber=fiber,
    N=2^13,
    n_saves=200,
    raman=true,
    shock=true,
    raman_model=Hollenbeck()
)

results = solve(pulse, params, method=:rk4ip)
```

### Self-Phase Modulation

```julia
# SPM without dispersion
fiber = FiberParams(0.1, 0.1, [0.0], 0.0, 1550e-9)
grid = create_grid(2^10, 10e-12, 1550e-9)
pulse = gaussian_pulse(grid, 1e-12, 1000.0, 1550e-9)

params = SimParams(fiber=fiber, raman=false, shock=false)
results = solve(pulse, params, method=:ssfm)
```

## Documentation

For detailed documentation, see:
- [User Guide](https://brian-sinquin.github.io/JuGNLSE.jl/stable/)
- [API Reference](https://brian-sinquin.github.io/JuGNLSE.jl/stable/api/)
- [Examples Gallery](https://brian-sinquin.github.io/JuGNLSE.jl/stable/examples/)

## Performance

Typical performance on modern CPU (Intel i7/i9 or AMD Ryzen):

| Grid Size | z-steps | Method | Time    |
|-----------|---------|--------|---------|
| 2^12      | 200     | RK4IP  | ~1-2s   |
| 2^13      | 200     | RK4IP  | ~3-5s   |
| 2^14      | 200     | RK4IP  | ~10-15s |

Compare to Python implementations: **5-10x faster**

### Running Benchmarks

Quick performance check:
```bash
julia quick_test.jl
```

Full benchmark suite:
```bash
julia benchmark/run_benchmarks.jl
```

## Testing

Run the test suite:
```julia
using Pkg
Pkg.test("JuGNLSE")
```

Or run specific test sets:
```bash
# All tests
julia test/run_all_tests.jl

# Quick tests (skip benchmarks)
julia test/run_all_tests.jl --quick

# Plotting tests only
julia test/run_all_tests.jl --plots

# Benchmarks only
julia test/run_all_tests.jl --bench
```

## Related Packages

- [FFTW.jl](https://github.com/JuliaMath/FFTW.jl) - Fast Fourier transforms
- [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) - ODE solvers
- [NonlinearSchrodinger.jl](https://github.com/oashour/NonlinearSchrodinger.jl) - Related NLS solver

## References

1. **Dudley, J. M., Genty, G., & Coen, S.** (2006). Supercontinuum generation in photonic crystal fiber. *Reviews of Modern Physics*, 78(4), 1135.

2. **Agrawal, G. P.** (2012). *Nonlinear Fiber Optics* (5th ed.). Academic Press.

3. **Hult, J.** (2007). A fourth-order Runge–Kutta in the interaction picture method for simulating supercontinuum generation in optical fibers. *Journal of Lightwave Technology*, 25(12), 3770-3775.

4. **Blow, K. J., & Wood, D.** (1989). Theoretical description of transient stimulated Raman scattering in optical fibers. *IEEE Journal of Quantum Electronics*, 25(12), 2665-2673.

5. **Lin, Q., & Agrawal, G. P.** (2006). Raman response function for silica fibers. *Optics Letters*, 31(21), 3086-3088.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use JuGNLSE.jl in your research, please cite:

```bibtex
@software{jugnlse2024,
  author = {Sinquin, Brian},
  title = {JuGNLSE.jl: A Julia Package for Solving the Generalized Nonlinear Schrödinger Equation},
  year = {2024},
  url = {https://github.com/brian-sinquin/JuGNLSE.jl}
}
```

## Acknowledgments

This package was inspired by:
- [gnlse-python](https://github.com/WUST-FOG/gnlse-python) by WUST-FOG
- Original MATLAB implementation from *Supercontinuum Generation in Optical Fibers* (Dudley & Taylor, 2010)
- [xmhk/gnlse](https://github.com/xmhk/gnlse) Python implementation
